import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchcrf import CRF
    HAS_CRF = True
except ImportError:
    HAS_CRF = False


class FeatureEncoder(nn.Module):
    def __init__(self, d_in=4096, d_model=128):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_in)
        d_mid = max(d_model * 2, 256)
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_mid),
            nn.GELU(),
            nn.LayerNorm(d_mid),
            nn.Dropout(0.1),
            nn.Linear(d_mid, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, hidden_states):
        x = self.norm_in(hidden_states.to(torch.float32))
        return self.proj(x).unsqueeze(1)


class TCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.trim = padding

    def forward(self, x):
        residual = x
        out = x.transpose(1, 2)
        out = self.conv1(out)
        if self.trim > 0:
            out = out[:, :, :-self.trim]
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        out = self.conv2(out)
        if self.trim > 0:
            out = out[:, :, :-self.trim]
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + residual


class TCNStyleExtractor(nn.Module):
    def __init__(self, d_model=128, num_levels=4, kernel_size=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.tcn_layers = nn.ModuleList([
            TCNBlock(d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_levels)
        ])
        self.attn_pool = nn.Sequential(nn.Linear(d_model, 1))

    def forward(self, token_probs, entropies):
        S = torch.stack([token_probs, entropies], dim=-1)
        x = self.input_proj(S)
        for tcn in self.tcn_layers:
            x = tcn(x)
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)
        z_style = (x * attn_weights).sum(dim=1, keepdim=True)
        return z_style


class EnhancedFrequencyExtractor(nn.Module):
    def __init__(self, max_word_len=96, d_freq=32):
        super().__init__()
        self.max_word_len = max_word_len
        self.freq_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, d_freq),
        )

    def forward(self, token_probs, valid_word_masks=None):
        surprisal = -torch.log2(token_probs + 1e-9)
        if valid_word_masks is None:
            valid_word_masks = (token_probs > 1e-8).float()
        sum_s = (surprisal * valid_word_masks).sum(dim=-1, keepdim=True)
        count = valid_word_masks.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_s = sum_s / count
        centered = (surprisal - mean_s) * valid_word_masks
        L = centered.shape[-1]
        window = torch.hann_window(L, device=centered.device)
        windowed = centered * window
        fft = torch.fft.rfft(windowed, dim=-1)
        power = torch.abs(fft) ** 2
        n_bins = power.shape[-1]
        third = max(1, n_bins // 3)
        total_power = power.sum(dim=-1, keepdim=False) + 1e-9
        low = power[:, :third].sum(dim=-1) / total_power
        mid = power[:, third:2 * third].sum(dim=-1) / total_power
        high = power[:, 2 * third:].sum(dim=-1) / total_power
        power_norm = power / (total_power.unsqueeze(-1) + 1e-9)
        spectral_entropy = -(power_norm * torch.log(power_norm + 1e-9)).sum(dim=-1)
        freq_feat = torch.stack([low, mid, high, spectral_entropy], dim=-1)
        return self.freq_mlp(freq_feat)


class DualCrossAttention(nn.Module):
    def __init__(self, d_model, d_diveye=4, d_freq=32, d_compress=96,
                 branch_dropout=0.0, use_freq=True):
        super().__init__()
        self.d_compress = d_compress
        self.branch_dropout = branch_dropout
        self.use_freq = use_freq

        self.compressor = nn.Sequential(
            nn.Linear(d_model, d_compress),
            nn.LayerNorm(d_compress),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.attn1 = nn.MultiheadAttention(
            d_compress, num_heads=4, batch_first=True, dropout=0.1)
        self.attn2 = nn.MultiheadAttention(
            d_compress, num_heads=4, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(d_compress)
        self.fusion = nn.Linear(2 * d_compress, d_compress)

        self.diveye_proj = nn.Sequential(
            nn.Linear(d_diveye, d_compress),
            nn.GELU(),
            nn.LayerNorm(d_compress),
        )
        if use_freq:
            self.freq_proj = nn.Sequential(
                nn.Linear(d_freq, d_compress),
                nn.GELU(),
                nn.LayerNorm(d_compress),
            )
            n_branches = 3
        else:
            n_branches = 2

        self.final_fusion = nn.Sequential(
            nn.Linear(d_compress * n_branches, d_compress),
            nn.GELU(),
            nn.LayerNorm(d_compress),
        )

    def _branch_drop(self, tensor):
        if self.training and self.branch_dropout > 0:
            if torch.rand(1).item() < self.branch_dropout:
                return torch.zeros_like(tensor)
        return tensor

    def forward(self, z_style, z_ins, diveye_features, freq_features=None):
        z_style_c = self.compressor(z_style)
        z_ins_c = self.compressor(z_ins)

        z_style_c = self._branch_drop(z_style_c)
        z_ins_c_dropped = self._branch_drop(z_ins_c)

        a1, _ = self.attn1(z_style_c, z_ins_c_dropped, z_ins_c_dropped)
        a2, _ = self.attn2(z_ins_c_dropped, z_style_c, z_style_c)
        z_cross = self.fusion(torch.cat([a1, a2], dim=-1))
        z_cross = self.cross_norm(z_cross + z_style_c)

        dv_proj = self._branch_drop(
            self.diveye_proj(diveye_features).unsqueeze(1))

        if self.use_freq and freq_features is not None:
            fq_proj = self._branch_drop(
                self.freq_proj(freq_features).unsqueeze(1))
            combined = torch.cat([z_cross, dv_proj, fq_proj], dim=-1)
        else:
            combined = torch.cat([z_cross, dv_proj], dim=-1)

        combined = self.final_fusion(combined)
        return combined


class HybridGCN(nn.Module):
    def __init__(self, d_in, num_layers=2, dropout=0.15,
                 use_hybrid_adj=False, sem_topk=3):
        super().__init__()
        self.num_layers = num_layers
        self.use_hybrid_adj = use_hybrid_adj
        self.sem_topk = sem_topk
        if use_hybrid_adj:
            self.alpha_logit = nn.Parameter(torch.tensor(2.0))
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(d_in, d_in))
            self.norms.append(nn.LayerNorm(d_in))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_position_adj(S, device):
        A = torch.eye(S, device=device)
        if S > 1:
            idx = torch.arange(S, device=device)
            A[idx[:-1], idx[1:]] = 1.0
            A[idx[1:], idx[:-1]] = 1.0
        if S > 2:
            idx = torch.arange(S, device=device)
            A[idx[:-2], idx[2:]] = 0.5
            A[idx[2:], idx[:-2]] = 0.5
        D_mat = A.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return A / D_mat

    def _build_semantic_adj(self, x):
        B, S, D = x.shape
        x_norm = F.normalize(x, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
        eye = torch.eye(S, device=x.device).unsqueeze(0).expand(B, -1, -1)
        sim_no_self = sim - eye * 1e9

        k = min(self.sem_topk, S - 1) if S > 1 else 1
        if k <= 0:
            return eye
        topk_vals, topk_idx = sim_no_self.topk(k, dim=-1)
        A_sem = torch.zeros_like(sim)
        A_sem.scatter_(-1, topk_idx, F.relu(topk_vals))
        A_sem = 0.5 * (A_sem + A_sem.transpose(1, 2))
        A_sem = A_sem + eye
        D_mat = A_sem.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return A_sem / D_mat

    def forward(self, x):
        B, S, D = x.shape
        A_pos = self._build_position_adj(S, x.device)

        if self.use_hybrid_adj:
            A_sem = self._build_semantic_adj(x)
            alpha = torch.sigmoid(self.alpha_logit)
            A_pos_b = A_pos.unsqueeze(0).expand(B, -1, -1)
            A_combined = alpha * A_pos_b + (1 - alpha) * A_sem
        else:
            A_combined = A_pos.unsqueeze(0).expand(B, -1, -1)

        for i in range(self.num_layers):
            residual = x
            ax = torch.bmm(A_combined, x)
            x = self.layers[i](ax)
            x = self.norms[i](x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = x + residual
        return x


class MLPClassifier(nn.Module):
    def __init__(self, d_feat, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.GELU(),
            nn.LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, num_classes),
        )

    def forward(self, z):
        return self.net(z)


class SenFlowNet(nn.Module):
    def __init__(
        self,
        d_in=4096,
        d_model=128,
        d_diveye=4,
        d_freq=32,
        d_compress=96,
        use_crf=False,
        max_seq_len=20,
        gcn_layers=2,
        gcn_dropout=0.15,
        branch_dropout=0.0,
        head_dropout=0.2,
        use_hybrid_adj=False,
        use_position_aux=False,
        ablate_tcn=False,
        ablate_freq=False,
        ablate_gcn=False,
        ablate_cl=False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.use_crf = use_crf and HAS_CRF
        self.ablate_tcn = ablate_tcn
        self.ablate_freq = ablate_freq
        self.ablate_gcn = ablate_gcn
        self.use_position_aux = use_position_aux

        self.encoder = FeatureEncoder(d_in=d_in, d_model=d_model)

        if ablate_tcn:
            self.style_mlp = nn.Sequential(
                nn.Linear(2, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.style = TCNStyleExtractor(d_model=d_model, num_levels=4)

        self.use_freq = not ablate_freq
        if self.use_freq:
            self.freq_extractor = EnhancedFrequencyExtractor(
                max_word_len=96, d_freq=d_freq)

        self.fusion = DualCrossAttention(
            d_model, d_diveye, d_freq, d_compress,
            branch_dropout=branch_dropout,
            use_freq=self.use_freq,
        )

        d_total = d_compress

        if not ablate_gcn:
            self.gcn = HybridGCN(
                d_total, num_layers=gcn_layers, dropout=gcn_dropout,
                use_hybrid_adj=use_hybrid_adj,
            )

        self.classifier = MLPClassifier(
            d_feat=d_total, num_classes=2, dropout=head_dropout)

        if use_position_aux:
            self.position_head = nn.Sequential(
                nn.Linear(d_total, d_total // 2),
                nn.GELU(),
                nn.LayerNorm(d_total // 2),
                nn.Dropout(head_dropout),
                nn.Linear(d_total // 2, 3),
            )

        if self.use_crf:
            self.crf = CRF(2, batch_first=True)

        self.dropout = nn.Dropout(head_dropout)

    def encode_to_zseq(self, hidden_states, token_probs, entropies,
                       diveye_features):
        B, S = hidden_states.shape[0], hidden_states.shape[1]
        hs_flat = hidden_states.view(B * S, -1)
        tp_flat = token_probs.view(B * S, -1)
        en_flat = entropies.view(B * S, -1)
        dv_flat = diveye_features.view(B * S, -1)

        z_ins = self.encoder(hs_flat)
        if self.ablate_tcn:
            style_in = torch.stack([tp_flat, en_flat], dim=-1)
            z_style = self.style_mlp(style_in).mean(dim=1, keepdim=True)
        else:
            z_style = self.style(tp_flat, en_flat)

        if self.use_freq:
            freq_feats = self.freq_extractor(tp_flat)
        else:
            freq_feats = None

        combined = self.fusion(z_style, z_ins, dv_flat, freq_feats)
        z_fused = combined.squeeze(1)
        z_seq = z_fused.view(B, S, -1)

        z_style_c = self.fusion.compressor(z_style).squeeze(1)
        z_style_seq = z_style_c.view(B, S, -1)
        return z_seq, z_style_seq

    def forward_from_zseq(self, z_seq, pad_mask=None, labels=None):
        B, S = z_seq.shape[0], z_seq.shape[1]

        if self.ablate_gcn:
            z_gcn = z_seq
        else:
            z_gcn = self.gcn(z_seq)

        z_gcn_flat = self.dropout(z_gcn.view(B * S, -1))
        logits_flat = self.classifier(z_gcn_flat)
        emissions = logits_flat.view(B, S, 2)

        position_logits = None
        if self.use_position_aux:
            position_logits = self.position_head(z_gcn_flat).view(B, S, 3)

        crf_loss = None
        if self.use_crf and labels is not None and pad_mask is not None:
            mask_bool = pad_mask.bool()
            crf_loss = -self.crf(emissions, labels, mask=mask_bool,
                                 reduction='mean')

        if self.use_crf and not self.training and pad_mask is not None:
            predictions = self.crf.decode(emissions, mask=pad_mask.bool())
        else:
            predictions = torch.argmax(emissions, dim=-1).cpu().numpy().tolist()

        return emissions, position_logits, crf_loss, predictions

    def forward(self, hidden_states, token_probs, entropies, diveye_features,
                pad_mask=None, labels=None):
        z_seq, z_style_seq = self.encode_to_zseq(
            hidden_states, token_probs, entropies, diveye_features)
        emissions, position_logits, crf_loss, predictions = \
            self.forward_from_zseq(z_seq, pad_mask=pad_mask, labels=labels)
        return predictions, emissions, z_style_seq, crf_loss, position_logits
