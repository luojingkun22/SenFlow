import argparse
import copy
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

from SenFlow_Net import SenFlowNet
from style_cl_loss import StyleContrastiveLoss

SEED = 42

D_MODEL = 128
D_DIVEYE = 4
D_FREQ = 32
D_COMPRESS = 96
GCN_LAYERS = 3
GCN_DROPOUT = 0.15
BRANCH_DROPOUT = 0.15
HEAD_DROPOUT = 0.2

MAX_SEQ_LEN = 20
EPOCHS = 50
BATCH_SIZE = 48
LR = 3e-4
WEIGHT_DECAY = 0.01
LR_WARMUP_EPOCHS = 3

ALPHA_CL = 0.05
ALPHA_CRF = 0.5
ALPHA_POS = 0.1
CL_WARMUP_EPOCHS = 2

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.5
LABEL_SMOOTHING = 0.10

CL_TEMPERATURE = 0.07

EMA_DECAY = 0.999

MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5

AWP_LR = 1e-3
AWP_EPS = 1e-3
AWP_START_EPOCH = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, label_smoothing=0.10):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        return alpha_t * (1 - pt) ** self.gamma * ce_loss


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone().detach()
            for n, p in model.named_parameters() if p.requires_grad
        }

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(
                    p.data, alpha=1.0 - self.decay)

    def apply_to(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if p.requires_grad and n in backup:
                p.data.copy_(backup[n])


class AWP:
    def __init__(self, model, lr=1e-3, eps=1e-3):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.backup = {}

    def attack_step(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and 'weight' in name:
                self.backup[name] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm > 1e-9:
                    r = self.lr * p.grad / (norm + 1e-9)
                    r = torch.clamp(r, -self.eps, self.eps)
                    p.data.add_(r)

    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def mixup_zseq(z_seq, padded_labels, pad_mask, alpha=0.2):
    B = z_seq.shape[0]
    lam = max(float(np.random.beta(alpha, alpha)), 0.5) if alpha > 0 else 1.0
    perm = torch.randperm(B, device=z_seq.device)
    mixed_z = lam * z_seq + (1 - lam) * z_seq[perm]
    return mixed_z, lam, perm


def make_position_labels(pad_mask):
    B, S = pad_mask.shape
    pos_labels = torch.full(
        (B, S), -100, dtype=torch.long, device=pad_mask.device)
    valid_count = pad_mask.sum(dim=-1).long()
    for i in range(B):
        n = int(valid_count[i].item())
        if n <= 0:
            continue
        pos_labels[i, :n] = 1
        pos_labels[i, 0] = 0
        if n > 1:
            pos_labels[i, n - 1] = 2
    return pos_labels


def load_cached_features(cache_path):
    print(f"[INFO] Loading: {cache_path}")
    data = torch.load(cache_path, weights_only=False)
    metadata = data.get("metadata", {})
    train = data.get("train", [])
    val = data.get("val", [])
    test = data.get("test", [])
    print(f"       Domain: {metadata.get('domain', '?')} | "
          f"Generator: {metadata.get('generator', '?')}")
    print(f"       Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test, metadata


def load_multi_cached(cache_paths):
    all_train, all_val, all_test, meta_list = [], [], [], []
    for p in cache_paths:
        t, v, ts, m = load_cached_features(p)
        all_train.extend(t)
        all_val.extend(v)
        all_test.extend(ts)
        meta_list.append(m)
    merged_meta = {
        "merged_from": [
            m.get("domain", "?") + "_" + m.get("generator", "?")
            for m in meta_list
        ],
        "num_sources": len(cache_paths),
    }
    print(f"\n[MULTI-LOAD] Merged {len(cache_paths)} sources")
    print(f"       Total: train={len(all_train)}  val={len(all_val)}  "
          f"test={len(all_test)}")
    return all_train, all_val, all_test, merged_meta


def get_batch(batch_data, device, max_seq_len):
    B = len(batch_data)
    hidden_states = torch.stack(
        [d["hidden_states"] for d in batch_data]).to(device)
    target_probs = torch.stack(
        [d["target_probs"] for d in batch_data]).to(device)
    entropies = torch.stack(
        [d["entropies"] for d in batch_data]).to(device)
    diveye_feats = torch.stack(
        [d["diveye_feats"] for d in batch_data]).to(device)
    pad_mask = torch.stack(
        [d["pad_mask"] for d in batch_data]).to(device)

    labels_list = [d["labels"] for d in batch_data]
    padded_labels = torch.zeros((B, max_seq_len), dtype=torch.long).to(device)
    for j, seq in enumerate(labels_list):
        seq_len = min(len(seq), max_seq_len)
        padded_labels[j, :seq_len] = torch.tensor(
            seq[:seq_len], dtype=torch.long)
    return (hidden_states, target_probs, entropies, diveye_feats,
            pad_mask, padded_labels, labels_list)


def compute_metrics(true_labels_list, emissions_tensor, mask_tensor,
                    search_thresh=True, fixed_thresh=0.5):
    flat_true, flat_probs = [], []
    probs_tensor = torch.softmax(emissions_tensor, dim=-1)[:, :, 1]
    for i in range(len(true_labels_list)):
        seq_len = int(mask_tensor[i].sum().item())
        flat_true.extend(true_labels_list[i][:seq_len])
        flat_probs.extend(
            probs_tensor[i, :seq_len].detach().cpu().numpy().tolist())
    flat_true = np.array(flat_true)
    flat_probs = np.array(flat_probs)
    auc = (roc_auc_score(flat_true, flat_probs)
           if len(np.unique(flat_true)) > 1 else 0.0)
    best_f1, best_thresh = 0.0, fixed_thresh
    if search_thresh:
        for thresh in np.arange(0.05, 0.95, 0.01):
            preds = (flat_probs >= thresh).astype(int)
            f1 = f1_score(flat_true, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
    else:
        preds = (flat_probs >= fixed_thresh).astype(int)
        best_f1 = f1_score(flat_true, preds, average='macro', zero_division=0)
    best_preds = (flat_probs >= best_thresh).astype(int)
    mcc = matthews_corrcoef(flat_true, best_preds)
    return {"F1": best_f1, "AUC": auc, "MCC": mcc}, best_thresh


@torch.no_grad()
def evaluate_set(model, dataset, device, batch_size, max_seq_len):
    model.eval()
    all_emissions, all_masks, all_labels = [], [], []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        (hidden_states, target_probs, entropies, diveye_feats,
         pad_mask, padded_labels, labels_list) = get_batch(
            batch, device, max_seq_len)
        _, emissions, _, _, _ = model(
            hidden_states, target_probs, entropies, diveye_feats,
            pad_mask=pad_mask,
        )
        all_emissions.append(emissions)
        all_masks.append(pad_mask)
        all_labels.extend(labels_list)
    return (torch.cat(all_emissions, dim=0),
            torch.cat(all_masks, dim=0), all_labels)


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(
            1, total_epochs - warmup_epochs)
        return (eta_min / optimizer.defaults['lr']
                + (1 - eta_min / optimizer.defaults['lr'])
                * 0.5 * (1 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_param_groups(model, weight_decay):
    no_decay = ['bias']
    return [
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]


def train_model(model, train_data, val_data, save_path, cfg):
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    lr = cfg['lr']
    wd = cfg['weight_decay']
    alpha_cl = cfg['alpha_cl']
    alpha_crf = cfg['alpha_crf']
    alpha_pos = cfg['alpha_pos']
    cl_warmup = cfg['cl_warmup_epochs']
    lr_warmup = cfg['lr_warmup_epochs']
    max_seq_len = cfg['max_seq_len']
    cl_temp = cfg['cl_temperature']
    use_crf = cfg['use_crf']
    grad_clip = cfg['grad_clip']
    focal_alpha = cfg['focal_alpha']
    focal_gamma = cfg['focal_gamma']
    label_smoothing = cfg['label_smoothing']
    use_ema = cfg['use_ema']
    ema_decay = cfg['ema_decay']
    use_mixup = cfg['use_mixup']
    mixup_alpha = cfg['mixup_alpha']
    mixup_prob = cfg['mixup_prob']
    use_awp = cfg['use_awp']
    awp_lr_p = cfg['awp_lr']
    awp_eps = cfg['awp_eps']
    awp_start = cfg['awp_start_epoch']
    use_position = cfg['use_position_aux']

    focal_fn = FocalLoss(
        gamma=focal_gamma, alpha=focal_alpha,
        label_smoothing=label_smoothing)
    cl_fn = StyleContrastiveLoss(temperature=cl_temp).to(device)

    param_groups = get_param_groups(model, wd)
    optimizer = optim.AdamW(param_groups, lr=lr)
    scheduler = get_lr_scheduler(optimizer, lr_warmup, epochs)

    ema = EMA(model, decay=ema_decay) if use_ema else None
    awp = AWP(model, lr=awp_lr_p, eps=awp_eps) if use_awp else None

    best_val_f1 = 0.0
    best_thresh = 0.5
    patience = 0
    max_patience = 10

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{epochs} "
              f"(lr={current_lr:.2e}) {'=' * 20}")

        model.train()
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        shuffled_train = [train_data[i] for i in indices]
        train_loss_total = 0.0
        num_batches = math.ceil(len(shuffled_train) / batch_size)

        for batch_idx_raw in range(0, len(shuffled_train), batch_size):
            batch = shuffled_train[batch_idx_raw:batch_idx_raw + batch_size]
            (hidden_states, token_probs, entropies, diveye_feats,
             pad_mask, padded_labels, _) = get_batch(
                batch, device, max_seq_len)

            valid_mask = pad_mask.bool()
            do_mixup = (use_mixup and epoch >= cl_warmup
                        and np.random.rand() < mixup_prob)

            optimizer.zero_grad()

            z_seq, z_style_seq = model.encode_to_zseq(
                hidden_states, token_probs, entropies, diveye_feats)
            emissions, position_logits, crf_loss, _ = \
                model.forward_from_zseq(
                    z_seq, pad_mask=pad_mask, labels=padded_labels)

            focal_raw = focal_fn(emissions.view(-1, 2), padded_labels.view(-1))
            focal_clean = ((focal_raw * valid_mask.view(-1).float()).sum()
                           / (valid_mask.sum() + 1e-9))

            cl_loss_val = 0.0
            cl_loss_t = torch.tensor(0.0, device=device)
            if epoch >= cl_warmup and not cfg['ablate_cl']:
                valid_styles = z_style_seq[valid_mask].unsqueeze(1)
                valid_labels = padded_labels[valid_mask]
                if valid_styles.shape[0] > 2:
                    cl_loss_t = cl_fn(valid_styles, valid_labels)
                    cl_loss_val = cl_loss_t.item()

            crf_loss_val = 0.0
            if use_crf and crf_loss is not None:
                crf_loss_val = crf_loss.item()
            else:
                crf_loss = torch.tensor(0.0, device=device)

            pos_loss_val = 0.0
            pos_loss_t = torch.tensor(0.0, device=device)
            if use_position and position_logits is not None:
                pos_labels = make_position_labels(pad_mask)
                pos_loss_t = F.cross_entropy(
                    position_logits.view(-1, 3), pos_labels.view(-1),
                    ignore_index=-100,
                )
                pos_loss_val = pos_loss_t.item()

            mixup_focal = torch.tensor(0.0, device=device)
            if do_mixup:
                mixed_z, lam, perm = mixup_zseq(
                    z_seq.detach(), padded_labels, pad_mask, alpha=mixup_alpha)
                emissions_mix, _, _, _ = model.forward_from_zseq(
                    mixed_z, pad_mask=pad_mask, labels=None)
                focal_a = focal_fn(
                    emissions_mix.view(-1, 2), padded_labels.view(-1))
                focal_b = focal_fn(
                    emissions_mix.view(-1, 2),
                    padded_labels[perm].view(-1))
                focal_a_m = ((focal_a * valid_mask.view(-1).float()).sum()
                             / (valid_mask.sum() + 1e-9))
                focal_b_m = ((focal_b * valid_mask.view(-1).float()).sum()
                             / (valid_mask.sum() + 1e-9))
                mixup_focal = lam * focal_a_m + (1 - lam) * focal_b_m

            total_loss = (
                focal_clean
                + alpha_cl * cl_loss_t
                + alpha_crf * crf_loss
                + (alpha_pos * pos_loss_t if use_position else 0.0)
                + (0.5 * mixup_focal if do_mixup else 0.0)
            )

            total_loss.backward()

            if use_awp and epoch >= awp_start:
                awp.attack_step()
                z_seq_adv, _ = model.encode_to_zseq(
                    hidden_states, token_probs, entropies, diveye_feats)
                emissions_adv, _, _, _ = model.forward_from_zseq(
                    z_seq_adv, pad_mask=pad_mask, labels=padded_labels)
                focal_adv = focal_fn(
                    emissions_adv.view(-1, 2), padded_labels.view(-1))
                adv_loss = ((focal_adv * valid_mask.view(-1).float()).sum()
                            / (valid_mask.sum() + 1e-9))
                adv_loss.backward()
                awp.restore()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            train_loss_total += total_loss.item()
            b_idx = batch_idx_raw // batch_size
            if b_idx % 30 == 0:
                mix_tag = " [MIX]" if do_mixup else ""
                awp_tag = (" [AWP]"
                           if (use_awp and epoch >= awp_start) else "")
                print(f"  Batch {b_idx}/{num_batches}{mix_tag}{awp_tag} | "
                      f"Loss: {total_loss.item():.4f} "
                      f"(F: {focal_clean.item():.4f} "
                      f"CL: {cl_loss_val:.4f} "
                      f"CRF: {crf_loss_val:.4f} "
                      f"Pos: {pos_loss_val:.4f})")

        avg_loss = train_loss_total / num_batches
        print(f"  Train avg loss: {avg_loss:.4f}")
        scheduler.step()

        if ema is not None:
            backup = ema.apply_to(model)

        val_emissions, val_masks, val_labels = evaluate_set(
            model, val_data, device, batch_size, max_seq_len)
        val_metrics, optimal_thresh = compute_metrics(
            val_labels, val_emissions, val_masks, search_thresh=True)
        print(f"  [Val{' EMA' if ema else ''}] "
              f"Thresh: {optimal_thresh:.2f} | "
              f"F1: {val_metrics['F1']:.4f} | "
              f"AUC: {val_metrics['AUC']:.4f} | "
              f"MCC: {val_metrics['MCC']:.4f}")

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            best_thresh = optimal_thresh
            patience = 0
            torch.save({
                'model_state': model.state_dict(),
                'best_thresh': float(optimal_thresh),
                'epoch': epoch + 1,
                'val_f1': best_val_f1,
            }, save_path)
            print(f"  ** New best! Saved to {save_path}")
        else:
            patience += 1

        if ema is not None:
            ema.restore(model, backup)

        if patience >= max_patience:
            print(f"  [EARLY STOP] No improvement for {max_patience} epochs.")
            break

    return best_val_f1, best_thresh


def test_model(model, test_data, save_path, fixed_thresh,
               batch_size, max_seq_len):
    print(f"\n{'=' * 20} Final Test {'=' * 20}")
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f"[INFO] Loaded best model from epoch {checkpoint.get('epoch', '?')}")
    print(f"[INFO] Val F1: {checkpoint.get('val_f1', 0):.4f}")
    print(f"[INFO] Fixed threshold: {fixed_thresh:.2f}")

    test_emissions, test_masks, test_labels = evaluate_set(
        model, test_data, device, batch_size, max_seq_len)
    test_metrics, _ = compute_metrics(
        test_labels, test_emissions, test_masks,
        search_thresh=False, fixed_thresh=fixed_thresh)
    print(f"\n  ** TEST RESULTS **")
    print(f"     F1 (Macro): {test_metrics['F1']:.4f}")
    print(f"     AUC:        {test_metrics['AUC']:.4f}")
    print(f"     MCC:        {test_metrics['MCC']:.4f}")
    return test_metrics


SWEEP_CONFIGS = {
    'D': {
        'label': 'D: GCN3 + branch_drop 0.15 + Freq ON',
        'd_compress': 96, 'gcn_layers': 3, 'cl_warmup_epochs': 2,
        'branch_dropout': 0.15, 'head_dropout': 0.2,
        'use_freq': True, 'use_hybrid_adj': False, 'use_position_aux': False,
        'use_ema': False, 'use_mixup': False, 'use_awp': False,
        'label_smoothing': 0.05,
    },
    'X': {
        'label': 'X: D + Freq OFF + EMA + Mixup + AWP + Pos + HybridAdj + LS0.10',
        'd_compress': 96, 'gcn_layers': 3, 'cl_warmup_epochs': 2,
        'branch_dropout': 0.15, 'head_dropout': 0.2,
        'use_freq': False, 'use_hybrid_adj': True, 'use_position_aux': True,
        'use_ema': True, 'use_mixup': True, 'use_awp': True,
        'label_smoothing': 0.10,
    },
    'X1': {
        'label': 'X1: D minus Freq',
        'd_compress': 96, 'gcn_layers': 3, 'cl_warmup_epochs': 2,
        'branch_dropout': 0.15, 'head_dropout': 0.2,
        'use_freq': False, 'use_hybrid_adj': False, 'use_position_aux': False,
        'use_ema': False, 'use_mixup': False, 'use_awp': False,
        'label_smoothing': 0.05,
    },
}


def build_model(config_name, use_crf,
                ablate_gcn=False, ablate_freq_arg=False,
                ablate_tcn=False, ablate_cl=False):
    cfg = SWEEP_CONFIGS[config_name]
    final_ablate_freq = (not cfg['use_freq']) or ablate_freq_arg
    model = SenFlowNet(
        d_in=4096, d_model=D_MODEL, d_diveye=D_DIVEYE, d_freq=D_FREQ,
        d_compress=cfg['d_compress'],
        use_crf=use_crf, max_seq_len=MAX_SEQ_LEN,
        gcn_layers=cfg['gcn_layers'], gcn_dropout=GCN_DROPOUT,
        branch_dropout=cfg['branch_dropout'],
        head_dropout=cfg['head_dropout'],
        use_hybrid_adj=cfg['use_hybrid_adj'],
        use_position_aux=cfg['use_position_aux'],
        ablate_gcn=ablate_gcn,
        ablate_freq=final_ablate_freq,
        ablate_tcn=ablate_tcn,
        ablate_cl=ablate_cl,
    ).to(device)
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--train_multi", type=str, nargs='+', default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints_senflow")
    parser.add_argument("--config", type=str, default='D',
                        choices=['D', 'X', 'X1'])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ablate_tcn", action="store_true")
    parser.add_argument("--ablate_freq", action="store_true")
    parser.add_argument("--ablate_gcn", action="store_true")
    parser.add_argument("--ablate_cl", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--no_awp", action="store_true")
    parser.add_argument("--no_position", action="store_true")
    parser.add_argument("--no_hybrid_adj", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.train and not args.train_multi:
        raise ValueError("Must specify --train or --train_multi")

    _active_seed = args.seed if args.seed is not None else SEED
    set_seed(_active_seed)
    print(f"[SEED] active seed = {_active_seed}")

    if args.train_multi:
        train_data, val_data, train_test_data, train_meta = \
            load_multi_cached(args.train_multi)
        train_name = "multi_" + "_".join(
            s.split("/")[-1].replace(".pt", "") for s in args.train_multi)
    else:
        train_data, val_data, train_test_data, train_meta = \
            load_cached_features(args.train)
        train_name = (f"{train_meta.get('domain', '?')}_"
                      f"{train_meta.get('generator', '?')}")

    if args.test:
        _, _, test_data, test_meta = load_cached_features(args.test)
        test_name = (f"{test_meta.get('domain', '?')}_"
                     f"{test_meta.get('generator', '?')}")
        exp_name = f"{train_name}_to_{test_name}"
    else:
        test_data = train_test_data
        test_name = train_name
        exp_name = f"{train_name}_unified"

    exp_name = f"{exp_name}__cfg{args.config}"
    ablate_tags = []
    if args.ablate_gcn:
        ablate_tags.append("noGCN")
    if not args.use_crf:
        ablate_tags.append("noCRF")
    if args.ablate_cl:
        ablate_tags.append("noCL")
    if args.ablate_freq:
        ablate_tags.append("noFreq")
    if args.ablate_tcn:
        ablate_tags.append("noTCN")
    if args.no_ema:
        ablate_tags.append("noEMA")
    if args.no_mixup:
        ablate_tags.append("noMixup")
    if args.no_awp:
        ablate_tags.append("noAWP")
    if args.no_position:
        ablate_tags.append("noPos")
    if args.no_hybrid_adj:
        ablate_tags.append("noHA")
    if ablate_tags:
        exp_name += "__" + "_".join(ablate_tags)
    if args.seed is not None:
        exp_name += f"__seed{args.seed}"

    print(f"\n[EXP] Name: {exp_name}")
    print(f"[EXP] Train: {len(train_data)} | Val: {len(val_data)} | "
          f"Test: {len(test_data)}")

    os.makedirs(args.save_dir, exist_ok=True)

    model, cfg_model = build_model(
        args.config, args.use_crf,
        ablate_gcn=args.ablate_gcn,
        ablate_freq_arg=args.ablate_freq,
        ablate_tcn=args.ablate_tcn,
        ablate_cl=args.ablate_cl,
    )
    if args.no_position:
        model.use_position_aux = False
    if args.no_hybrid_adj and hasattr(model, 'gcn'):
        model.gcn.use_hybrid_adj = False

    cl_warmup = cfg_model['cl_warmup_epochs']
    param_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {param_count:,}")
    print(f"[MODEL] Config: {args.config} | {cfg_model['label']}")
    print(f"[MODEL] CRF: {'ON' if args.use_crf else 'OFF'}")
    print(f"[MODEL] use_freq: "
          f"{cfg_model['use_freq'] and not args.ablate_freq}")
    print(f"[MODEL] use_hybrid_adj: "
          f"{cfg_model['use_hybrid_adj'] and not args.no_hybrid_adj}")
    print(f"[MODEL] use_position_aux: "
          f"{cfg_model['use_position_aux'] and not args.no_position}")

    save_path = os.path.join(args.save_dir, f"best_{exp_name}.pth")

    cfg = {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'alpha_cl': ALPHA_CL if not args.ablate_cl else 0.0,
        'alpha_crf': ALPHA_CRF,
        'alpha_pos': ALPHA_POS,
        'cl_warmup_epochs': cl_warmup,
        'lr_warmup_epochs': LR_WARMUP_EPOCHS,
        'max_seq_len': MAX_SEQ_LEN,
        'cl_temperature': CL_TEMPERATURE,
        'use_crf': args.use_crf,
        'grad_clip': 2.0,
        'focal_alpha': FOCAL_ALPHA,
        'focal_gamma': FOCAL_GAMMA,
        'label_smoothing': cfg_model['label_smoothing'],
        'use_ema': cfg_model['use_ema'] and not args.no_ema,
        'ema_decay': EMA_DECAY,
        'use_mixup': cfg_model['use_mixup'] and not args.no_mixup,
        'mixup_alpha': MIXUP_ALPHA,
        'mixup_prob': MIXUP_PROB,
        'use_awp': cfg_model['use_awp'] and not args.no_awp,
        'awp_lr': AWP_LR,
        'awp_eps': AWP_EPS,
        'awp_start_epoch': AWP_START_EPOCH,
        'use_position_aux': (cfg_model['use_position_aux']
                             and not args.no_position),
        'ablate_cl': args.ablate_cl,
    }

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False)
        best_val_f1 = ckpt.get('val_f1', 0.0)
        best_thresh = ckpt.get('best_thresh', 0.5)
        save_path = args.checkpoint
        print(f"[CHECKPOINT] Loading: {args.checkpoint}")
        print(f"[CHECKPOINT] Val F1: {best_val_f1:.4f} | "
              f"Thresh: {best_thresh:.2f}")
    else:
        best_val_f1, best_thresh = train_model(
            model, train_data, val_data, save_path, cfg)

    test_metrics = test_model(
        model, test_data, save_path, best_thresh, BATCH_SIZE, MAX_SEQ_LEN)

    result_path = os.path.join(args.save_dir, f"result_{exp_name}.txt")
    with open(result_path, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        if args.train:
            f.write(f"Train file: {args.train}\n")
        if args.train_multi:
            f.write(f"Train files: {args.train_multi}\n")
        f.write(f"Test file: {args.test or 'same as train'}\n")
        f.write(f"Config name: {args.config}\n")
        f.write(f"Config: {json.dumps(cfg, indent=2)}\n")
        f.write(f"Best val F1: {best_val_f1:.4f}\n")
        f.write(f"Threshold: {best_thresh:.2f}\n")
        f.write(f"Test F1 (Macro): {test_metrics['F1']:.4f}\n")
        f.write(f"Test AUC: {test_metrics['AUC']:.4f}\n")
        f.write(f"Test MCC: {test_metrics['MCC']:.4f}\n")

    print(f"\n[SAVE] Results: {result_path}")
    print(f"[DONE]")


if __name__ == "__main__":
    main()
