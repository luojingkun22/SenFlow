import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_style, labels):
        z = z_style.squeeze(1)
        z = F.normalize(z, dim=1)

        similarity_matrix = torch.matmul(z, z.T) / self.temperature

        diag_mask = torch.eye(
            similarity_matrix.shape[0],
            device=similarity_matrix.device,
        ).bool()
        sim_masked = similarity_matrix.masked_fill(diag_mask, float('-inf'))
        max_sim = sim_masked.max(dim=1, keepdim=True)[0]
        similarity_matrix = similarity_matrix - max_sim.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        logits_mask = (~diag_mask).float()
        mask = mask * logits_mask

        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(
            exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_sum = mask.sum(1)
        valid_anchors = mask_sum > 0

        if not valid_anchors.any():
            return (z * 0).sum()

        mean_log_prob_pos = (
            mask[valid_anchors] * log_prob[valid_anchors]
        ).sum(1) / mask_sum[valid_anchors]
        loss = -mean_log_prob_pos.mean()

        return loss
