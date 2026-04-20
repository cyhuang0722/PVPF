from __future__ import annotations

import torch
import torch.nn.functional as F

from .quantile import quantile_crossing_penalty, quantile_loss


def scsn_training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kl_loss: torch.Tensor,
    recon_rbr: torch.Tensor,
    target_rbr: torch.Tensor,
    loss_cfg: dict,
) -> dict[str, torch.Tensor]:
    pv_loss = quantile_loss(prediction, target, [0.1, 0.5, 0.9])
    pv_loss = pv_loss + float(loss_cfg.get("crossing_weight", 0.2)) * quantile_crossing_penalty(prediction)
    recon_loss = F.l1_loss(recon_rbr, target_rbr)
    total = (
        float(loss_cfg.get("pv_weight", 1.0)) * pv_loss
        + float(loss_cfg.get("kl_weight", 0.02)) * kl_loss
        + float(loss_cfg.get("reconstruction_weight", 0.2)) * recon_loss
    )
    return {
        "total": total,
        "pv": pv_loss,
        "kl": kl_loss,
        "recon": recon_loss,
    }
