from __future__ import annotations

import torch
import torch.nn.functional as F


def scsn_training_loss(
    pv_mu: torch.Tensor,
    pv_logvar: torch.Tensor,
    target: torch.Tensor,
    kl_loss: torch.Tensor,
    recon_rbr: torch.Tensor,
    target_rbr: torch.Tensor,
    loss_cfg: dict,
) -> dict[str, torch.Tensor]:
    target = target.view_as(pv_mu)
    pv_loss = 0.5 * (torch.exp(-pv_logvar) * (target - pv_mu).pow(2) + pv_logvar)
    pv_loss = pv_loss.mean()
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
