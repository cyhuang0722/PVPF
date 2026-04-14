from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(pred)
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weight
    denom = weight.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_direction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dot = (pred * target).sum(dim=2)
    pred_norm = torch.linalg.norm(pred, dim=2)
    target_norm = torch.linalg.norm(target, dim=2)
    cosine = dot / (pred_norm * target_norm + eps)
    loss = (1.0 - cosine) * mask.squeeze(2)
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_patch_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dot = (pred * target).sum(dim=-1)
    pred_norm = torch.linalg.norm(pred, dim=-1)
    target_norm = torch.linalg.norm(target, dim=-1)
    cosine = dot / (pred_norm * target_norm + eps)
    valid = mask.squeeze(-1)
    loss = (1.0 - cosine) * valid
    denom = valid.sum().clamp_min(1.0)
    return loss.sum() / denom
