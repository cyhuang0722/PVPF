from __future__ import annotations

import torch
import torch.nn.functional as F


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    losses = []
    for idx, q in enumerate(quantiles):
        err = target - pred[:, idx]
        losses.append(torch.maximum(q * err, (q - 1.0) * err).mean())
    return torch.stack(losses).sum()


def quantile_crossing_penalty(pred: torch.Tensor) -> torch.Tensor:
    lower, median, upper = pred[:, 0], pred[:, 1], pred[:, 2]
    return F.relu(lower - median).mean() + F.relu(median - upper).mean()

