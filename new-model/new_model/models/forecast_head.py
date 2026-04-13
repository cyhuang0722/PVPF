from __future__ import annotations

import torch
import torch.nn as nn


class DeterministicForecastHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, out_dim: int = 1):
        super().__init__()
        if out_dim != 1:
            raise ValueError("DeterministicForecastHead expects exactly one output.")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(fused))
