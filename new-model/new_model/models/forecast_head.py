from __future__ import annotations

import torch
import torch.nn as nn


class QuantileForecastHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)

