from __future__ import annotations

import torch
import torch.nn as nn


class QuantileForecastHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, out_dim: int = 3):
        super().__init__()
        if out_dim != 3:
            raise ValueError("QuantileForecastHead expects exactly 3 quantile outputs (q10/q50/q90).")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )
        self.softplus = nn.Softplus()

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        raw = self.net(fused)
        q50 = torch.sigmoid(raw[:, 0:1])
        d1 = self.softplus(raw[:, 1:2])
        d2 = self.softplus(raw[:, 2:3])
        q10 = torch.clamp(q50 - d1, min=0.0, max=1.0)
        q90 = torch.clamp(q50 + d2, min=0.0, max=1.0)
        return torch.cat([q10, q50, q90], dim=1)
