from __future__ import annotations

import torch
import torch.nn as nn


class PVHistoryEncoder(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
            nn.GELU(),
        )

    def forward(self, pv_history: torch.Tensor) -> torch.Tensor:
        return self.net(pv_history)

