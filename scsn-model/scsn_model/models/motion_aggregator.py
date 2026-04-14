from __future__ import annotations

import torch
import torch.nn as nn


class MotionAggregator(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, out_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),
        )

    def forward(self, motion_fields: torch.Tensor) -> torch.Tensor:
        x = motion_fields.permute(0, 2, 1, 3, 4)
        x = self.net(x)
        return x.mean(dim=2)

