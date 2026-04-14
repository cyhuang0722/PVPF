from __future__ import annotations

import torch
import torch.nn as nn


class MotionFieldHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: list[int], max_displacement: float):
        super().__init__()
        in_dim = feature_dim * 3
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(prev, hidden, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                ]
            )
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.motion = nn.Conv2d(prev, 2, kernel_size=3, padding=1)
        self.max_displacement = max_displacement

    def forward(self, prev_feat: torch.Tensor, curr_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([prev_feat, curr_feat, curr_feat - prev_feat], dim=1)
        hidden = self.backbone(x)
        flow = torch.tanh(self.motion(hidden)) * self.max_displacement
        return flow, hidden

