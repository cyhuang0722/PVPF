from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
        )
        self.q50 = nn.Linear(64, 1)
        self.lower_width = nn.Linear(64, 1)
        self.upper_width = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        q50 = self.q50(feat)
        q10 = q50 - F.softplus(self.lower_width(feat))
        q90 = q50 + F.softplus(self.upper_width(feat))
        return torch.cat([q10, q50, q90], dim=-1)


class SunConditionedPowerHead(nn.Module):
    def __init__(self, cloud_dim: int, pv_history_dim: int, solar_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.pv_encoder = nn.Sequential(
            nn.Linear(pv_history_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = QuantileMLPHead(in_dim=cloud_dim + hidden_dim + solar_dim + 4, hidden_dim=128)

    def forward(
        self,
        pooled_cloud: torch.Tensor,
        pv_history: torch.Tensor,
        solar_vec: torch.Tensor,
        sun_local_transmission: torch.Tensor,
        sun_local_gap: torch.Tensor,
        sun_local_opacity: torch.Tensor,
        sun_occlusion_risk: torch.Tensor,
    ) -> torch.Tensor:
        pv_feature = self.pv_encoder(pv_history)
        features = torch.cat(
            [
                pooled_cloud,
                pv_feature,
                solar_vec,
                sun_local_transmission,
                sun_local_gap,
                sun_local_opacity,
                sun_occlusion_risk,
            ],
            dim=-1,
        )
        return self.head(features)
