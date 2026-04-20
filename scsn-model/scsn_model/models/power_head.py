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
    def __init__(self, cloud_dim: int, pv_history_dim: int, solar_dim: int, hidden_dim: int, use_global_cloud: bool = False) -> None:
        super().__init__()
        self.use_global_cloud = use_global_cloud
        self.pv_encoder = nn.Sequential(
            nn.Linear(pv_history_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        head_in_dim = hidden_dim + solar_dim + 4 + (cloud_dim if use_global_cloud else 0)
        self.head = QuantileMLPHead(in_dim=head_in_dim, hidden_dim=128)

    def forward(
        self,
        pooled_cloud: torch.Tensor,
        pv_history: torch.Tensor,
        solar_vec: torch.Tensor,
        sun_local_cloud_prob: torch.Tensor,
        global_cloud_prob: torch.Tensor,
        sun_local_motion_hotspot: torch.Tensor,
        sun_occlusion_risk: torch.Tensor,
    ) -> torch.Tensor:
        pv_feature = self.pv_encoder(pv_history)
        feature_parts = [
            pv_feature,
            solar_vec,
            sun_local_cloud_prob,
            global_cloud_prob,
            sun_local_motion_hotspot,
            sun_occlusion_risk,
        ]
        if self.use_global_cloud:
            feature_parts.insert(0, pooled_cloud)
        features = torch.cat(feature_parts, dim=-1)
        return self.head(features)
