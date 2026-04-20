from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        feature_dim = hidden_dim + solar_dim + 4 + (cloud_dim if use_global_cloud else 0)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(64, 1)
        self.base_logvar_head = nn.Linear(64, 1)
        self.variation_logvar = nn.Linear(2, 1, bias=False)

    def forward(
        self,
        pooled_cloud: torch.Tensor,
        pv_history: torch.Tensor,
        solar_vec: torch.Tensor,
        global_rbr_mean: torch.Tensor,
        sun_local_rbr_mean: torch.Tensor,
        global_rbr_variance: torch.Tensor,
        sun_local_rbr_variance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pv_feature = self.pv_encoder(pv_history)
        distribution_features = [
            global_rbr_mean,
            sun_local_rbr_mean,
            global_rbr_variance,
            sun_local_rbr_variance,
        ]
        feature_parts = [pv_feature, solar_vec, *distribution_features]
        if self.use_global_cloud:
            feature_parts.insert(0, pooled_cloud)
        features = torch.cat(feature_parts, dim=-1)
        hidden = self.backbone(features)
        pv_mu = self.mu_head(hidden)
        variance_drivers = torch.cat([global_rbr_variance, sun_local_rbr_variance], dim=-1)
        pv_logvar = self.base_logvar_head(hidden) + F.linear(variance_drivers, F.softplus(self.variation_logvar.weight))
        pv_logvar = pv_logvar.clamp(min=-8.0, max=4.0)
        pv_sigma = torch.exp(0.5 * pv_logvar)
        z = torch.tensor([-1.2815516, -0.67448975, 0.0, 0.67448975, 1.2815516], device=pv_mu.device, dtype=pv_mu.dtype)
        quantiles = pv_mu + pv_sigma * z.view(1, -1)
        return {
            "prediction": quantiles,
            "pv_mu": pv_mu,
            "pv_logvar": pv_logvar,
            "pv_sigma": pv_sigma,
        }
