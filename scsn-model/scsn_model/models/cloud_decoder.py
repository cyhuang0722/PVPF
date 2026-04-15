from __future__ import annotations

import torch
import torch.nn as nn


class FutureCloudStateDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, feature_hw: int, max_motion: float) -> None:
        super().__init__()
        self.feature_hw = feature_hw
        self.max_motion = max_motion
        decoded_dim = latent_dim + hidden_dim
        self.motion_decoder = nn.Sequential(
            nn.Linear(decoded_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2 * feature_hw * feature_hw),
        )
        self.opacity_decoder = nn.Sequential(
            nn.Linear(decoded_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_hw * feature_hw),
        )
        self.gap_decoder = nn.Sequential(
            nn.Linear(decoded_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_hw * feature_hw),
        )
        self.sun_occ_decoder = nn.Sequential(
            nn.Linear(decoded_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, future_z: torch.Tensor, hidden_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps, _, = future_z.shape
        decoded = torch.cat([future_z, hidden_seq], dim=-1)
        motion = torch.tanh(self.motion_decoder(decoded)).view(batch, steps, 2, self.feature_hw, self.feature_hw) * self.max_motion
        opacity = torch.sigmoid(self.opacity_decoder(decoded)).view(batch, steps, 1, self.feature_hw, self.feature_hw)
        gap = torch.sigmoid(self.gap_decoder(decoded)).view(batch, steps, 1, self.feature_hw, self.feature_hw)
        sun_occ = torch.sigmoid(self.sun_occ_decoder(decoded)).squeeze(-1)
        transmission = torch.clamp((1.0 - opacity) * (0.25 + 0.75 * gap), 0.0, 1.0)
        return {
            "motion_fields": motion,
            "opacity_maps": opacity,
            "gap_maps": gap,
            "sun_occlusion": sun_occ,
            "transmission_maps": transmission,
        }
