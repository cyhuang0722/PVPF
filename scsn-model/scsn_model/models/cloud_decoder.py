from __future__ import annotations

import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class FutureCloudStateDecoder(nn.Module):
    def __init__(self, spatial_dim: int, latent_dim: int, hidden_dim: int, feature_hw: int) -> None:
        super().__init__()
        self.feature_hw = feature_hw
        in_channels = spatial_dim + latent_dim + hidden_dim
        trunk_channels = max(spatial_dim, 128)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualConvBlock(trunk_channels),
            ResidualConvBlock(trunk_channels),
        )
        self.change_hotspot_head = nn.Conv2d(trunk_channels, 1, kernel_size=3, padding=1)
        self.cloud_prob_head = nn.Conv2d(trunk_channels, 1, kernel_size=3, padding=1)
        self.sun_occ_decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.current_state_head = nn.Sequential(
            nn.Conv2d(spatial_dim, trunk_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualConvBlock(trunk_channels),
        )
        self.current_cloud_prob = nn.Conv2d(trunk_channels, 1, kernel_size=3, padding=1)

    def forward(self, future_z: torch.Tensor, hidden_seq: torch.Tensor, spatial_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps, _ = future_z.shape
        spatial_rep = spatial_feat.unsqueeze(1).expand(-1, steps, -1, -1, -1)
        hidden_map = hidden_seq.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.feature_hw, self.feature_hw)
        z_map = future_z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.feature_hw, self.feature_hw)
        decoder_in = torch.cat([spatial_rep, z_map, hidden_map], dim=2).reshape(batch * steps, -1, self.feature_hw, self.feature_hw)
        trunk = self.input_proj(decoder_in)
        change_hotspot = torch.sigmoid(self.change_hotspot_head(trunk)).view(batch, steps, 1, self.feature_hw, self.feature_hw)
        cloud_prob = torch.sigmoid(self.cloud_prob_head(trunk)).view(batch, steps, 1, self.feature_hw, self.feature_hw)
        sun_occ_in = torch.cat([future_z, hidden_seq], dim=-1)
        sun_occ = torch.sigmoid(self.sun_occ_decoder(sun_occ_in)).squeeze(-1)

        current_feat = self.current_state_head(spatial_feat)
        return {
            "future_change_hotspot_maps": change_hotspot,
            "sun_occlusion": sun_occ,
            "future_cloud_prob_maps": cloud_prob,
            "current_cloud_prob": torch.sigmoid(self.current_cloud_prob(current_feat)),
        }
