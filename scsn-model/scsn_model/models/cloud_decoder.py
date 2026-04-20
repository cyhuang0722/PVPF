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
        self.rbr_mean_head = nn.Conv2d(trunk_channels, 1, kernel_size=3, padding=1)
        self.rbr_logvar_head = nn.Conv2d(trunk_channels, 1, kernel_size=3, padding=1)

    def forward(self, future_z: torch.Tensor, hidden_seq: torch.Tensor, spatial_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps, _ = future_z.shape
        spatial_rep = spatial_feat.unsqueeze(1).expand(-1, steps, -1, -1, -1)
        hidden_map = hidden_seq.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.feature_hw, self.feature_hw)
        z_map = future_z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.feature_hw, self.feature_hw)
        decoder_in = torch.cat([spatial_rep, z_map, hidden_map], dim=2).reshape(batch * steps, -1, self.feature_hw, self.feature_hw)
        trunk = self.input_proj(decoder_in)
        rbr_mean = torch.sigmoid(self.rbr_mean_head(trunk)).view(batch, steps, 1, self.feature_hw, self.feature_hw)
        rbr_logvar = self.rbr_logvar_head(trunk).clamp(min=-7.0, max=3.0).view(batch, steps, 1, self.feature_hw, self.feature_hw)
        return {
            "future_rbr_mean_maps": rbr_mean,
            "future_rbr_logvar_maps": rbr_logvar,
        }
