from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchFrameEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WeatherConditionedSunAwareModel(nn.Module):
    def __init__(
        self,
        global_input_dim: int,
        patch_channels: int = 6,
        weather_classes: int = 3,
        weather_embed_dim: int = 12,
        patch_embed_dim: int = 96,
        temporal_hidden_dim: int = 96,
        global_hidden_dim: int = 96,
        dropout: float = 0.25,
        stable_residual_limit: float = 0.15,
        cloudy_residual_limit: float = 0.75,
        min_sigma: float = 0.015,
        max_sigma: float = 0.65,
        min_df: float = 2.5,
        max_df: float = 30.0,
    ) -> None:
        super().__init__()
        self.stable_residual_limit = float(stable_residual_limit)
        self.cloudy_residual_limit = float(cloudy_residual_limit)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.min_df = float(min_df)
        self.max_df = float(max_df)

        self.patch_encoder = PatchFrameEncoder(patch_channels, patch_embed_dim)
        self.patch_gru = nn.GRU(patch_embed_dim, temporal_hidden_dim, batch_first=True)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, global_hidden_dim),
            nn.LayerNorm(global_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.GELU(),
        )
        self.weather_embedding = nn.Embedding(weather_classes, weather_embed_dim)
        joint_dim = temporal_hidden_dim + global_hidden_dim + weather_embed_dim
        self.gate_head = nn.Sequential(nn.Linear(joint_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.residual_head = nn.Sequential(nn.Linear(joint_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.scale_head = nn.Sequential(nn.Linear(joint_dim + 1, 64), nn.GELU(), nn.Linear(64, 1))
        self.df_head = nn.Sequential(nn.Linear(joint_dim + 1, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, patch_seq: torch.Tensor, global_x: torch.Tensor, weather_idx: torch.Tensor, baseline_csi: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = patch_seq.shape
        frame_emb = self.patch_encoder(patch_seq.reshape(batch * steps, channels, height, width)).view(batch, steps, -1)
        _, hidden = self.patch_gru(frame_emb)
        temporal_emb = hidden[-1]
        global_emb = self.global_encoder(global_x)
        weather_emb = self.weather_embedding(weather_idx)
        joint = torch.cat([temporal_emb, global_emb, weather_emb], dim=-1)

        cloud_gate = torch.sigmoid(self.gate_head(joint))
        residual_limit = self.stable_residual_limit + (self.cloudy_residual_limit - self.stable_residual_limit) * cloud_gate
        residual_loc = residual_limit * torch.tanh(self.residual_head(joint))
        loc = (baseline_csi.view_as(residual_loc) + residual_loc).clamp(-0.1, 1.25)

        uncertainty_context = torch.cat([joint, cloud_gate], dim=-1)
        scale_unit = torch.sigmoid(self.scale_head(uncertainty_context))
        df_unit = torch.sigmoid(self.df_head(uncertainty_context))
        scale = self.min_sigma + (self.max_sigma - self.min_sigma) * scale_unit
        df = self.min_df + (self.max_df - self.min_df) * df_unit
        return {
            "loc": loc,
            "scale": scale,
            "df": df,
            "cloud_gate": cloud_gate,
            "residual_limit": residual_limit,
            "residual_loc": residual_loc,
        }


def student_t_nll(
    loc: torch.Tensor,
    scale: torch.Tensor,
    df: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    target = target.view_as(loc)
    dist = torch.distributions.StudentT(df=df.clamp_min(2.01), loc=loc, scale=scale.clamp_min(1e-4))
    loss = -dist.log_prob(target)
    if weight is None:
        return loss.mean()
    weight = weight.view_as(loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def interval_width_regularizer(scale: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(scale, torch.full_like(scale, 0.08), reduction="none")
    if weight is None:
        return loss.mean()
    weight = weight.view_as(loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
