from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentTResidualModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.15,
        min_df: float = 2.5,
        max_df: float = 30.0,
        min_sigma: float = 0.01,
        max_sigma: float = 0.45,
    ) -> None:
        super().__init__()
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.loc_head = nn.Linear(hidden_dim // 2, 1)
        self.scale_head = nn.Linear(hidden_dim // 2, 1)
        self.df_head = nn.Linear(hidden_dim // 2, 1)
        self.aux_head = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x: torch.Tensor, baseline_csi: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(x)
        residual_loc = 0.35 * torch.tanh(self.loc_head(hidden))
        scale_unit = torch.sigmoid(self.scale_head(hidden))
        scale = self.min_sigma + (self.max_sigma - self.min_sigma) * scale_unit
        df_unit = torch.sigmoid(self.df_head(hidden))
        df = self.min_df + (self.max_df - self.min_df) * df_unit
        loc = baseline_csi.view_as(residual_loc) + residual_loc
        loc = loc.clamp(-0.1, 1.25)
        aux = self.aux_head(hidden)
        return {
            "loc": loc,
            "scale": scale,
            "df": df,
            "residual_loc": residual_loc,
            "aux": aux,
        }


def student_t_nll(loc: torch.Tensor, scale: torch.Tensor, df: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.view_as(loc)
    dist = torch.distributions.StudentT(df=df.clamp_min(2.01), loc=loc, scale=scale.clamp_min(1e-4))
    return -dist.log_prob(target).mean()


def interval_width_regularizer(scale: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(scale, torch.full_like(scale, 0.08))

