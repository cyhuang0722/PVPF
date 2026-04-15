from __future__ import annotations

import torch
import torch.nn as nn

from .image_encoder import ResidualBlock


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = x.new_zeros((x.shape[0], self.hidden_dim, x.shape[-2], x.shape[-1]))
            c = x.new_zeros((x.shape[0], self.hidden_dim, x.shape[-2], x.shape[-1]))
        else:
            h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class TemporalCloudStateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)
        self.refine = ResidualBlock(hidden_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        for t in range(features.shape[1]):
            state = self.cell(features[:, t], state)
        hidden, _ = state
        spatial_feat = self.refine(hidden)
        global_feat = spatial_feat.mean(dim=(2, 3))
        return {
            "spatial_feat": spatial_feat,
            "global_feat": global_feat,
        }
