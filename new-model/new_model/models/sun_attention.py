from __future__ import annotations

import torch
import torch.nn as nn


class SunConditionedAttention(nn.Module):
    def __init__(self, visual_dim: int, sun_dim: int, attention_dim: int):
        super().__init__()
        self.sun_mlp = nn.Sequential(
            nn.Linear(sun_dim, 64),
            nn.GELU(),
            nn.Linear(64, attention_dim),
        )
        self.key_proj = nn.Conv2d(visual_dim, attention_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(visual_dim, attention_dim, kernel_size=1)
        self.scale = attention_dim ** 0.5

    def forward(self, visual_map: torch.Tensor, solar_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, h, w = visual_map.shape
        query = self.sun_mlp(solar_vec)
        keys = self.key_proj(visual_map).flatten(2).transpose(1, 2)
        values = self.value_proj(visual_map).flatten(2).transpose(1, 2)
        score = torch.einsum("bd,bnd->bn", query, keys) / self.scale
        attn = torch.softmax(score, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", attn, values)
        return pooled, attn.view(bsz, h, w)

