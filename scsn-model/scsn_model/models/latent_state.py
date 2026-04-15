from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredLatentState(nn.Module):
    def __init__(self, in_dim: int, latent_dims: dict[str, int]) -> None:
        super().__init__()
        self.latent_dims = latent_dims
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(in_dim, max(in_dim, dim * 2)),
                    nn.GELU(),
                    nn.Linear(max(in_dim, dim * 2), dim * 2),
                )
                for name, dim in latent_dims.items()
            }
        )

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        pooled = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
        posterior: dict[str, dict[str, torch.Tensor]] = {}
        samples: dict[str, torch.Tensor] = {}
        kl_loss = pooled.new_zeros(pooled.shape[0])
        for name, head in self.heads.items():
            stats = head(pooled)
            mu, logvar = torch.chunk(stats, 2, dim=-1)
            logvar = logvar.clamp(min=-6.0, max=4.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std if self.training else mu
            posterior[name] = {"mu": mu, "logvar": logvar}
            samples[name] = z
            kl_loss = kl_loss + 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
        return {
            "pooled": pooled,
            "posterior": posterior,
            "samples": samples,
            "kl_loss": kl_loss.mean(),
        }
