from __future__ import annotations

import torch
import torch.nn as nn


class VariationalGRUDynamics(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, future_steps: int) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.hidden_init = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(self, z0: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = torch.tanh(self.hidden_init(z0))
        z = z0
        future_z = []
        hidden_seq = []
        for _ in range(self.future_steps):
            hidden = self.gru(z, hidden)
            stats = self.transition(hidden)
            mu, logvar = torch.chunk(stats, 2, dim=-1)
            logvar = logvar.clamp(min=-6.0, max=4.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std if self.training else mu
            future_z.append(z)
            hidden_seq.append(hidden)
        return {
            "future_z": torch.stack(future_z, dim=1),
            "hidden_seq": torch.stack(hidden_seq, dim=1),
        }
