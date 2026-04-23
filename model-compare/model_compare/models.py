from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallFrameEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.Conv2d(72, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = int(hidden_channels)
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.cell = ConvLSTMCell(32, hidden_channels)
        self.head = RegressionHead(hidden_channels, dropout)

    def forward(self, images: torch.Tensor, history_x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = images.shape
        h = c = None
        for step in range(steps):
            x = self.stem(images[:, step])
            if h is None:
                shape = (batch, self.cell.hidden_channels, x.shape[-2], x.shape[-1])
                h = x.new_zeros(shape)
                c = x.new_zeros(shape)
            h, c = self.cell(x, (h, c))
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.head(pooled)


class PvHistoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 11, embed_dim: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CnnGruBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 96, hidden_dim: int = 96, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = SmallFrameEncoder(in_channels, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(self, images: torch.Tensor, history_x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = images.shape
        emb = self.encoder(images.reshape(batch * steps, channels, height, width)).view(batch, steps, -1)
        _, hidden = self.gru(emb)
        return self.head(hidden[-1])


class ImageRegressorBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = SmallFrameEncoder(in_channels, embed_dim)
        self.head = RegressionHead(embed_dim, dropout)

    def forward(self, images: torch.Tensor, history_x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        latest = images[:, -1]
        return self.head(self.encoder(latest))


class ConvLSTMPvBaseline(ConvLSTMBaseline):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 48, history_dim: int = 11, history_embed_dim: int = 48, dropout: float = 0.2) -> None:
        nn.Module.__init__(self)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.cell = ConvLSTMCell(32, hidden_channels)
        self.history = PvHistoryEncoder(history_dim, history_embed_dim, dropout)
        self.head = RegressionHead(hidden_channels + history_embed_dim, dropout)

    def forward(self, images: torch.Tensor, history_x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = images.shape
        h = c = None
        for step in range(steps):
            x = self.stem(images[:, step])
            if h is None:
                shape = (batch, self.cell.hidden_channels, x.shape[-2], x.shape[-1])
                h = x.new_zeros(shape)
                c = x.new_zeros(shape)
            h, c = self.cell(x, (h, c))
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        if history_x is None:
            history_x = pooled.new_zeros((pooled.shape[0], 11))
        return self.head(torch.cat([pooled, self.history(history_x)], dim=-1))


class CnnGruPvBaseline(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        hidden_dim: int = 96,
        history_dim: int = 11,
        history_embed_dim: int = 48,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = SmallFrameEncoder(in_channels, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.history = PvHistoryEncoder(history_dim, history_embed_dim, dropout)
        self.head = RegressionHead(hidden_dim + history_embed_dim, dropout)

    def forward(self, images: torch.Tensor, history_x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = images.shape
        emb = self.encoder(images.reshape(batch * steps, channels, height, width)).view(batch, steps, -1)
        _, hidden = self.gru(emb)
        temporal = hidden[-1]
        if history_x is None:
            history_x = temporal.new_zeros((temporal.shape[0], 11))
        return self.head(torch.cat([temporal, self.history(history_x)], dim=-1))


class ImageRegressorPvBaseline(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        history_dim: int = 11,
        history_embed_dim: int = 48,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = SmallFrameEncoder(in_channels, embed_dim)
        self.history = PvHistoryEncoder(history_dim, history_embed_dim, dropout)
        self.head = RegressionHead(embed_dim + history_embed_dim, dropout)

    def forward(self, images: torch.Tensor, history_x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        visual = self.encoder(images[:, -1])
        if history_x is None:
            history_x = visual.new_zeros((visual.shape[0], 11))
        return self.head(torch.cat([visual, self.history(history_x)], dim=-1))


class VaeFrameEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.Conv2d(72, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mu = nn.Linear(96, latent_dim)
        self.logvar = nn.Linear(96, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


class VaeFrameDecoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int, image_size: int = 96) -> None:
        super().__init__()
        if int(image_size) != 96:
            raise ValueError("VaeFrameDecoder currently expects image_size=96")
        self.proj = nn.Sequential(nn.Linear(latent_dim, 96 * 6 * 6), nn.GELU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(96, 72, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.ConvTranspose2d(72, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.ConvTranspose2d(24, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(z.shape[0], 96, 6, 6)
        return self.net(h)


class VaeRegressorBaseline(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 48,
        temporal_hidden_dim: int = 64,
        dropout: float = 0.25,
        image_size: int = 96,
    ) -> None:
        super().__init__()
        self.encoder = VaeFrameEncoder(in_channels, latent_dim)
        self.decoder = VaeFrameDecoder(in_channels, latent_dim, image_size=image_size)
        self.temporal = nn.GRU(latent_dim, temporal_hidden_dim, batch_first=True)
        self.head = RegressionHead(temporal_hidden_dim, dropout)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, images: torch.Tensor, history_x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = images.shape
        flat = images.reshape(batch * steps, channels, height, width)
        mu, logvar = self.encoder(flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).view(batch, steps, channels, height, width)
        z_seq = z.view(batch, steps, -1)
        _, hidden = self.temporal(z_seq)
        out = self.head(hidden[-1])
        out.update(
            {
                "recon": recon,
                "mu": mu.view(batch, steps, -1),
                "logvar": logvar.view(batch, steps, -1),
            }
        )
        return out


class VaeRegressorPvBaseline(VaeRegressorBaseline):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 48,
        temporal_hidden_dim: int = 64,
        history_dim: int = 11,
        history_embed_dim: int = 48,
        dropout: float = 0.25,
        image_size: int = 96,
    ) -> None:
        nn.Module.__init__(self)
        self.encoder = VaeFrameEncoder(in_channels, latent_dim)
        self.decoder = VaeFrameDecoder(in_channels, latent_dim, image_size=image_size)
        self.temporal = nn.GRU(latent_dim, temporal_hidden_dim, batch_first=True)
        self.history = PvHistoryEncoder(history_dim, history_embed_dim, dropout)
        self.head = RegressionHead(temporal_hidden_dim + history_embed_dim, dropout)

    def forward(self, images: torch.Tensor, history_x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = images.shape
        flat = images.reshape(batch * steps, channels, height, width)
        mu, logvar = self.encoder(flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).view(batch, steps, channels, height, width)
        z_seq = z.view(batch, steps, -1)
        _, hidden = self.temporal(z_seq)
        temporal = hidden[-1]
        if history_x is None:
            history_x = temporal.new_zeros((temporal.shape[0], 11))
        out = self.head(torch.cat([temporal, self.history(history_x)], dim=-1))
        out.update(
            {
                "recon": recon,
                "mu": mu.view(batch, steps, -1),
                "logvar": logvar.view(batch, steps, -1),
            }
        )
        return out


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 64),
            nn.GELU(),
        )
        self.loc = nn.Linear(64, 1)
        self.scale = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x)
        loc = torch.sigmoid(self.loc(h)) * 1.25
        scale = 0.015 + 0.60 * torch.sigmoid(self.scale(h))
        return {"loc": loc, "scale": scale}


def build_model(model_name: str, cfg: dict) -> nn.Module:
    name = str(model_name).strip().lower().replace("-", "_")
    if name == "convlstm":
        return ConvLSTMBaseline(**cfg.get("convlstm", {}))
    if name == "cnn_gru":
        return CnnGruBaseline(**cfg.get("cnn_gru", {}))
    if name == "image_regressor":
        return ImageRegressorBaseline(**cfg.get("image_regressor", {}))
    if name == "vae_regressor":
        return VaeRegressorBaseline(**cfg.get("vae_regressor", {}))
    if name == "convlstm_pv":
        return ConvLSTMPvBaseline(**cfg.get("convlstm_pv", {}))
    if name == "cnn_gru_pv":
        return CnnGruPvBaseline(**cfg.get("cnn_gru_pv", {}))
    if name == "image_regressor_pv":
        return ImageRegressorPvBaseline(**cfg.get("image_regressor_pv", {}))
    if name == "vae_regressor_pv":
        return VaeRegressorPvBaseline(**cfg.get("vae_regressor_pv", {}))
    raise ValueError(f"unknown model: {model_name}")


def gaussian_nll(loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.view_as(loc)
    dist = torch.distributions.Normal(loc=loc, scale=scale.clamp_min(1e-4))
    return -dist.log_prob(target).mean()


def vae_loss(out: dict[str, torch.Tensor], images: torch.Tensor, recon_weight: float, kl_weight: float) -> torch.Tensor:
    if "recon" not in out:
        return images.new_tensor(0.0)
    recon = F.mse_loss(out["recon"], images)
    mu = out["mu"]
    logvar = out["logvar"]
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return float(recon_weight) * recon + float(kl_weight) * kl
