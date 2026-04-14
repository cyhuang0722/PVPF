from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list[int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = hidden_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            ResidualBlock(c1),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            ResidualBlock(c2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            ResidualBlock(c3),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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


class StructuredLatentHead(nn.Module):
    def __init__(self, in_dim: int, latent_dims: dict[str, int]) -> None:
        super().__init__()
        self.latent_dims = latent_dims
        self.heads = nn.ModuleDict()
        for name, dim in latent_dims.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(in_dim, max(in_dim, dim * 2)),
                nn.GELU(),
                nn.Linear(max(in_dim, dim * 2), dim * 2),
            )

    def forward(self, pooled: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, head in self.heads.items():
            stats = head(pooled)
            mu, logvar = torch.chunk(stats, 2, dim=-1)
            outputs[name] = (mu, logvar.clamp(min=-6.0, max=4.0))
        return outputs


class QuantileHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.center = nn.Linear(hidden_dim // 2, 1)
        self.lower_width = nn.Linear(hidden_dim // 2, 1)
        self.upper_width = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        q50 = self.center(feat)
        lower = q50 - F.softplus(self.lower_width(feat))
        upper = q50 + F.softplus(self.upper_width(feat))
        return torch.cat([lower, q50, upper], dim=-1)


class SunConditionedStochasticCloudModel(nn.Module):
    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.future_steps = int(model_cfg.get("future_steps", 15))
        self.feature_hw = int(model_cfg.get("feature_resolution", 32))
        self.max_motion = float(model_cfg.get("max_motion_displacement", 2.5))
        self.sun_sigma = float(model_cfg.get("sun_attention_sigma", 2.5))

        encoder_channels = list(model_cfg.get("frame_channels", [32, 64, 96, 128]))
        latent_dims = {
            "motion": int(model_cfg.get("motion_latent_dim", 64)),
            "opacity": int(model_cfg.get("opacity_latent_dim", 32)),
            "gap": int(model_cfg.get("gap_latent_dim", 32)),
            "sun": int(model_cfg.get("sun_latent_dim", 16)),
        }
        self.latent_total_dim = sum(latent_dims.values())
        temporal_dim = int(model_cfg.get("temporal_hidden_dim", 256))
        dynamics_hidden = int(model_cfg.get("dynamics_hidden_dim", 128))
        pv_hidden = int(model_cfg.get("pv_hidden_dim", 64))

        self.image_encoder = ImageEncoder(in_channels=int(model_cfg.get("input_channels", 6)), hidden_channels=encoder_channels)
        self.temporal_encoder = ConvLSTMCell(input_dim=encoder_channels[-1], hidden_dim=temporal_dim)
        self.temporal_refine = ResidualBlock(temporal_dim)
        self.latent_head = StructuredLatentHead(in_dim=temporal_dim, latent_dims=latent_dims)

        self.hidden_init = nn.Linear(self.latent_total_dim, dynamics_hidden)
        self.dynamics_cell = nn.GRUCell(self.latent_total_dim, dynamics_hidden)
        self.transition_head = nn.Sequential(
            nn.Linear(dynamics_hidden, dynamics_hidden),
            nn.GELU(),
            nn.Linear(dynamics_hidden, self.latent_total_dim * 2),
        )

        decoded_dim = self.latent_total_dim + dynamics_hidden
        self.motion_decoder = nn.Sequential(
            nn.Linear(decoded_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2 * self.feature_hw * self.feature_hw),
        )
        self.opacity_decoder = nn.Sequential(
            nn.Linear(decoded_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.feature_hw * self.feature_hw),
        )
        self.gap_decoder = nn.Sequential(
            nn.Linear(decoded_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.feature_hw * self.feature_hw),
        )
        self.sun_occ_decoder = nn.Sequential(
            nn.Linear(decoded_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(temporal_dim, temporal_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(temporal_dim // 2, 1, kernel_size=1),
        )

        self.pv_encoder = nn.Sequential(
            nn.Linear(int(model_cfg.get("pv_history_dim", 4)), pv_hidden),
            nn.GELU(),
            nn.Linear(pv_hidden, pv_hidden),
            nn.GELU(),
        )
        self.quantile_head = QuantileHead(in_dim=temporal_dim + pv_hidden + 5 + 4, hidden_dim=int(model_cfg.get("head_hidden_dim", 128)))

    def forward(
        self,
        images: torch.Tensor,
        pv_history: torch.Tensor,
        solar_vec: torch.Tensor,
        sun_xy: torch.Tensor,
        target_sun_xy: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, seq_len, channels, height, width = images.shape
        feats = self.image_encoder(images.view(batch * seq_len, channels, height, width))
        _, feat_channels, feat_h, feat_w = feats.shape
        feats = feats.view(batch, seq_len, feat_channels, feat_h, feat_w)

        state: tuple[torch.Tensor, torch.Tensor] | None = None
        for step in range(seq_len):
            state = self.temporal_encoder(feats[:, step], state)
        h_state, _ = state
        h_state = self.temporal_refine(h_state)
        pooled = F.adaptive_avg_pool2d(h_state, output_size=1).flatten(1)

        latent_stats = self.latent_head(pooled)
        z_parts = []
        kl_loss = pooled.new_zeros(batch)
        for mu, logvar in latent_stats.values():
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample = mu + eps * std if self.training else mu
            z_parts.append(sample)
            kl_loss = kl_loss + 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
        z = torch.cat(z_parts, dim=-1)

        hidden = torch.tanh(self.hidden_init(z))
        motion_fields = []
        opacity_maps = []
        gap_maps = []
        transmission_maps = []
        sun_occlusion = []

        current_z = z
        for _ in range(self.future_steps):
            hidden = self.dynamics_cell(current_z, hidden)
            trans_stats = self.transition_head(hidden)
            mu, logvar = torch.chunk(trans_stats, 2, dim=-1)
            logvar = logvar.clamp(min=-6.0, max=4.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            current_z = mu + eps * std if self.training else mu
            decoded = torch.cat([current_z, hidden], dim=-1)

            motion = torch.tanh(self.motion_decoder(decoded)).view(batch, 2, feat_h, feat_w) * self.max_motion
            opacity = torch.sigmoid(self.opacity_decoder(decoded)).view(batch, 1, feat_h, feat_w)
            gap = torch.sigmoid(self.gap_decoder(decoded)).view(batch, 1, feat_h, feat_w)
            sun_occ = torch.sigmoid(self.sun_occ_decoder(decoded)).squeeze(-1)
            transmission = torch.clamp((1.0 - opacity) * (0.25 + 0.75 * gap), 0.0, 1.0)

            motion_fields.append(motion)
            opacity_maps.append(opacity)
            gap_maps.append(gap)
            sun_occlusion.append(sun_occ)
            transmission_maps.append(transmission)

        motion_fields_t = torch.stack(motion_fields, dim=1)
        opacity_maps_t = torch.stack(opacity_maps, dim=1)
        gap_maps_t = torch.stack(gap_maps, dim=1)
        transmission_maps_t = torch.stack(transmission_maps, dim=1)
        sun_occlusion_t = torch.stack(sun_occlusion, dim=1)

        target_sun_xy = target_sun_xy if target_sun_xy is not None else sun_xy
        attention_map = self._build_sun_attention(target_sun_xy, feat_h=feat_h, feat_w=feat_w, image_h=height, image_w=width)
        final_transmission = transmission_maps_t[:, -1]
        final_opacity = opacity_maps_t[:, -1]
        final_gap = gap_maps_t[:, -1]

        sun_local_transmission = (final_transmission * attention_map).sum(dim=(2, 3)) / attention_map.sum(dim=(2, 3)).clamp_min(1e-6)
        sun_local_gap = (final_gap * attention_map).sum(dim=(2, 3)) / attention_map.sum(dim=(2, 3)).clamp_min(1e-6)
        sun_local_opacity = (final_opacity * attention_map).sum(dim=(2, 3)) / attention_map.sum(dim=(2, 3)).clamp_min(1e-6)

        pv_feat = self.pv_encoder(pv_history)
        head_features = torch.cat(
            [
                pooled,
                pv_feat,
                solar_vec,
                sun_local_transmission,
                sun_local_gap,
                sun_local_opacity,
                sun_occlusion_t[:, -1:].contiguous(),
            ],
            dim=-1,
        )
        prediction = self.quantile_head(head_features)
        recon_rbr = torch.sigmoid(self.reconstruction_head(h_state))

        motion_reg = self._motion_smoothness(motion_fields_t)
        return {
            "prediction": prediction,
            "motion_fields": motion_fields_t,
            "opacity_maps": opacity_maps_t,
            "gap_maps": gap_maps_t,
            "transmission_maps": transmission_maps_t,
            "sun_occlusion": sun_occlusion_t,
            "attention_map": attention_map,
            "sun_prior": attention_map,
            "recon_rbr": recon_rbr,
            "kl_loss": kl_loss.mean(),
            "motion_reg_loss": motion_reg,
        }

    def _build_sun_attention(
        self,
        sun_xy: torch.Tensor,
        feat_h: int,
        feat_w: int,
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        device = sun_xy.device
        yy, xx = torch.meshgrid(
            torch.arange(feat_h, device=device, dtype=torch.float32),
            torch.arange(feat_w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        scale_x = feat_w / float(image_w)
        scale_y = feat_h / float(image_h)
        sx = sun_xy[:, 0:1, None] * scale_x
        sy = sun_xy[:, 1:2, None] * scale_y
        dist_sq = (xx.unsqueeze(0) - sx) ** 2 + (yy.unsqueeze(0) - sy) ** 2
        attention = torch.exp(-dist_sq / (2.0 * self.sun_sigma**2))
        attention = attention / attention.flatten(1).sum(dim=-1, keepdim=True).view(-1, 1, 1).clamp_min(1e-6)
        return attention.unsqueeze(1)

    @staticmethod
    def _motion_smoothness(motion_fields: torch.Tensor) -> torch.Tensor:
        dx = motion_fields[..., :, 1:] - motion_fields[..., :, :-1]
        dy = motion_fields[..., 1:, :] - motion_fields[..., :-1, :]
        return dx.abs().mean() + dy.abs().mean()
