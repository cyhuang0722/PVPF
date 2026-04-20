from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cloud_decoder import FutureCloudStateDecoder
from .image_encoder import ImageEncoder
from .latent_state import StructuredLatentState
from .power_head import SunConditionedPowerHead
from .stochastic_dynamics import VariationalGRUDynamics
from .temporal_encoder import TemporalCloudStateEncoder


class SunConditionedStochasticCloudModel(nn.Module):
    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.future_steps = int(model_cfg.get("future_steps", 15))
        self.feature_hw = int(model_cfg.get("feature_resolution", 32))
        self.sun_sigma = float(model_cfg.get("sun_attention_sigma", 2.5))
        self.spatial_feature_dim = int(model_cfg.get("spatial_feature_dim", 128))
        self.disable_sun_attention = bool(model_cfg.get("disable_sun_attention", False))

        encoder_channels = list(model_cfg.get("frame_channels", [32, 64, 96, 128]))
        temporal_dim = int(model_cfg.get("temporal_hidden_dim", 256))
        latent_dims = {
            "dynamics": int(model_cfg.get("dynamics_latent_dim", 64)),
        }
        latent_total_dim = sum(latent_dims.values())
        dynamics_hidden = int(model_cfg.get("dynamics_hidden_dim", 128))

        self.image_encoder = ImageEncoder(in_channels=int(model_cfg.get("input_channels", 6)), channels=encoder_channels)
        self.temporal_encoder = TemporalCloudStateEncoder(input_dim=encoder_channels[-1], hidden_dim=temporal_dim)
        self.latent_state = StructuredLatentState(
            in_dim=temporal_dim,
            latent_dims=latent_dims,
            spatial_dim=self.spatial_feature_dim,
        )
        self.dynamics = VariationalGRUDynamics(latent_dim=latent_total_dim, hidden_dim=dynamics_hidden, future_steps=self.future_steps)
        self.cloud_decoder = FutureCloudStateDecoder(
            spatial_dim=self.spatial_feature_dim,
            latent_dim=latent_total_dim,
            hidden_dim=dynamics_hidden,
            feature_hw=self.feature_hw,
        )
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(temporal_dim, temporal_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(temporal_dim // 2, 1, kernel_size=1),
        )
        self.power_head = SunConditionedPowerHead(
            cloud_dim=temporal_dim,
            pv_history_dim=int(model_cfg.get("pv_history_dim", 4)),
            solar_dim=int(model_cfg.get("solar_feature_dim", 5)),
            hidden_dim=int(model_cfg.get("pv_hidden_dim", 64)),
            use_global_cloud=bool(model_cfg.get("use_global_cloud_in_head", False)),
        )

    def forward(
        self,
        images: torch.Tensor,
        pv_history: torch.Tensor,
        solar_vec: torch.Tensor,
        sun_xy: torch.Tensor,
        target_sun_xy: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, seq_len, channels, height, width = images.shape
        encoded = self.image_encoder(images.view(batch * seq_len, channels, height, width))
        encoded = encoded.view(batch, seq_len, encoded.shape[1], encoded.shape[2], encoded.shape[3])

        temporal_out = self.temporal_encoder(encoded)
        cloud_feature = temporal_out["spatial_feat"]
        latent = self.latent_state(cloud_feature)
        z0 = latent["samples"]["dynamics"]

        dynamics = self.dynamics(z0)
        decoded = self.cloud_decoder(
            dynamics["future_z"],
            dynamics["hidden_seq"],
            spatial_feat=latent["spatial_feat"],
        )

        target_sun_xy = target_sun_xy if target_sun_xy is not None else sun_xy
        attention_map = self._build_sun_attention(target_sun_xy, decoded["future_cloud_prob_maps"].shape[-2], decoded["future_cloud_prob_maps"].shape[-1], height, width)
        future_cloud_prob_15min = decoded["future_cloud_prob_maps"].mean(dim=1)
        change_hotspot = decoded["future_change_hotspot_maps"]
        change_hotspot_15min = change_hotspot.mean(dim=1)
        cloud_uncertainty = (0.65 * (4.0 * decoded["future_cloud_prob_maps"] * (1.0 - decoded["future_cloud_prob_maps"])) + 0.35 * change_hotspot).clamp(0.0, 1.0)
        cloud_uncertainty_15min = cloud_uncertainty.mean(dim=1)
        attention_norm = attention_map.sum(dim=(2, 3)).clamp_min(1e-6)
        sun_local_cloud_prob = (future_cloud_prob_15min * attention_map).sum(dim=(2, 3)) / attention_norm
        sun_local_change_hotspot = (change_hotspot_15min * attention_map).sum(dim=(2, 3)) / attention_norm
        future_sun_cloud_prob = (
            decoded["future_cloud_prob_maps"] * attention_map.unsqueeze(1)
        ).sum(dim=(3, 4)).squeeze(-1) / attention_norm.unsqueeze(1).squeeze(-1)
        global_cloud_prob = future_cloud_prob_15min.mean(dim=(2, 3))
        sun_hotspot_risk = (0.75 * sun_local_cloud_prob + 0.25 * sun_local_change_hotspot).clamp(0.0, 1.0)
        sun_occlusion_risk = torch.maximum(decoded["sun_occlusion"].mean(dim=1, keepdim=True), sun_hotspot_risk)

        prediction = self.power_head(
            pooled_cloud=temporal_out["global_feat"],
            pv_history=pv_history,
            solar_vec=solar_vec,
            sun_local_cloud_prob=sun_local_cloud_prob,
            global_cloud_prob=global_cloud_prob,
            sun_local_change_hotspot=sun_local_change_hotspot,
            sun_occlusion_risk=sun_occlusion_risk,
        )
        recon_rbr = torch.sigmoid(self.reconstruction_head(cloud_feature))
        return {
            "prediction": prediction,
            "sun_occlusion": decoded["sun_occlusion"],
            "future_cloud_prob_maps": decoded["future_cloud_prob_maps"],
            "future_cloud_prob_15min": future_cloud_prob_15min,
            "future_change_hotspot_maps": change_hotspot,
            "future_change_hotspot_15min": change_hotspot_15min,
            "future_cloud_uncertainty_maps": cloud_uncertainty,
            "future_cloud_uncertainty_15min": cloud_uncertainty_15min,
            "future_sun_cloud_prob": future_sun_cloud_prob,
            "current_cloud_prob": decoded["current_cloud_prob"],
            "attention_map": attention_map,
            "recon_rbr": recon_rbr,
            "kl_loss": latent["kl_loss"],
        }

    def _build_sun_attention(
        self,
        sun_xy: torch.Tensor,
        feat_h: int,
        feat_w: int,
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        if self.disable_sun_attention:
            return torch.ones(
                (sun_xy.shape[0], 1, feat_h, feat_w),
                device=sun_xy.device,
                dtype=torch.float32,
            )
        device = sun_xy.device
        yy, xx = torch.meshgrid(
            torch.arange(feat_h, device=device, dtype=torch.float32),
            torch.arange(feat_w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        sx = sun_xy[:, 0:1, None] * (feat_w / float(image_w))
        sy = sun_xy[:, 1:2, None] * (feat_h / float(image_h))
        dist_sq = (xx.unsqueeze(0) - sx) ** 2 + (yy.unsqueeze(0) - sy) ** 2
        attention = torch.exp(-dist_sq / (2.0 * self.sun_sigma**2))
        attention = attention / attention.flatten(1).sum(dim=-1, keepdim=True).view(-1, 1, 1).clamp_min(1e-6)
        return attention.unsqueeze(1)
