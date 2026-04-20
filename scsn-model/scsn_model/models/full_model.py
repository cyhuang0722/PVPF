from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .latent_state import StructuredLatentState
from .power_head import SunConditionedPowerHead
from .rbr_decoder import FutureRBRDistributionDecoder
from .stochastic_dynamics import VariationalGRUDynamics
from .temporal_encoder import TemporalRBRStateEncoder


class SunConditionedStochasticRBRModel(nn.Module):
    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.future_steps = int(model_cfg.get("future_steps", 15))
        self.feature_hw = int(model_cfg.get("feature_resolution", 32))
        self.sun_sigma = float(model_cfg.get("sun_attention_sigma", 2.5))
        self.spatial_feature_dim = int(model_cfg.get("spatial_feature_dim", 128))

        encoder_channels = list(model_cfg.get("frame_channels", [32, 64, 96, 128]))
        temporal_dim = int(model_cfg.get("temporal_hidden_dim", 256))
        latent_dims = {
            "dynamics": int(model_cfg.get("dynamics_latent_dim", 64)),
        }
        latent_total_dim = sum(latent_dims.values())
        dynamics_hidden = int(model_cfg.get("dynamics_hidden_dim", 128))

        self.image_encoder = ImageEncoder(in_channels=int(model_cfg.get("input_channels", 6)), channels=encoder_channels)
        self.temporal_encoder = TemporalRBRStateEncoder(input_dim=encoder_channels[-1], hidden_dim=temporal_dim)
        self.latent_state = StructuredLatentState(
            in_dim=temporal_dim,
            latent_dims=latent_dims,
            spatial_dim=self.spatial_feature_dim,
        )
        self.dynamics = VariationalGRUDynamics(latent_dim=latent_total_dim, hidden_dim=dynamics_hidden, future_steps=self.future_steps)
        self.rbr_decoder = FutureRBRDistributionDecoder(
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
            pv_history_dim=int(model_cfg.get("pv_history_dim", 4)),
            solar_dim=int(model_cfg.get("solar_feature_dim", 5)),
            hidden_dim=int(model_cfg.get("pv_hidden_dim", 64)),
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
        temporal_feature = temporal_out["spatial_feat"]
        latent = self.latent_state(temporal_feature)
        z0 = latent["samples"]["dynamics"]

        dynamics = self.dynamics(z0)
        decoded = self.rbr_decoder(
            dynamics["future_z"],
            dynamics["hidden_seq"],
            spatial_feat=latent["spatial_feat"],
        )

        target_sun_xy = target_sun_xy if target_sun_xy is not None else sun_xy
        rbr_mean = decoded["future_rbr_mean_maps"]
        rbr_logvar = decoded["future_rbr_logvar_maps"]
        rbr_variance = torch.exp(rbr_logvar)
        rbr_mean_15min = rbr_mean.mean(dim=1)
        rbr_variance_15min = rbr_variance.mean(dim=1)
        rbr_logvar_15min = torch.log(rbr_variance_15min.clamp_min(1e-6))
        attention_map = self._build_sun_attention(target_sun_xy, rbr_mean_15min.shape[-2], rbr_mean_15min.shape[-1], height, width)
        attention_norm = attention_map.sum(dim=(2, 3)).clamp_min(1e-6)
        sun_local_rbr_mean = (rbr_mean_15min * attention_map).sum(dim=(2, 3)) / attention_norm
        sun_local_rbr_variance = (rbr_variance_15min * attention_map).sum(dim=(2, 3)) / attention_norm
        global_rbr_mean = rbr_mean_15min.mean(dim=(2, 3))
        global_rbr_variance = rbr_variance_15min.mean(dim=(2, 3))

        power = self.power_head(
            pv_history=pv_history,
            solar_vec=solar_vec,
            global_rbr_mean=global_rbr_mean,
            sun_local_rbr_mean=sun_local_rbr_mean,
            global_rbr_variance=global_rbr_variance,
            sun_local_rbr_variance=sun_local_rbr_variance,
        )
        recon_rbr = torch.sigmoid(self.reconstruction_head(temporal_feature))
        return {
            "prediction": power["prediction"],
            "pv_mu": power["pv_mu"],
            "pv_logvar": power["pv_logvar"],
            "pv_sigma": power["pv_sigma"],
            "future_rbr_mean_maps": rbr_mean,
            "future_rbr_logvar_maps": rbr_logvar,
            "future_rbr_variance_maps": rbr_variance,
            "future_rbr_mean_15min": rbr_mean_15min,
            "future_rbr_logvar_15min": rbr_logvar_15min,
            "future_rbr_variance_15min": rbr_variance_15min,
            "global_rbr_mean": global_rbr_mean,
            "sun_local_rbr_mean": sun_local_rbr_mean,
            "global_rbr_variance": global_rbr_variance,
            "sun_local_rbr_variance": sun_local_rbr_variance,
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
