from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forecast_head import DeterministicForecastHead
from .frame_encoder import ConvBlock, FrameEncoder
from .pv_encoder import PVHistoryEncoder


class DualTimescaleSunAwarePVModel(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        channels = model_cfg["frame_channels"]
        feature_dim = channels[-1]
        motion_in_channels = int(model_cfg.get("motion_input_channels", 15))
        sun_hidden_dim = int(model_cfg.get("sun_hidden_dim", 64))
        fusion_hidden_dim = int(model_cfg.get("fusion_hidden_dim", 128))
        pv_hidden_dim = int(model_cfg.get("pv_hidden_dim", 64))
        self.patch_grid_size = int(model_cfg.get("patch_grid_size", 8))

        self.context_encoder = FrameEncoder(channels)
        self.motion_encoder = nn.Sequential(
            ConvBlock(motion_in_channels, channels[0], stride=2),
            ConvBlock(channels[0], channels[1], stride=2),
            ConvBlock(channels[1], channels[2], stride=2),
            ConvBlock(channels[2], channels[3], stride=1),
            nn.Conv2d(channels[3], channels[3], kernel_size=1),
        )
        self.context_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.motion_modulator = nn.Sequential(
            ConvBlock(feature_dim * 2 + 1, feature_dim, stride=1),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.GELU(),
        )
        self.global_motion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )
        self.patch_motion_head = nn.Conv2d(feature_dim, 2, kernel_size=1)
        self.sun_mlp = nn.Sequential(
            nn.Linear(2, sun_hidden_dim),
            nn.GELU(),
            nn.Linear(sun_hidden_dim, sun_hidden_dim),
            nn.GELU(),
        )
        self.pv_encoder = PVHistoryEncoder(in_dim=4, hidden_dim=pv_hidden_dim)
        self.head = DeterministicForecastHead(
            in_dim=feature_dim * 2 + sun_hidden_dim + pv_hidden_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=float(model_cfg["dropout"]),
            out_dim=1,
        )
        self.max_target_value = float(model_cfg.get("max_target_value", 1.2))
        self.sun_spot_sigma = float(model_cfg.get("sun_spot_sigma", model_cfg.get("sun_gaussian_sigma", 4.0)))
        self.sun_corridor_width = float(model_cfg.get("sun_corridor_width", 3.0))
        self.sun_corridor_length = float(model_cfg.get("sun_corridor_length", 10.0))

    def _build_sun_prior(
        self,
        sun_xy: torch.Tensor,
        image_hw: tuple[int, int],
        feat_hw: tuple[int, int],
        global_motion_vec: torch.Tensor,
    ) -> torch.Tensor:
        bsz = sun_xy.shape[0]
        feat_h, feat_w = feat_hw
        image_h, image_w = image_hw
        yy, xx = torch.meshgrid(
            torch.arange(feat_h, device=sun_xy.device, dtype=torch.float32),
            torch.arange(feat_w, device=sun_xy.device, dtype=torch.float32),
            indexing="ij",
        )
        scale_x = feat_w / float(image_w)
        scale_y = feat_h / float(image_h)
        sun_x = sun_xy[:, 0].view(bsz, 1, 1) * scale_x
        sun_y = sun_xy[:, 1].view(bsz, 1, 1) * scale_y
        rel_x = xx.unsqueeze(0) - sun_x
        rel_y = yy.unsqueeze(0) - sun_y
        dist2 = rel_x ** 2 + rel_y ** 2
        spot = torch.exp(-dist2 / (2.0 * self.sun_spot_sigma ** 2))

        upwind = -F.normalize(global_motion_vec, dim=1, eps=1e-6)
        dir_x = upwind[:, 0].view(bsz, 1, 1)
        dir_y = upwind[:, 1].view(bsz, 1, 1)
        perp_x = -dir_y
        perp_y = dir_x
        along = rel_x * dir_x + rel_y * dir_y
        perp = rel_x * perp_x + rel_y * perp_y
        corridor = torch.exp(-(perp ** 2) / (2.0 * self.sun_corridor_width ** 2))
        corridor = corridor * torch.exp(-((along - self.sun_corridor_length * 0.35) ** 2) / (2.0 * self.sun_corridor_length ** 2))
        corridor = corridor * torch.sigmoid(along)

        prior = torch.clamp(spot + 0.7 * corridor, 0.0, 1.0)
        return prior.unsqueeze(1)

    def forward(self, images: torch.Tensor, pv_history: torch.Tensor, sun_angles: torch.Tensor, sun_xy: torch.Tensor) -> dict:
        bsz, seq_len, ch, img_h, img_w = images.shape
        if seq_len != 8:
            raise ValueError(f"DualTimescaleSunAwarePVModel expects 8 frames, got {seq_len}")

        ctx = images[:, :5]
        mot = images[:, 5:]

        ctx_feat = self.context_encoder(ctx.reshape(bsz * 5, ch, img_h, img_w))
        _, feat_ch, feat_h, feat_w = ctx_feat.shape
        ctx_feat = ctx_feat.view(bsz, 5, feat_ch, feat_h, feat_w)
        f_ctx_map = ctx_feat.mean(dim=1)
        f_ctx_global = F.adaptive_avg_pool2d(f_ctx_map, 1).flatten(1)

        d1 = mot[:, 1] - mot[:, 0]
        d2 = mot[:, 2] - mot[:, 1]
        motion_input = torch.cat([mot[:, 0], mot[:, 1], mot[:, 2], d1, d2], dim=1)
        f_mot_map = self.motion_encoder(motion_input)
        global_motion_vec = self.global_motion_head(f_mot_map)

        f_sun = self.sun_mlp(sun_angles)
        a_sun = self._build_sun_prior(
            sun_xy=sun_xy,
            image_hw=(img_h, img_w),
            feat_hw=(feat_h, feat_w),
            global_motion_vec=global_motion_vec,
        )
        mod_input = torch.cat([f_mot_map, a_sun, self.context_proj(f_ctx_map)], dim=1)
        f_mot_mod = self.motion_modulator(mod_input)
        f_mot_global = F.adaptive_avg_pool2d(f_mot_mod, 1).flatten(1)

        pv_feat = self.pv_encoder(pv_history)
        pred_raw = self.head(torch.cat([f_ctx_global, f_mot_global, f_sun, pv_feat], dim=1))
        prediction = self.max_target_value * torch.sigmoid(pred_raw)

        patch_motion = self.patch_motion_head(F.adaptive_avg_pool2d(f_mot_mod, (self.patch_grid_size, self.patch_grid_size)))
        patch_motion = patch_motion.flatten(2).transpose(1, 2)
        patch_motion = F.normalize(patch_motion, dim=-1, eps=1e-6)
        return {
            "prediction": prediction,
            "patch_motion_pred": patch_motion,
            "context_map": f_ctx_map,
            "motion_map": f_mot_map,
            "modulated_motion_map": f_mot_mod,
            "sun_prior": a_sun,
            "attention_map": a_sun.squeeze(1),
            "global_motion_vector": global_motion_vec,
            "frame_features": ctx_feat,
        }
