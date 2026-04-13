from __future__ import annotations

import torch
import torch.nn as nn

from .forecast_head import DeterministicForecastHead
from .frame_encoder import FrameEncoder
from .motion_aggregator import MotionAggregator
from .motion_field_head import MotionFieldHead
from .pv_encoder import PVHistoryEncoder
from .sun_attention import SunConditionedAttention


class MinimalSunConditionedPVModel(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        channels = model_cfg["frame_channels"]
        feature_dim = channels[-1]
        motion_feature_dim = model_cfg["motion_feature_dim"]
        attention_dim = model_cfg["attention_dim"]

        self.frame_encoder = FrameEncoder(channels)
        self.motion_head = MotionFieldHead(
            feature_dim=feature_dim,
            hidden_dims=model_cfg["motion_hidden"],
            max_displacement=float(model_cfg["max_motion_displacement"]),
        )
        self.motion_aggregator = MotionAggregator(out_dim=motion_feature_dim)
        self.sun_attention = SunConditionedAttention(
            visual_dim=feature_dim + motion_feature_dim,
            sun_dim=5,
            attention_dim=attention_dim,
        )
        self.pv_encoder = PVHistoryEncoder(in_dim=4, hidden_dim=model_cfg["pv_hidden_dim"])
        self.head = DeterministicForecastHead(
            in_dim=attention_dim + model_cfg["pv_hidden_dim"],
            hidden_dim=model_cfg["fusion_hidden_dim"],
            dropout=float(model_cfg["dropout"]),
            out_dim=1,
        )

    def forward(self, images: torch.Tensor, pv_history: torch.Tensor, solar_vec: torch.Tensor) -> dict:
        bsz, seq_len, ch, h, w = images.shape
        feats = self.frame_encoder(images.view(bsz * seq_len, ch, h, w))
        _, feat_ch, feat_h, feat_w = feats.shape
        feats = feats.view(bsz, seq_len, feat_ch, feat_h, feat_w)

        flows = []
        for idx in range(1, seq_len):
            flow, _ = self.motion_head(feats[:, idx - 1], feats[:, idx])
            flows.append(flow)
        motion_fields = torch.stack(flows, dim=1)
        motion_feature = self.motion_aggregator(motion_fields)

        last_feature = feats[:, -1]
        fused_map = torch.cat([last_feature, motion_feature], dim=1)
        sun_feature, attention_map = self.sun_attention(fused_map, solar_vec)
        pv_feature = self.pv_encoder(pv_history)
        prediction = self.head(torch.cat([sun_feature, pv_feature], dim=1))

        return {
            "prediction": prediction,
            "motion_fields": motion_fields,
            "frame_features": feats,
            "attention_map": attention_map,
        }
