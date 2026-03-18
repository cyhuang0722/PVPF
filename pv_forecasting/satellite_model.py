from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.act(x)


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, stage_multipliers: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        stem_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        stages = []
        prev = stem_channels
        for idx, mult in enumerate(stage_multipliers):
            out = base_channels * mult
            stride = 1 if idx == 0 else 2
            stages.append(
                nn.Sequential(
                    ResidualBlock(prev, out, stride=stride),
                    ResidualBlock(out, out, stride=1),
                )
            )
            prev = out
        self.stages = nn.ModuleList(stages)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class TemporalFusion(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        context = self.temporal_pool(x).flatten(1)
        spatial = x[:, :, -1]
        gate = self.spatial_gate(spatial)
        gated_spatial = (spatial * gate).mean(dim=(-2, -1))
        return torch.cat([context, gated_spatial], dim=1)


class ForecastHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SatelliteOnlyForecaster(nn.Module):
    """
    Paper-oriented satellite branch:
    shared spatial encoder per frame + explicit 3D spatio-temporal fusion.

    The papers use the same image backbone family across modalities. Here we
    reproduce that idea with a shared residual encoder over each frame and a
    temporal fusion block that sees the full [T, H, W] volume.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        stage_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        temporal_hidden_dim: int = 128,
        head_hidden_dim: int = 256,
        dropout: float = 0.2,
        out_dim: int = 4,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = SpatialEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            stage_multipliers=stage_multipliers,
        )
        self.temporal_fusion = TemporalFusion(
            channels=self.encoder.out_channels,
            hidden_channels=temporal_hidden_dim,
            dropout=dropout,
        )
        self.head = ForecastHead(
            input_dim=temporal_hidden_dim * 2,
            hidden_dim=head_hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )
        self.debug_shapes = debug_shapes
        self._debug_shapes_printed = False

    def forward(self, satellite: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = satellite.shape
        flat = satellite.reshape(bsz * steps, channels, height, width)
        feats = self.encoder(flat)[-1]
        feats = feats.reshape(bsz, steps, feats.shape[1], feats.shape[2], feats.shape[3]).permute(0, 2, 1, 3, 4)
        fused = self.temporal_fusion(feats)
        if self.debug_shapes and not self._debug_shapes_printed:
            print(f"[debug] input satellite: {tuple(satellite.shape)}")
            print(f"[debug] encoded map: {tuple(feats.shape)}")
            print(f"[debug] fused feature: {tuple(fused.shape)}")
            self._debug_shapes_printed = True
        return self.head(fused)
