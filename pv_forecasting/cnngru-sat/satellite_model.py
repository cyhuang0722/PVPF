from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TemporalEncoder(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[:, :, -1]


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_zr = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=True)
        self.conv_h = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        zr = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = torch.chunk(zr, 2, dim=1)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_tilde


class FutureStatePredictor(nn.Module):
    def __init__(self, channels: int, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = ConvGRUCell(channels)
        self.refine = nn.Sequential(*[ResidualBlock(channels, channels, stride=1) for _ in range(num_layers)])

    def forward(self, state: torch.Tensor, steps: int) -> torch.Tensor:
        outputs = []
        hidden = state
        current = state
        for _ in range(steps):
            hidden = self.gru(current, hidden)
            current = self.refine(hidden)
            outputs.append(current)
        return torch.stack(outputs, dim=1)


class IrradianceDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, _, _ = states.shape
        flat = states.reshape(bsz * steps, channels, states.shape[-2], states.shape[-1])
        pooled = self.pool(flat).flatten(1)
        out = self.head(pooled)
        return out.reshape(bsz, steps)


class CloudIndexDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        mid_channels = max(32, hidden_channels // 2)
        self.net = nn.Sequential(
            ResidualBlock(in_channels, hidden_channels, stride=1),
            ResidualBlock(hidden_channels, mid_channels, stride=1),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, states: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        bsz, steps, channels, _, _ = states.shape
        flat = states.reshape(bsz * steps, channels, states.shape[-2], states.shape[-1])
        up = F.interpolate(flat, size=output_size, mode="bilinear", align_corners=False)
        ci = torch.sigmoid(self.net(up))
        return ci.reshape(bsz, steps, output_size[0], output_size[1])


class SatelliteOnlyForecaster(nn.Module):
    """
    ECLIPSE-style satellite backbone:
    spatial encoder -> 3D temporal encoder -> recurrent future states.

    The model produces both future power predictions and future cloud-index
    maps, matching the multi-task setup used in the satellite papers much
    better than a single MLP forecasting head.
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
        self.temporal_encoder = TemporalEncoder(
            channels=self.encoder.out_channels,
            hidden_channels=temporal_hidden_dim,
            dropout=dropout,
        )
        self.future_predictor = FutureStatePredictor(channels=temporal_hidden_dim)
        self.power_decoder = IrradianceDecoder(
            in_channels=temporal_hidden_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        self.cloud_index_decoder = CloudIndexDecoder(
            in_channels=temporal_hidden_dim,
            hidden_channels=max(64, temporal_hidden_dim),
        )
        self.out_dim = out_dim
        self.debug_shapes = debug_shapes
        self._debug_shapes_printed = False

    def forward(self, satellite: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, steps, channels, height, width = satellite.shape
        flat = satellite.reshape(bsz * steps, channels, height, width)
        feats = self.encoder(flat)[-1]
        feats = feats.reshape(bsz, steps, feats.shape[1], feats.shape[2], feats.shape[3]).permute(0, 2, 1, 3, 4)
        state = self.temporal_encoder(feats)
        future_states = self.future_predictor(state, steps=self.out_dim)
        power = self.power_decoder(future_states)
        cloud_index = self.cloud_index_decoder(future_states, output_size=(height, width))
        if self.debug_shapes and not self._debug_shapes_printed:
            print(f"[debug] input satellite: {tuple(satellite.shape)}")
            print(f"[debug] encoded map: {tuple(feats.shape)}")
            print(f"[debug] temporal state: {tuple(state.shape)}")
            print(f"[debug] future states: {tuple(future_states.shape)}")
            print(f"[debug] cloud index: {tuple(cloud_index.shape)}")
            self._debug_shapes_printed = True
        return {"power": power, "cloud_index": cloud_index}
