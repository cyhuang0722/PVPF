from __future__ import annotations

import torch
import torch.nn as nn


class SatelliteEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: tuple[int, int, int] = (32, 64, 128)) -> None:
        super().__init__()
        layers = []
        prev = in_channels
        for out in hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(prev, out, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out, out, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out),
                    nn.ReLU(inplace=True),
                ]
            )
            prev = out
        self.net = nn.Sequential(*layers)
        self.out_channels = hidden_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        return h, c


class TemporalModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        cells = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _, height, width = x.shape
        layer_input = x
        for cell in self.cells:
            h, c = cell.init_state(batch_size, (height, width), x.device, x.dtype)
            outputs = []
            for t in range(steps):
                h, c = cell(layer_input[:, t], (h, c))
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)
        return layer_input[:, -1]


class ForecastHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SatelliteOnlyForecaster(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: tuple[int, int, int] = (32, 64, 128),
        convlstm_hidden_dim: int = 128,
        convlstm_layers: int = 1,
        convlstm_kernel_size: int = 3,
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
        out_dim: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = SatelliteEncoder(in_channels=in_channels, hidden_channels=encoder_channels)
        self.temporal_model = TemporalModel(
            input_dim=self.encoder.out_channels,
            hidden_dim=convlstm_hidden_dim,
            num_layers=convlstm_layers,
            kernel_size=convlstm_kernel_size,
        )
        self.head = ForecastHead(
            input_dim=convlstm_hidden_dim,
            hidden_dim=head_hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

    def forward(self, satellite: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = satellite.shape
        encoded = self.encoder(satellite.reshape(bsz * steps, channels, height, width))
        encoded = encoded.reshape(bsz, steps, encoded.shape[1], encoded.shape[2], encoded.shape[3])
        context_map = self.temporal_model(encoded)
        pooled = context_map.mean(dim=(-2, -1))
        return self.head(pooled)
