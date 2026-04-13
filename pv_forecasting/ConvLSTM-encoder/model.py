from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([x_t, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class ConvLSTMEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, C, H, W)
        if x_seq.ndim != 5:
            raise ValueError(f"Expected 5D input (B,T,C,H,W), got shape={tuple(x_seq.shape)}")
        b, t, _, h, w = x_seq.shape
        device = x_seq.device
        h_t = torch.zeros((b, self.hidden_channels, h, w), device=device, dtype=x_seq.dtype)
        c_t = torch.zeros((b, self.hidden_channels, h, w), device=device, dtype=x_seq.dtype)
        for step in range(t):
            h_t, c_t = self.cell(x_seq[:, step], h_t, c_t)
        return h_t


class ConvLSTMPVRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        pv_history_dim: int = 4,
        pv_hidden: int = 32,
        fc_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = ConvLSTMEncoder(
            input_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pv_head = nn.Sequential(
            nn.Linear(pv_history_dim, pv_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels + pv_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor, pv_history: torch.Tensor) -> torch.Tensor:
        feat_map = self.encoder(x_seq)
        pooled = self.pool(feat_map)
        image_feat = torch.flatten(pooled, start_dim=1)
        pv_feat = self.pv_head(pv_history)
        fused = torch.cat([image_feat, pv_feat], dim=1)
        out = self.head(fused)
        return out.squeeze(1)
