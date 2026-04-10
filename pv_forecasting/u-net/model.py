from __future__ import annotations

import torch
import torch.nn as nn


class Conv3x3Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BottleNeckBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class Up2x2Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPVMapper(nn.Module):
    """
    SkyGPT-style modified U-Net for regression.
    Input is 8 stacked RGB frames: 24 channels by default.
    """

    def __init__(
        self,
        in_channels: int = 24,
        img_size: tuple[int, int] = (64, 64),
        num_filters: int = 12,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.img_size = img_size

        self.input_proj = nn.Conv2d(in_channels, num_filters, kernel_size=1)

        self.enc1 = Conv3x3Block(num_filters, num_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = Conv3x3Block(num_filters, 2 * num_filters)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = Conv3x3Block(2 * num_filters, 4 * num_filters)
        self.bottleneck1 = BottleNeckBlock(4 * num_filters)
        self.bottleneck2 = BottleNeckBlock(4 * num_filters)

        self.up1 = Up2x2Conv3x3(4 * num_filters, 2 * num_filters)
        self.dec1 = Conv3x3Block(4 * num_filters, 2 * num_filters)
        self.drop1 = nn.Dropout(dropout)

        self.up2 = Up2x2Conv3x3(2 * num_filters, num_filters)
        self.dec2 = Conv3x3Block(2 * num_filters, num_filters)
        self.drop2 = nn.Dropout(dropout)

        self.reg_map = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(img_size[0] * img_size[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_proj(x)
        x2 = self.enc1(x1)
        x3 = self.pool1(x2)
        x4 = self.enc2(x3)
        x5 = self.pool2(x4)
        x6 = self.enc3(x5)
        x7 = self.bottleneck1(x6)
        x8 = self.bottleneck2(x7)

        x9 = self.up1(x8)
        x10 = torch.cat([x9, x4], dim=1)
        x11 = self.drop1(self.dec1(x10))

        x12 = self.up2(x11)
        x13 = torch.cat([x12, x2], dim=1)
        x14 = self.drop2(self.dec2(x13))

        y = self.reg_map(x14)
        y = torch.flatten(y, start_dim=1)
        y = self.head(y)
        return y.squeeze(1)
