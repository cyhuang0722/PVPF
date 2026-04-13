from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FrameEncoder(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stage1 = ConvBlock(3, c1, stride=2)
        self.stage2 = ConvBlock(c1, c2, stride=2)
        self.stage3 = ConvBlock(c2, c3, stride=2)
        self.stage4 = ConvBlock(c3, c4, stride=1)
        self.proj = nn.Conv2d(c4, c4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.proj(x)
