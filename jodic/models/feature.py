"""Shared speckle feature encoder with LDConv and MaSA."""

from __future__ import annotations

import torch
from torch import nn
from .ldconv import LDConv2d
from .masa import DecomposedManhattanSelfAttention


class ConvBNAct(nn.Module):
    def __init__(self, inc: int, outc: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(outc),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class JOFeatureEncoder(nn.Module):
    """Feature encoder for speckle image correspondence.

    Input can be grayscale or RGB. Output is at 1/4 input resolution by default.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 64,
        ldconv_points: int = 7,
        masa_heads: int = 4,
    ) -> None:
        super().__init__()
        mid = max(16, feature_channels // 2)
        self.stem = ConvBNAct(in_channels, mid, 5, 2, 2)
        self.ldc = LDConv2d(mid, feature_channels, num_points=ldconv_points, stride=1)
        self.down = ConvBNAct(feature_channels, feature_channels, 3, 2, 1)
        self.res1 = ResidualBlock(feature_channels)
        self.attn = DecomposedManhattanSelfAttention(feature_channels, num_heads=masa_heads)
        self.res2 = ResidualBlock(feature_channels)
        self.out = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.ldc(x)
        x = self.down(x)
        x = self.res1(x)
        x = x + self.attn(x)
        x = self.res2(x)
        return self.out(x)
