"""Linear deformable convolution used for adaptive speckle feature enhancement.

This module follows the measurement motivation of LDConv:
learn N irregular sampling points around each output location and aggregate
the sampled features with a lightweight point-wise convolution.

Compared with a full deformable convolution implementation, this version is
kept compact and dependency-free by using torch.nn.functional.grid_sample.
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def _make_initial_offsets(num_points: int) -> torch.Tensor:
    """Return roughly rectangular N-point offsets centered at zero.

    Shape: [N, 2], with columns [dy, dx] in pixel coordinates.
    """
    base = max(1, round(math.sqrt(num_points)))
    rows = math.ceil(num_points / base)
    coords = []
    for i in range(rows):
        for j in range(base):
            if len(coords) < num_points:
                coords.append((float(i), float(j)))
    p = torch.tensor(coords, dtype=torch.float32)
    p[:, 0] -= p[:, 0].mean()
    p[:, 1] -= p[:, 1].mean()
    return p


class LDConv2d(nn.Module):
    """Linear deformable convolution.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        num_points: number of adaptive sampling points. The paper response
            uses N=7 as the recommended setting.
        stride: output stride for the offset field.
        bias: whether to use bias in the aggregation convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_points: int = 7,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.stride = int(stride)

        self.offset = nn.Conv2d(
            in_channels,
            2 * self.num_points,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

        self.register_buffer("base_offsets", _make_initial_offsets(self.num_points))
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * self.num_points, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.last_offset: torch.Tensor | None = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        offset = self.offset(x)
        self.last_offset = offset
        _, _, ho, wo = offset.shape
        n = self.num_points

        yy = torch.arange(ho, device=x.device, dtype=x.dtype) * self.stride
        xx = torch.arange(wo, device=x.device, dtype=x.dtype) * self.stride
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")

        off = offset.view(b, 2, n, ho, wo)
        samples = []

        for k in range(n):
            dy0, dx0 = self.base_offsets[k].to(device=x.device, dtype=x.dtype)
            y = yy[None] + dy0 + off[:, 0, k]
            xcoord = xx[None] + dx0 + off[:, 1, k]

            if h > 1:
                y_norm = 2.0 * y / (h - 1) - 1.0
            else:
                y_norm = torch.zeros_like(y)
            if w > 1:
                x_norm = 2.0 * xcoord / (w - 1) - 1.0
            else:
                x_norm = torch.zeros_like(xcoord)

            grid = torch.stack([x_norm, y_norm], dim=-1)
            sampled = F.grid_sample(
                x,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            samples.append(sampled)

        stacked = torch.cat(samples, dim=1)
        return self.proj(stacked)
