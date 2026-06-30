"""Synthetic speckle sequence dataset for pipeline verification.

This generator creates four grayscale images:
left0, right0, left1, right1,
with synthetic disparity and optical flow targets.

It is intended for code-level reproduction and debugging rather than replacing
the real calibrated experimental dataset.
"""

from __future__ import annotations

import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from jodic.models.geometry import warp


def _gaussian_kernel(size: int = 7, sigma: float = 1.4, device=None) -> torch.Tensor:
    ax = torch.arange(size, device=device) - size // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    k = k / k.sum()
    return k.view(1, 1, size, size)


def make_speckle(height: int, width: int, density: float = 0.035, device=None) -> torch.Tensor:
    img = torch.rand(1, 1, height, width, device=device)
    dots = (torch.rand(1, 1, height, width, device=device) < density).float()
    img = 0.25 * img + 0.75 * dots
    kernel = _gaussian_kernel(7, 1.2, device=device)
    img = F.conv2d(img, kernel, padding=3)
    img = (img - img.amin(dim=(-2, -1), keepdim=True)) / (img.amax(dim=(-2, -1), keepdim=True) - img.amin(dim=(-2, -1), keepdim=True) + 1e-6)
    return img


class SyntheticSpeckleDataset(Dataset):
    def __init__(
        self,
        length: int = 512,
        height: int = 128,
        width: int = 160,
        max_disp: int = 48,
        seed: int = 1234,
    ) -> None:
        self.length = int(length)
        self.height = int(height)
        self.width = int(width)
        self.max_disp = int(max_disp)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(self.seed + idx)
        h, w = self.height, self.width

        base = make_speckle(h, w, density=0.035)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        xx = xx.float()[None, None]
        yy = yy.float()[None, None]

        d_base = 0.25 * self.max_disp + 0.08 * self.max_disp * torch.sin(xx / max(w, 1) * 2 * math.pi)
        d0 = d_base + torch.rand(1, 1, h, w, generator=g) * 2.0
        d1 = d0 + 0.5 * torch.sin((xx + yy) / 31.0)

        tx = 2.0 + 1.5 * torch.sin(yy / max(h, 1) * 2 * math.pi)
        ty = 1.0 + 0.8 * torch.cos(xx / max(w, 1) * 2 * math.pi)
        flow = torch.cat([tx, ty], dim=1)

        right0, _ = warp(base, torch.cat([-d0, torch.zeros_like(d0)], dim=1))
        left1, _ = warp(base, -flow)
        right1, _ = warp(left1, torch.cat([-d1, torch.zeros_like(d1)], dim=1))

        noise = 0.015
        left0 = (base + noise * torch.randn(base.shape, generator=g)).clamp(0, 1)
        right0 = (right0 + noise * torch.randn(right0.shape, generator=g)).clamp(0, 1)
        left1 = (left1 + noise * torch.randn(left1.shape, generator=g)).clamp(0, 1)
        right1 = (right1 + noise * torch.randn(right1.shape, generator=g)).clamp(0, 1)

        return {
            "left0": left0[0],
            "right0": right0[0],
            "left1": left1[0],
            "right1": right1[0],
            "disp0": d0[0],
            "disp1": d1[0],
            "flow": flow[0],
            "disp_mask0": torch.ones(1, h, w),
            "disp_mask1": torch.ones(1, h, w),
            "flow_mask": torch.ones(1, h, w),
        }
