"""disparity estimation branch."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def build_1d_correlation_volume(left: torch.Tensor, right: torch.Tensor, max_disp: int) -> torch.Tensor:
    """Build a left-right correlation volume for positive disparities.

    Args:
        left, right: [B, C, H, W] normalized feature maps.
        max_disp: number of disparity hypotheses at feature scale.

    Returns:
        cost volume [B, D, H, W], where larger values indicate better matching.
    """
    b, c, h, w = left.shape
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    vols = []
    for d in range(max_disp):
        if d == 0:
            corr = (left * right).mean(dim=1, keepdim=True)
        else:
            corr = left.new_zeros(b, 1, h, w)
            corr[:, :, :, d:] = (left[:, :, :, d:] * right[:, :, :, :-d]).mean(dim=1, keepdim=True)
        vols.append(corr)
    return torch.cat(vols, dim=1)


class CGIStereoBranch(nn.Module):
    """Lightweight stereo branch inspired by cost-volume regularization.

    The branch consumes shared features from the JOFeatureEncoder and predicts
    a full-resolution disparity map through soft argmin.
    """

    def __init__(self, feature_channels: int = 64, max_disp: int = 64, scale: int = 4) -> None:
        super().__init__()
        assert max_disp >= scale
        self.max_disp = int(max_disp)
        self.scale = int(scale)
        self.max_disp_low = max(2, self.max_disp // self.scale)

        self.context = nn.Sequential(
            nn.Conv2d(feature_channels, self.max_disp_low, kernel_size=1),
            nn.Sigmoid(),
        )
        self.regularizer = nn.Sequential(
            nn.Conv2d(self.max_disp_low, self.max_disp_low, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.max_disp_low),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.max_disp_low, self.max_disp_low, 3, padding=1, bias=True),
        )
        disp_values = torch.arange(self.max_disp_low, dtype=torch.float32).view(1, self.max_disp_low, 1, 1)
        self.register_buffer("disp_values", disp_values)

    def forward(self, feat_l: torch.Tensor, feat_r: torch.Tensor, output_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        volume = build_1d_correlation_volume(feat_l, feat_r, self.max_disp_low)
        gate = self.context(feat_l)
        logits = self.regularizer(volume * (1.0 + gate))
        prob = F.softmax(logits, dim=1)
        disp_low = (prob * self.disp_values.to(prob)).sum(dim=1, keepdim=True) * self.scale
        disp = F.interpolate(disp_low, size=output_size, mode="bilinear", align_corners=True)
        return {"disp": disp, "disp_low": disp_low, "cost_volume": volume}
