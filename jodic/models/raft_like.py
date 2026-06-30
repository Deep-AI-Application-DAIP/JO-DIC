"""optical-flow branch."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from .geometry import warp


class FlowUpdateBlock(nn.Module):
    def __init__(self, feature_channels: int, hidden_channels: int = 96) -> None:
        super().__init__()
        inc = feature_channels * 3 + 2
        self.net = nn.Sequential(
            nn.Conv2d(inc, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f0: torch.Tensor, f1_warp: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f0, f1_warp, f0 - f1_warp, flow], dim=1)
        return self.net(x)


class RAFTFlowBranch(nn.Module):
    """Compact RAFT-style recurrent refinement.

    It iteratively warps t1 features toward t0 features and predicts residual
    flow updates at feature scale.
    """

    def __init__(self, feature_channels: int = 64, scale: int = 4, iters: int = 4) -> None:
        super().__init__()
        self.scale = int(scale)
        self.iters = int(iters)
        self.update = FlowUpdateBlock(feature_channels)

    def forward(
        self,
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        output_size: tuple[int, int],
        iters: int | None = None,
    ) -> dict[str, torch.Tensor]:
        b, c, h, w = feat0.shape
        flow = feat0.new_zeros(b, 2, h, w)
        preds = []
        steps = int(self.iters if iters is None else iters)
        for _ in range(steps):
            feat1_warp, _ = warp(feat1, flow)
            delta = self.update(feat0, feat1_warp, flow)
            flow = flow + delta
            preds.append(flow)

        flow_full = F.interpolate(flow, size=output_size, mode="bilinear", align_corners=True)
        flow_full[:, 0] *= output_size[1] / max(w, 1)
        flow_full[:, 1] *= output_size[0] / max(h, 1)
        return {"flow": flow_full, "flow_low": flow, "flow_predictions_low": preds}
