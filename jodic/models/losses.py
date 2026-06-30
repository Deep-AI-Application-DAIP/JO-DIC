"""Loss functions for JO-DIC."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from .geometry import warp


def charbonnier(x: torch.Tensor, eps: float = 1e-3, alpha: float = 0.5) -> torch.Tensor:
    return (x * x + eps * eps).pow(alpha)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def endpoint_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    epe = torch.linalg.vector_norm(pred - target, dim=1, keepdim=True)
    return masked_mean(epe, mask)


def geometric_consistency_loss(
    disp0: torch.Tensor,
    disp1: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_c = rho(W(d_t1, x + flow) - d_t0)."""
    disp1_warp, valid = warp(disp1, flow)
    if mask is None:
        mask = valid
    else:
        mask = mask * valid
    return masked_mean(charbonnier(disp1_warp - disp0), mask)


class JODICLoss(nn.Module):
    def __init__(self, lambda_disp: float = 1.0, lambda_flow: float = 1.0, lambda_consistency: float = 0.2):
        super().__init__()
        self.lambda_disp = float(lambda_disp)
        self.lambda_flow = float(lambda_flow)
        self.lambda_consistency = float(lambda_consistency)

    def forward(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = {}

        if "disp0" in targets:
            mask0 = targets.get("disp_mask0", None)
            losses["disp0"] = masked_mean(charbonnier(outputs["disp0"] - targets["disp0"]), mask0)
        else:
            losses["disp0"] = outputs["disp0"].new_tensor(0.0)

        if "disp1" in targets:
            mask1 = targets.get("disp_mask1", None)
            losses["disp1"] = masked_mean(charbonnier(outputs["disp1"] - targets["disp1"]), mask1)
        else:
            losses["disp1"] = outputs["disp1"].new_tensor(0.0)

        if "flow" in targets:
            losses["flow"] = endpoint_error(outputs["flow"], targets["flow"], targets.get("flow_mask", None))
        else:
            losses["flow"] = outputs["flow"].new_tensor(0.0)

        losses["consistency"] = geometric_consistency_loss(
            outputs["disp0"], outputs["disp1"], outputs["flow"], targets.get("consistency_mask", None)
        )

        total = (
            self.lambda_disp * (losses["disp0"] + losses["disp1"])
            + self.lambda_flow * losses["flow"]
            + self.lambda_consistency * losses["consistency"]
        )
        losses["total"] = total
        return total, losses
