"""Complete JO-DIC model."""

from __future__ import annotations

import torch
from torch import nn
from .feature import JOFeatureEncoder
from .stereo_like import CGIStereoBranch
from .raft_like import RAFTFlowBranch
from .geometry import displacement_from_disp_flow
from .strain import strain_from_displacement


class JODICModel(nn.Module):
    """Binocular full-field 3D displacement and strain measurement network.

    Inputs:
        left0, right0: stereo images at reference time t0.
        left1, right1: stereo images at current time t1.
        intrinsics: optional calibration dict with fx, fy, cx, cy, baseline.

    Outputs:
        disp0, disp1: full-resolution disparities.
        flow: t0 -> t1 optical flow on left images.
        disp3d: optional [U,V,W] if intrinsics are supplied.
        strain: optional strain dictionary if intrinsics are supplied.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 64,
        max_disp: int = 64,
        ldconv_points: int = 7,
        masa_heads: int = 4,
        flow_iters: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = JOFeatureEncoder(
            in_channels=in_channels,
            feature_channels=feature_channels,
            ldconv_points=ldconv_points,
            masa_heads=masa_heads,
        )
        self.stereo = CGIStereoBranch(feature_channels=feature_channels, max_disp=max_disp, scale=4)
        self.flow = RAFTFlowBranch(feature_channels=feature_channels, scale=4, iters=flow_iters)

    def forward(
        self,
        left0: torch.Tensor,
        right0: torch.Tensor,
        left1: torch.Tensor,
        right1: torch.Tensor,
        intrinsics: dict | None = None,
        flow_iters: int | None = None,
    ) -> dict:
        h, w = left0.shape[-2:]

        f_l0 = self.encoder(left0)
        f_r0 = self.encoder(right0)
        f_l1 = self.encoder(left1)
        f_r1 = self.encoder(right1)

        st0 = self.stereo(f_l0, f_r0, output_size=(h, w))
        st1 = self.stereo(f_l1, f_r1, output_size=(h, w))
        fl = self.flow(f_l0, f_l1, output_size=(h, w), iters=flow_iters)

        out = {
            "disp0": st0["disp"],
            "disp1": st1["disp"],
            "disp0_low": st0["disp_low"],
            "disp1_low": st1["disp_low"],
            "cost0": st0["cost_volume"],
            "cost1": st1["cost_volume"],
            "flow": fl["flow"],
            "flow_low": fl["flow_low"],
        }

        if intrinsics is not None:
            disp3d, xyz0, valid = displacement_from_disp_flow(out["disp0"], out["disp1"], out["flow"], intrinsics)
            out["disp3d"] = disp3d
            out["xyz0"] = xyz0
            out["geometry_valid"] = valid
            out["strain"] = strain_from_displacement(disp3d)

        return out
