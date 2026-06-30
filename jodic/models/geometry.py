"""Geometry utilities for warping, triangulation and 3D displacement."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coords_grid(batch: int, height: int, width: int, device=None, dtype=None) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x, y], dim=0)[None].repeat(batch, 1, 1, 1)


def warp(x: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> tuple[torch.Tensor, torch.Tensor]:
    """Warp x using a dense flow field.

    Args:
        x: [B, C, H, W]
        flow: [B, 2, H, W], in pixel units, order [dx, dy]

    Returns:
        warped tensor and valid mask.
    """
    b, c, h, w = x.shape
    grid0 = coords_grid(b, h, w, x.device, x.dtype)
    grid = grid0 + flow

    if w > 1:
        x_norm = 2.0 * grid[:, 0] / (w - 1) - 1.0
    else:
        x_norm = torch.zeros_like(grid[:, 0])
    if h > 1:
        y_norm = 2.0 * grid[:, 1] / (h - 1) - 1.0
    else:
        y_norm = torch.zeros_like(grid[:, 1])

    sample_grid = torch.stack([x_norm, y_norm], dim=-1)
    warped = F.grid_sample(x, sample_grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    valid = (x_norm.abs() <= 1.0) & (y_norm.abs() <= 1.0)
    return warped, valid[:, None].to(x.dtype)


def triangulate_from_disparity(
    disparity: torch.Tensor,
    fx,
    fy,
    cx,
    cy,
    baseline,
    x_coords: torch.Tensor | None = None,
    y_coords: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Triangulate 3D coordinates from disparity.

    Args:
        disparity: [B, 1, H, W] in pixels.
        fx, fy, cx, cy, baseline: calibration parameters.
        x_coords, y_coords: optional image coordinates. If None, regular grid is used.

    Returns:
        xyz: [B, 3, H, W]
    """
    b, _, h, w = disparity.shape
    fx = _scalar_or_tensor(fx, disparity)
    fy = _scalar_or_tensor(fy, disparity)
    cx = _scalar_or_tensor(cx, disparity)
    cy = _scalar_or_tensor(cy, disparity)
    baseline = _scalar_or_tensor(baseline, disparity)

    if x_coords is None or y_coords is None:
        grid = coords_grid(b, h, w, disparity.device, disparity.dtype)
        x_coords = grid[:, 0:1]
        y_coords = grid[:, 1:2]
    else:
        if x_coords.ndim == 3:
            x_coords = x_coords[:, None]
        if y_coords.ndim == 3:
            y_coords = y_coords[:, None]

    z = fx * baseline / disparity.clamp_min(eps)
    x = (x_coords - cx) * z / fx
    y = (y_coords - cy) * z / fy
    return torch.cat([x, y, z], dim=1)


def displacement_from_disp_flow(
    disp0: torch.Tensor,
    disp1: torch.Tensor,
    flow: torch.Tensor,
    intrinsics: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 3D displacement between t0 and t1.

    d_t1 is sampled at x + flow(x). Coordinates at t1 are also x + flow(x).
    """
    b, _, h, w = disp0.shape
    disp1_warp, valid = warp(disp1, flow)

    grid0 = coords_grid(b, h, w, disp0.device, disp0.dtype)
    grid1 = grid0 + flow

    xyz0 = triangulate_from_disparity(
        disp0,
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
        intrinsics["baseline"],
        x_coords=grid0[:, 0:1],
        y_coords=grid0[:, 1:2],
    )
    xyz1 = triangulate_from_disparity(
        disp1_warp,
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
        intrinsics["baseline"],
        x_coords=grid1[:, 0:1],
        y_coords=grid1[:, 1:2],
    )
    return xyz1 - xyz0, xyz0, valid
