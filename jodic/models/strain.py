"""Numerical strain-field reconstruction from a surface displacement field."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _central_diff(v: torch.Tensor, dim: int, spacing: float = 1.0) -> torch.Tensor:
    """Central difference with replication padding."""
    if dim == -1:
        padded = F.pad(v, (1, 1, 0, 0), mode="replicate")
        return (padded[..., 2:] - padded[..., :-2]) / (2.0 * spacing)
    if dim == -2:
        padded = F.pad(v, (0, 0, 1, 1), mode="replicate")
        return (padded[..., 2:, :] - padded[..., :-2, :]) / (2.0 * spacing)
    raise ValueError("dim must be -1 or -2")


def strain_from_displacement(
    disp3d: torch.Tensor,
    spacing_x: float = 1.0,
    spacing_y: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute small-strain quantities from [U,V,W] over image/surface grid.

    Args:
        disp3d: [B, 3, H, W] displacement field.

    Returns:
        Dictionary of strain components.
    """
    u = disp3d[:, 0:1]
    v = disp3d[:, 1:2]
    w = disp3d[:, 2:3]

    du_dx = _central_diff(u, -1, spacing_x)
    du_dy = _central_diff(u, -2, spacing_y)
    dv_dx = _central_diff(v, -1, spacing_x)
    dv_dy = _central_diff(v, -2, spacing_y)
    dw_dx = _central_diff(w, -1, spacing_x)
    dw_dy = _central_diff(w, -2, spacing_y)

    return {
        "eps_xx": du_dx,
        "eps_yy": dv_dy,
        "eps_xy": 0.5 * (du_dy + dv_dx),
        "eps_zx": 0.5 * dw_dx,
        "eps_zy": 0.5 * dw_dy,
        "eps_zz": torch.zeros_like(w),
    }
