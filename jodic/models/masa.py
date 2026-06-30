"""Manhattan Self-Attention (MaSA).

The module introduces an explicit spatial prior through a Manhattan-distance
decay matrix. Two variants are provided:

1. ManhattanSelfAttention: full 2D attention with D_nm = gamma^ManhattanDistance.
2. DecomposedManhattanSelfAttention: axial decomposition along width and height,
   preserving the Manhattan prior while reducing memory/computation.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _head_gammas(num_heads: int, gamma_min: float, gamma_max: float) -> torch.Tensor:
    vals = torch.linspace(float(gamma_min), float(gamma_max), int(num_heads))
    return vals.clamp(1e-4, 0.9999)


class ManhattanSelfAttention(nn.Module):
    """Full Manhattan Self-Attention for small feature maps."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        gamma_min: float = 0.75,
        gamma_max: float = 0.996,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.register_buffer("gammas", _head_gammas(num_heads, gamma_min, gamma_max))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, n)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        decay = self._decay(h, w, x.device, x.dtype)
        attn = attn * decay[None]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        out = self.proj(out)
        return self.proj_drop(out)


class DecomposedManhattanSelfAttention(nn.Module):
    """Axially decomposed MaSA.

    Horizontal attention is computed along image rows, followed by vertical
    attention along image columns. Each axis uses a 1D Manhattan-distance decay.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        gamma_min: float = 0.75,
        gamma_max: float = 0.996,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_lce: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_lce = use_lce

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lce = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) if use_lce else None
        self.register_buffer("gammas", _head_gammas(num_heads, gamma_min, gamma_max))

    def _decay_1d(self, length: int, device, dtype) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype)
        gammas = self.gammas.to(device=device, dtype=dtype)
        return torch.pow(gammas[:, None, None], dist[None])

    @staticmethod
    def _apply_decay_attention(scores: torch.Tensor, values: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        attn = F.softmax(scores, dim=-1)
        attn = attn * decay
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.matmul(attn, values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h, w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q_w = q.permute(0, 1, 3, 4, 2).reshape(b * self.num_heads * h, w, self.head_dim)
        k_w = k.permute(0, 1, 3, 4, 2).reshape(b * self.num_heads * h, w, self.head_dim)
        v_w = v.permute(0, 1, 3, 4, 2).reshape(b * self.num_heads * h, w, self.head_dim)

        scores_w = torch.matmul(q_w, k_w.transpose(-2, -1)) * self.scale
        decay_w = self._decay_1d(w, x.device, x.dtype)
        decay_w = decay_w[None, :, None].expand(b, self.num_heads, h, w, w).reshape(
            b * self.num_heads * h, w, w
        )
        out_w = self._apply_decay_attention(scores_w, v_w, decay_w)
        out_w = out_w.reshape(b, self.num_heads, h, w, self.head_dim)

        q_h = q.permute(0, 1, 4, 3, 2).reshape(b * self.num_heads * w, h, self.head_dim)
        k_h = k.permute(0, 1, 4, 3, 2).reshape(b * self.num_heads * w, h, self.head_dim)
        v_h = out_w.permute(0, 1, 3, 2, 4).reshape(b * self.num_heads * w, h, self.head_dim)

        scores_h = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
        decay_h = self._decay_1d(h, x.device, x.dtype)
        decay_h = decay_h[None, :, None].expand(b, self.num_heads, w, h, h).reshape(
            b * self.num_heads * w, h, h
        )
        out = self._apply_decay_attention(scores_h, v_h, decay_h)
        out = out.reshape(b, self.num_heads, w, h, self.head_dim)
        out = out.permute(0, 1, 4, 3, 2).reshape(b, c, h, w)

        if self.lce is not None:
            out = out + self.lce(x)
        out = self.proj(out)
        return self.proj_drop(out)
