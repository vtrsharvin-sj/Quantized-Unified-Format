"""MSE-optimal clipping: per-group scale search over a fixed grid of clipping ratios.

The grid always contains 1.0 (the unclipped range) and ties keep the earliest
ratio, so the searched result is never worse than round-to-nearest.
"""

from __future__ import annotations

import torch

# 1.0, 0.95, ..., 0.5 (1.0 first so ties resolve to the unclipped result)
CLIP_RATIOS: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)

SYMMETRIC_ZERO = 8
_MIN_SCALE = 2.0**-24  # smallest positive fp16 subnormal


def scale_zero_for_ratio(
    g: torch.Tensor, symmetric: bool, ratio: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(scale16, zero)`` for groups ``g`` of shape ``[..., group_size]``.

    ``scale16`` is the fp16 scale that is stored (shape ``[...]``); ``zero`` is float32
    in 0..15. ``ratio`` shrinks the group range before the scale is computed.
    """
    if symmetric:
        scale = g.abs().amax(dim=-1) * ratio / 7.0
        wmin = None
    else:
        # The range always includes 0 so a constant group stays representable.
        wmin = g.amin(dim=-1).clamp(max=0.0) * ratio
        wmax = g.amax(dim=-1).clamp(min=0.0) * ratio
        scale = (wmax - wmin) / 15.0
    scale16 = scale.to(torch.float16)
    scale16 = torch.where(scale16 == 0, torch.ones_like(scale16), scale16).clamp(min=_MIN_SCALE)
    scale32 = scale16.to(torch.float32)
    if symmetric:
        zero = torch.full_like(scale32, float(SYMMETRIC_ZERO))
    else:
        zero = torch.round(-wmin / scale32).clamp(0, 15)
    return scale16, zero


def quantize_with(
    g: torch.Tensor, scale16: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    """Quantize groups ``g`` to integer values 0..15 (float32) with the given scale and zero."""
    q = torch.round(g / scale16.to(torch.float32).unsqueeze(-1)) + zero.unsqueeze(-1)
    return q.clamp(0, 15)


def search_clip(
    g: torch.Tensor, symmetric: bool, ratios: tuple[float, ...] = CLIP_RATIOS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick, per group, the ratio with the smallest squared weight error.

    ``g`` has shape ``[..., group_size]`` (float32). Returns ``(scale16, zero)``.
    """
    best_scale, best_zero = scale_zero_for_ratio(g, symmetric, ratios[0])
    best_err = _group_error(g, best_scale, best_zero)
    for r in ratios[1:]:
        scale, zero = scale_zero_for_ratio(g, symmetric, r)
        err = _group_error(g, scale, zero)
        better = err < best_err  # strict: ties keep the earlier (less clipped) ratio
        best_scale = torch.where(better, scale, best_scale)
        best_zero = torch.where(better, zero, best_zero)
        best_err = torch.where(better, err, best_err)
    return best_scale, best_zero


def _group_error(g: torch.Tensor, scale16: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    q = quantize_with(g, scale16, zero)
    # Round to fp16 exactly as reference_dequantize does, so the search ranks the weights that
    # are actually reconstructed from the stored (q, scale, zero), not an idealised fp32 version.
    deq = ((q - zero.unsqueeze(-1)) * scale16.to(torch.float32).unsqueeze(-1)).to(torch.float16)
    return ((deq.to(torch.float32) - g) ** 2).sum(dim=-1)
