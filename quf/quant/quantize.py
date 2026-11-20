"""Deterministic round-to-nearest group quantizer (reference implementation).

Layout per linear layer with weight shape ``[out, in]`` (see design.md):

* ``qweight``: int32 ``[in/8, out]``, 8 values packed along the input dimension
* ``scales``:  fp16  ``[in/group_size, out]``
* ``qzeros``:  int32 ``[in/group_size, out/8]`` (asymmetric only)

Symmetric schemes use an implicit zero point of 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from quf.quant.clip import (  # noqa: F401
    _MIN_SCALE,
    SYMMETRIC_ZERO,
    quantize_with,
    scale_zero_for_ratio,
    search_clip,
)
from quf.quant.pack import pack_int4

GROUP_SIZE = 128


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int8: returns ``(q int8 [out, in], scales fp16 [out])``."""
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in], got shape {tuple(w.shape)}")
    w32 = w.detach().to(device="cpu", dtype=torch.float32)
    scale16 = (w32.abs().amax(dim=1) / 127.0).to(torch.float16)
    scale16 = torch.where(scale16 == 0, torch.ones_like(scale16), scale16).clamp(min=2.0**-24)
    q = torch.round(w32 / scale16.to(torch.float32).unsqueeze(1)).clamp(-127, 127)
    return q.to(torch.int8), scale16


def dequantize_int8(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Return ``q * scale`` as fp16 with the shape of ``q`` (``scales`` is ``[out]``)."""
    return (q.to(torch.float32) * scales.to(torch.float32).unsqueeze(1)).to(torch.float16)


class GroupSizeError(ValueError):
    """Raised when the input dimension is not a multiple of the group size."""

    def __init__(self, dim: int, group_size: int):
        self.dim = dim
        self.group_size = group_size
        super().__init__(
            f"input dimension {dim} is not a multiple of the group size {group_size}"
        )


@dataclass
class QuantizedWeight:
    qweight: torch.Tensor  # int32 packed, [in/8, out]
    scales: torch.Tensor  # fp16, [in/group_size, out]
    qzeros: torch.Tensor | None  # int32 packed, [in/group_size, out/8], or None if symmetric
    group_size: int
    in_features: int
    out_features: int
    g_idx: torch.Tensor | None = None  # int32 [in] column -> group, act-order only


def quantize_groupwise(
    w: torch.Tensor,
    group_size: int = GROUP_SIZE,
    symmetric: bool = False,
    clip_search: bool = False,
) -> QuantizedWeight:
    """Quantize a ``[out, in]`` weight matrix to 4 bits with per-group scales.

    ``group_size`` is 32, 64, or 128 in QUF files (any divisor of ``in`` works here).
    With ``clip_search`` each group picks the clipping ratio of minimum squared error.
    """
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in], got shape {tuple(w.shape)}")
    out_features, in_features = w.shape
    if group_size <= 0 or in_features % group_size != 0:
        raise GroupSizeError(in_features, group_size)
    if out_features % 8 != 0:
        raise ValueError(f"output dimension {out_features} is not a multiple of 8")

    n_groups = in_features // group_size
    # [out, n_groups, group_size] in float32 for deterministic arithmetic.
    g = w.detach().to(device="cpu", dtype=torch.float32).reshape(
        out_features, n_groups, group_size
    )

    # Quantize with the fp16-rounded scale, i.e. the scale that is stored.
    if clip_search:
        scale16, zero = search_clip(g, symmetric)
    else:
        scale16, zero = scale_zero_for_ratio(g, symmetric, 1.0)

    q = quantize_with(g, scale16, zero)
    q = q.to(torch.uint8).reshape(out_features, in_features)

    qweight = pack_int4(q.numpy())  # [out, in/8]
    qweight = np.ascontiguousarray(qweight.T)  # [in/8, out]

    scales = scale16.T.contiguous()  # [n_groups, out]

    qzeros = None
    if not symmetric:
        zero_u8 = zero.T.contiguous().to(torch.uint8).numpy()  # [n_groups, out]
        qzeros = torch.from_numpy(pack_int4(zero_u8))  # [n_groups, out/8]

    return QuantizedWeight(
        qweight=torch.from_numpy(qweight),
        scales=scales,
        qzeros=qzeros,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
    )
