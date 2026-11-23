"""Quantized_Embedding: per-row group-wise int4 / int8 table (format version 3).

Layout (row-major, one row per vocabulary entry, so a row gather and a GEMV over all rows,
``logits = x @ W.T``, read the same bytes):

* int4: ``packed`` int8 ``[rows, cols/2]``; column ``2j`` is the low nibble and ``2j+1`` the
  high nibble of byte ``j``. Nibbles are unsigned 0..15 with zero point 8.
* int8: ``packed`` int8 ``[rows, cols]``, signed -127..127, zero point 0.
* ``scales`` fp16 ``[rows, cols/group_size]``; column ``c`` uses ``scales[r, c // group_size]``.

``w[r, c] = (q[r, c] - zero) * scales[r, c // group_size]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from quf.format.constants import EMBED_BITS, EMBED_GROUP_SIZES

INT4_ZERO = 8


@dataclass
class QuantizedEmbedding:
    packed: np.ndarray  # int8, [rows, cols/2] (bits 4) or [rows, cols] (bits 8)
    scales: np.ndarray  # fp16, [rows, cols/group_size]
    bits: int
    group_size: int

    @property
    def rows(self) -> int:
        return int(self.packed.shape[0])

    @property
    def cols(self) -> int:
        return int(self.packed.shape[1]) * (2 if self.bits == 4 else 1)


def _check(bits: int, group_size: int, cols: int) -> None:
    if bits not in EMBED_BITS:
        raise ValueError(f"unsupported embedding bits {bits}: use 4 or 8")
    if group_size not in EMBED_GROUP_SIZES:
        raise ValueError(f"unsupported embedding group size {group_size}: use 32 or 64")
    if cols % group_size != 0:
        raise ValueError(f"hidden size {cols} is not a multiple of group size {group_size}")


def quantize_embedding_rtn(weight, bits: int, group_size: int) -> QuantizedEmbedding:
    """Round-to-nearest, symmetric, per-row per-group quantization of ``[rows, cols]``."""
    w = torch.as_tensor(weight).to(torch.float32)
    if w.ndim != 2:
        raise ValueError(f"embedding weight must be 2-D, got shape {tuple(w.shape)}")
    rows, cols = w.shape
    _check(bits, group_size, cols)
    qmax = 7 if bits == 4 else 127
    g = w.reshape(rows, cols // group_size, group_size)
    scale = (g.abs().amax(dim=-1) / qmax).to(torch.float16)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(g / scale.to(torch.float32).unsqueeze(-1)).clamp(-qmax, qmax)
    q = q.reshape(rows, cols).to(torch.int16)
    if bits == 4:
        u = (q + INT4_ZERO).to(torch.uint8)
        packed = (u[:, 0::2] | (u[:, 1::2] << 4)).view(torch.int8)
    else:
        packed = q.to(torch.int8)
    return QuantizedEmbedding(
        packed=np.ascontiguousarray(packed.numpy()),
        scales=np.ascontiguousarray(scale.numpy()),
        bits=bits,
        group_size=group_size,
    )


_CHUNK_ELEMENTS = 1 << 24  # float32 elements per search chunk (64 MB)


def _pack(q: torch.Tensor, bits: int) -> np.ndarray:
    """Pack signed integer values ``[rows, cols]`` into the stored layout."""
    q = q.to(torch.int16)
    if bits == 4:
        u = (q + INT4_ZERO).to(torch.uint8)
        packed = (u[:, 0::2] | (u[:, 1::2] << 4)).view(torch.int8)
    else:
        packed = q.to(torch.int8)
    return np.ascontiguousarray(packed.numpy())


def quantize_embedding_clip(
    weight, bits: int, group_size: int, ratios: tuple[float, ...] | None = None
) -> QuantizedEmbedding:
    """Symmetric per-row per-group quantization with an MSE-optimal clipping ratio per group.

    Uses the ratio grid of :mod:`quf.quant.clip` (1.0 first, strict improvement needed to
    switch), so the squared error of every group is never larger than that of
    :func:`quantize_embedding_rtn`. The format stores a fixed zero point (8 for int4, 0 for
    int8), so there is no asymmetric variant.
    """
    from quf.quant.clip import CLIP_RATIOS  # noqa: PLC0415

    ratios = CLIP_RATIOS if ratios is None else ratios
    w = torch.as_tensor(weight).to(torch.float32)
    if w.ndim != 2:
        raise ValueError(f"embedding weight must be 2-D, got shape {tuple(w.shape)}")
    rows, cols = w.shape
    _check(bits, group_size, cols)
    qmax = 7 if bits == 4 else 127
    step = max(1, _CHUNK_ELEMENTS // cols)
    q_parts, s_parts = [], []
    for r0 in range(0, rows, step):
        g = w[r0 : r0 + step].reshape(-1, cols // group_size, group_size)
        amax = g.abs().amax(dim=-1)
        best_scale = best_err = None
        for ratio in ratios:
            scale = (amax * ratio / qmax).to(torch.float16)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            s32 = scale.to(torch.float32).unsqueeze(-1)
            err = ((torch.round(g / s32).clamp(-qmax, qmax) * s32 - g) ** 2).sum(dim=-1)
            if best_scale is None:
                best_scale, best_err = scale, err
            else:
                better = err < best_err
                best_scale = torch.where(better, scale, best_scale)
                best_err = torch.where(better, err, best_err)
        s32 = best_scale.to(torch.float32).unsqueeze(-1)
        q = torch.round(g / s32).clamp(-qmax, qmax).reshape(g.shape[0], cols)
        q_parts.append(q)
        s_parts.append(best_scale)
    q_all = torch.cat(q_parts) if q_parts else torch.zeros(0, cols)
    s_all = torch.cat(s_parts) if s_parts else torch.zeros(0, cols // group_size, dtype=torch.float16)
    return QuantizedEmbedding(
        packed=_pack(q_all, bits),
        scales=np.ascontiguousarray(s_all.numpy()),
        bits=bits,
        group_size=group_size,
    )


def quantize_embedding(
    weight, bits: int, group_size: int, method: str = "clip"
) -> QuantizedEmbedding:
    """``method`` is ``"rtn"`` (plain round-to-nearest) or ``"clip"`` (MSE-optimal clipping)."""
    if method == "rtn":
        return quantize_embedding_rtn(weight, bits, group_size)
    if method == "clip":
        return quantize_embedding_clip(weight, bits, group_size)
    raise ValueError(f"unknown embedding method {method!r}: use rtn or clip")


def dequantize_embedding_rows(
    packed, scales, bits: int, group_size: int, ids=None
) -> torch.Tensor:
    """Reference row dequantization: fp16 ``[len(ids), cols]`` (all rows when ``ids`` is None).

    Only the requested rows are unpacked. Source of truth for every Backend's row gather.
    """
    if ids is not None and not torch.is_tensor(packed):
        # gather in numpy first so a read-only mmap view is never copied whole
        sel = np.asarray(ids, dtype=np.int64)
        packed, scales = np.asarray(packed)[sel], np.asarray(scales)[sel]
        ids = None
    p = packed if torch.is_tensor(packed) else torch.from_numpy(np.ascontiguousarray(packed))
    s = scales if torch.is_tensor(scales) else torch.from_numpy(np.ascontiguousarray(scales))
    if ids is not None:
        idx = torch.as_tensor(ids).to(device=p.device, dtype=torch.long)
        p, s = p[idx], s[idx]
    if bits == 4:
        u = p.view(torch.uint8)
        q = torch.stack((u & 0xF, u >> 4), dim=-1).reshape(*u.shape[:-1], u.shape[-1] * 2)
        q = q.to(torch.float32) - INT4_ZERO
    elif bits == 8:
        q = p.to(torch.float32)
    else:
        raise ValueError(f"unsupported embedding bits {bits}: use 4 or 8")
    cols = q.shape[-1]
    _check(bits, group_size, cols)
    if s.shape[-1] * group_size != cols:
        raise ValueError(
            f"scales have {s.shape[-1]} groups per row, expected {cols // group_size}"
        )
    w = q * s.to(torch.float32).repeat_interleave(group_size, dim=-1)
    return w.to(torch.float16)


def dequantize_embedding(qe: QuantizedEmbedding) -> torch.Tensor:
    """Full fp16 table ``[rows, cols]``."""
    return dequantize_embedding_rows(qe.packed, qe.scales, qe.bits, qe.group_size)
