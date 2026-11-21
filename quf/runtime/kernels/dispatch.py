"""Dispatcher for the Triton path: int4 -> Int4Layout, everything else -> cached fp16 (task 28.3).

Requirements 33.6, 33.11
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import QuantizedWeight
from quf.runtime.backends.base import Int8Weight, dequantize_int8_weight
from quf.runtime.kernels.triton_kernels import (
    MAX_GEMV_M,
    SUPPORTED_GROUP_SIZES,
    Int4Layout,
    int4_gemv,
    repack_quantized_weight,
)

log = logging.getLogger(__name__)


def route_summary(prepared_weights) -> str:
    """One line such as ``int4 kernel: 161 modules; cached fp16: 7 (int8)`` for a model load."""
    n4 = 0
    reasons: dict[str, int] = {}
    for p in prepared_weights:
        if isinstance(p, Int4Layout):
            n4 += 1
        elif isinstance(p, CachedFp16Weight):
            reasons[p.reason] = reasons.get(p.reason, 0) + 1
    out = f"int4 kernel: {n4} modules"
    if reasons:
        detail = ", ".join(f"{k} x{v}" if len(reasons) > 1 else k for k, v in sorted(reasons.items()))
        out += f"; cached fp16: {sum(reasons.values())} ({detail})"
    return out


@dataclass
class CachedFp16Weight:
    """fp16 ``[out, in]`` weight used for int8, fp16, and unsupported int4 modules."""

    weight: torch.Tensor
    reason: str = "fp16"

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])


def _fallback(weight: torch.Tensor, name: str | None, reason: str) -> CachedFp16Weight:
    log.debug("module %s uses the cached fp16 path (%s)", name or "<unnamed>", reason)
    return CachedFp16Weight(weight.to(torch.float16).contiguous(), reason)


def prepare_for_kernel(weight, name: str | None = None, device=None) -> Int4Layout | CachedFp16Weight:
    """Prepare ``weight`` for `kernel_linear`.

    Supported int4 weights become an `Int4Layout`. Int8 weights, plain fp16 tensors, and int4
    weights with an unsupported shape or group size become a `CachedFp16Weight`; the module
    name is logged at DEBUG level (the model load logs one aggregated line).
    """
    if isinstance(weight, QuantizedWeight):
        k = weight.qweight.shape[0] * 8
        gs = weight.group_size
        if gs not in SUPPORTED_GROUP_SIZES or k % gs or weight.qweight.shape[1] % 8:
            w = reference_dequantize(weight)
            if device is not None:
                w = w.to(device)
            return _fallback(w, name, f"unsupported int4 shape K={k} group_size={gs}")
        layout = repack_quantized_weight(weight)
        if device is not None:
            layout = Int4Layout(
                layout.packed.to(device), layout.scales.to(device),
                None if layout.zeros is None else layout.zeros.to(device),
                None if layout.perm is None else layout.perm.to(device),
                layout.group_size, layout.in_features, layout.out_features,
            )
        return layout
    if isinstance(weight, Int8Weight):
        return _fallback(dequantize_int8_weight(weight, device), name, "int8")
    if isinstance(weight, torch.Tensor):
        w = weight if device is None else weight.to(device)
        return _fallback(w, name, "fp16")
    raise TypeError(f"cannot prepare weight of type {type(weight).__name__}")


def kernel_linear(x: torch.Tensor, prepared: Int4Layout | CachedFp16Weight) -> torch.Tensor:
    """``x @ W.T`` for a prepared weight. ``x`` is ``[..., K]``."""
    if isinstance(prepared, CachedFp16Weight):
        return F.linear(x, prepared.weight.to(x.dtype) if prepared.weight.dtype != x.dtype else prepared.weight)
    if isinstance(prepared, Int4Layout):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[0] > MAX_GEMV_M:
            return _int4_prefill(x2, prepared).reshape(*lead, prepared.out_features)
        return int4_gemv(x2, prepared).reshape(*lead, prepared.out_features)
    raise TypeError(f"unknown prepared weight {type(prepared).__name__}")



# ---------------------------------------------------------------------------------------------
# Prefill (M > 16): dequantize into a reused per-device fp16 buffer, then F.linear (task 28.4).
# No full fp16 weight cache is kept; the transient buffer is sized to the largest layer seen.
# ---------------------------------------------------------------------------------------------

_DEQUANT_KERNEL = None
_PREFILL_BUFFERS: dict[tuple[str, int | None], torch.Tensor] = {}


def _build_dequant_kernel():
    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415

    @triton.jit
    def _int4_dequant_kernel(
        w_ptr, s_ptr, z_ptr, perm_ptr, out_ptr, N, K,
        GROUP: tl.constexpr, HAS_ZEROS: tl.constexpr, HAS_PERM: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)  # kernel column order
        mask = (offs_n < N)[:, None] & (offs_k < K)[None, :]
        word = tl.load(w_ptr + offs_n[:, None] * (K // 8) + (offs_k // 8)[None, :], mask=mask, other=0)
        q = ((word >> ((offs_k % 8) * 4)[None, :]) & 15).to(tl.float32)
        g = offs_k // GROUP
        ng = K // GROUP
        s = tl.load(s_ptr + offs_n[:, None] * ng + g[None, :], mask=mask, other=0.0).to(tl.float32)
        if HAS_ZEROS:
            z = tl.load(z_ptr + offs_n[:, None] * ng + g[None, :], mask=mask, other=0.0).to(tl.float32)
        else:
            z = 8.0
        w = ((q - z) * s).to(tl.float16)
        if HAS_PERM:
            dst = tl.load(perm_ptr + offs_k, mask=offs_k < K, other=0)  # original column
        else:
            dst = offs_k
        tl.store(out_ptr + offs_n[:, None] * K + dst[None, :], w, mask=mask)

    return _int4_dequant_kernel


def _prefill_buffer(numel: int, device: torch.device) -> torch.Tensor:
    key = (device.type, device.index if device.index is not None else torch.cuda.current_device())
    buf = _PREFILL_BUFFERS.get(key)
    if buf is None or buf.numel() < numel:
        _PREFILL_BUFFERS.pop(key, None)  # free the old one before allocating the larger one
        buf = torch.empty(numel, dtype=torch.float16, device=device)
        _PREFILL_BUFFERS[key] = buf
    return buf


def release_prefill_buffers() -> None:
    """Drop the transient prefill buffers (they are re-created on demand)."""
    _PREFILL_BUFFERS.clear()


def prefill_buffer_bytes() -> int:
    return sum(b.numel() * b.element_size() for b in _PREFILL_BUFFERS.values())


def dequantize_into_buffer(layout: Int4Layout, device: torch.device | None = None) -> torch.Tensor:
    """Dequantize ``layout`` into the shared fp16 buffer; returns an ``[N, K]`` view.

    The columns are in the original order (the act-order perm is applied on the store), so
    no activation permutation is needed. The view is overwritten by the next call.
    """
    global _DEQUANT_KERNEL
    import triton  # noqa: PLC0415

    n, k = layout.out_features, layout.in_features
    dev = device or layout.packed.device
    buf = _prefill_buffer(n * k, dev)
    if _DEQUANT_KERNEL is None:
        _DEQUANT_KERNEL = _build_dequant_kernel()
    zeros = layout.zeros if layout.zeros is not None else layout.scales
    perm = layout.perm if layout.perm is not None else layout.packed
    block_n, block_k = 16, 128
    _DEQUANT_KERNEL[(triton.cdiv(n, block_n), triton.cdiv(k, block_k))](
        layout.packed, layout.scales, zeros, perm, buf, n, k,
        GROUP=layout.group_size, HAS_ZEROS=layout.zeros is not None, HAS_PERM=layout.perm is not None,
        BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4,
    )
    return buf[: n * k].view(n, k)


def _int4_prefill(x2: torch.Tensor, layout: Int4Layout) -> torch.Tensor:
    """M > 16 on int4: dequantize to the reused fp16 buffer, then ``F.linear``."""
    if x2.dtype != torch.float16 or not x2.is_cuda:
        raise ValueError("x must be a CUDA fp16 tensor")
    if x2.shape[-1] != layout.in_features:
        raise ValueError(f"x has K={x2.shape[-1]}, layout expects {layout.in_features}")
    w = dequantize_into_buffer(layout, x2.device)
    return F.linear(x2, w)
