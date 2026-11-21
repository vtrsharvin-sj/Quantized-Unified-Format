"""Triton_Int4 backend: int4 weights stay packed on the GPU and run through the Triton kernels.

``prepare`` calls `prepare_for_kernel`: supported int4 modules become an `Int4Layout` (about a
quarter of the fp16 bytes); int8, fp16 and unsupported int4 modules become a cached fp16 weight.
``matmul`` calls `kernel_linear` (GEMV for M <= 16, transient dequantize + F.linear above).

Requirements 33-36
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import torch

from quf.quant.quantize import QuantizedWeight
from quf.runtime.kernels.dispatch import kernel_linear, prepare_for_kernel

log = logging.getLogger(__name__)


class TritonInt4Backend:
    name = "triton_int4"

    def prepare(self, qw: Any, device: Any = "cuda", g_idx: torch.Tensor | None = None, name: str | None = None) -> object:
        dev = torch.device("cuda" if device is None else device)
        if dev.type != "cuda":
            raise ValueError(f"triton_int4 backend requires a CUDA device, got {dev}")
        if isinstance(qw, QuantizedWeight) and g_idx is not None:
            qw = dataclasses.replace(qw, g_idx=g_idx.to(dev))
        return prepare_for_kernel(qw, name=name, device=dev)

    def matmul(self, x: torch.Tensor, prepared: object) -> torch.Tensor:
        return kernel_linear(x, prepared)  # type: ignore[arg-type]


def triton_int4_self_test(seed: int = 0, tolerance: float = 2e-2) -> tuple[bool, str]:
    """Run a small layer through the kernel and compare with `reference_dequantize`.

    Covers the GEMV path (M = 1, 4) and the prefill path (M = 32), with and without act-order.
    Returns ``(ok, reason)``. Never raises.
    """
    try:
        from quf.quant.dequant import reference_dequantize  # noqa: PLC0415
        from quf.quant.quantize import quantize_groupwise  # noqa: PLC0415
        from quf.runtime.kernels.triton_kernels import triton_available  # noqa: PLC0415

        ok, why = triton_available()
        if not ok:
            return False, why
        gen = torch.Generator().manual_seed(seed)
        be = TritonInt4Backend()
        n, k = 128, 256
        for symmetric in (False, True):
            qw = quantize_groupwise(torch.randn(n, k, generator=gen) * 0.1, 64, symmetric=symmetric)
            for use_gidx in (False, True):
                g_idx = None
                if use_gidx:
                    g_idx = (torch.randperm(k, generator=gen) // 64).to(torch.int32)
                ref_w = (reference_dequantize(qw) if g_idx is None else reference_dequantize(qw, g_idx)).cuda()
                prepared = be.prepare(qw, "cuda", None if g_idx is None else g_idx.cuda())
                for m in (1, 4, 32):
                    x = (torch.randn(m, k, generator=gen)).to(torch.float16).cuda()
                    out = be.matmul(x, prepared)
                    torch.cuda.synchronize()
                    expected = x.float() @ ref_w.float().t()
                    if not bool(torch.isfinite(out).all()):
                        return False, f"non-finite output (M={m}, symmetric={symmetric}, g_idx={use_gidx})"
                    err = (out.float() - expected).abs().max().item()
                    if err > tolerance:
                        return False, (
                            f"max error {err:.4g} > {tolerance} (M={m}, symmetric={symmetric}, g_idx={use_gidx})"
                        )
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - the self-test must never raise
        return False, f"{type(exc).__name__}: {exc}"
