"""CPU backend: Reference_Dequantizer followed by a PyTorch matmul."""

from __future__ import annotations

from typing import Any

import torch

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import QuantizedWeight
from quf.runtime.backends.base import Int8Weight, dequantize_int8_weight


class CPUBackend:
    name = "cpu"

    def prepare(self, qw: Any, device: Any = "cpu", g_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Dequantize once at load time; returns the weight ``[out, in]`` cached in fp32.

        fp16 matmul is not universally supported on the CPU, so the compute dtype is fp32 and
        the cast happens here once, not on every ``matmul``. The cached weight takes 2x the fp16
        bytes (the fp16 intermediate is released). Accepts an int4 ``QuantizedWeight`` (with
        optional ``g_idx``), an ``Int8Weight``, or an fp16 ``[out, in]`` tensor.
        """
        return self._prepare_fp16(qw, g_idx).to(torch.float32)

    def _prepare_fp16(self, qw: Any, g_idx: torch.Tensor | None) -> torch.Tensor:
        if isinstance(qw, Int8Weight):
            return dequantize_int8_weight(qw, "cpu")
        if isinstance(qw, torch.Tensor):
            return qw.to("cpu", torch.float16)
        cpu_qw = QuantizedWeight(
            qweight=qw.qweight.to("cpu"),
            scales=qw.scales.to("cpu"),
            qzeros=None if qw.qzeros is None else qw.qzeros.to("cpu"),
            group_size=qw.group_size,
            in_features=qw.in_features,
            out_features=qw.out_features,
        )
        return reference_dequantize(cpu_qw, None if g_idx is None else g_idx.to("cpu"))

    def matmul(self, x: torch.Tensor, prepared: torch.Tensor) -> torch.Tensor:
        # ``prepared`` is already fp32 (see ``prepare``); ``.to`` is a no-op then.
        y = x.to(torch.float32) @ prepared.to(torch.float32).t()
        return y.to(x.dtype)
