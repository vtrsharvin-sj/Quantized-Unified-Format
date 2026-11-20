"""Torch_GPU backend: Reference_Dequantizer on the GPU followed by a GPU matmul."""

from __future__ import annotations

from typing import Any

import torch

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import QuantizedWeight
from quf.runtime.backends.base import Int8Weight, dequantize_int8_weight


def _resolve_device(device: Any) -> torch.device:
    dev = torch.device("cuda" if device is None else device)
    if dev.type != "cuda":
        raise ValueError(f"Torch_GPU backend requires a CUDA device, got {dev}")
    return dev


class TorchGPUBackend:
    name = "torch_gpu"

    def prepare(self, qw: Any, device: Any = "cuda", g_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Move the weight to the GPU and dequantize it there.

        Returns the fp16 weight ``[out, in]`` on the GPU. Accepts an int4 ``QuantizedWeight``
        (with optional ``g_idx``), an ``Int8Weight``, or an fp16 ``[out, in]`` tensor.
        The input is not modified.
        """
        dev = _resolve_device(device)
        if isinstance(qw, Int8Weight):
            return dequantize_int8_weight(qw, dev)
        if isinstance(qw, torch.Tensor):
            return qw.to(dev, torch.float16)
        gpu_qw = QuantizedWeight(
            qweight=qw.qweight.to(dev),
            scales=qw.scales.to(dev),
            qzeros=None if qw.qzeros is None else qw.qzeros.to(dev),
            group_size=qw.group_size,
            in_features=qw.in_features,
            out_features=qw.out_features,
        )
        if g_idx is None:
            return reference_dequantize(gpu_qw)
        return reference_dequantize(gpu_qw, g_idx.to(dev))

    def matmul(self, x: torch.Tensor, prepared: torch.Tensor) -> torch.Tensor:
        if x.device != prepared.device:
            x = x.to(prepared.device)
        return torch.matmul(x.to(prepared.dtype), prepared.t()).to(x.dtype)
