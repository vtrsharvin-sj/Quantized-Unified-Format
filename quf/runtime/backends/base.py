"""LinearBackend protocol shared by the CPU, Torch_GPU, and Marlin backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from quf.quant.quantize import QuantizedWeight


@dataclass
class Int8Weight:
    """Per-output-channel int8 weight: ``q`` int8 ``[out, in]``, ``scales`` fp16 ``[out]``."""

    q: torch.Tensor
    scales: torch.Tensor

    @property
    def out_features(self) -> int:
        return int(self.q.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.q.shape[1])


def dequantize_int8_weight(w: Int8Weight, device: Any = None) -> torch.Tensor:
    """fp16 ``[out, in]`` equal to ``q * scale`` (same as ``dequantize_int8``), on ``device``."""
    q, s = w.q, w.scales
    if device is not None:
        q, s = q.to(device), s.to(device)
    return (q.to(torch.float32) * s.to(torch.float32).unsqueeze(1)).to(torch.float16)


@runtime_checkable
class LinearBackend(Protocol):
    name: str

    def prepare(self, qw: Any, device: Any, g_idx: torch.Tensor | None = None) -> object:
        """Repack or convert the weight for this backend (load time, in memory only).

        ``qw`` is a ``QuantizedWeight`` (int4, optional ``g_idx``), an ``Int8Weight``, or a plain
        fp16 ``[out, in]`` tensor.
        """
        ...

    def matmul(self, x: torch.Tensor, prepared: object) -> torch.Tensor:
        """Return ``x @ W.T`` for the prepared weight ``W`` of shape ``[out, in]``."""
        ...
