"""QUFLinear: holds one layer's prepared weight for a single backend."""

from __future__ import annotations

from typing import Any

import torch

from quf.quant.quantize import QuantizedWeight
from quf.runtime.backends.base import Int8Weight, LinearBackend


class QUFLinear:
    def __init__(
        self,
        qw: QuantizedWeight | Int8Weight | torch.Tensor,
        backend: LinearBackend,
        device: Any = "cpu",
        bias: torch.Tensor | None = None,
        g_idx: torch.Tensor | None = None,
    ):
        """``qw`` is an int4 ``QuantizedWeight`` (optionally with ``g_idx``), an ``Int8Weight``,
        or an fp16 ``[out, in]`` tensor."""
        self.backend = backend
        if isinstance(qw, torch.Tensor):
            self.out_features, self.in_features = int(qw.shape[0]), int(qw.shape[1])
            self.precision = "fp16"
        else:
            self.in_features = qw.in_features
            self.out_features = qw.out_features
            self.precision = "int8" if isinstance(qw, Int8Weight) else "int4"
        if g_idx is not None and self.precision != "int4":
            raise ValueError("g_idx applies to int4 modules only")
        # Version 1 call shape (no g_idx) stays unchanged for third-party backends.
        self.prepared = backend.prepare(qw, device) if g_idx is None else backend.prepare(qw, device, g_idx)
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backend.matmul(x, self.prepared)
        if self.bias is not None:
            y = y + self.bias.to(device=y.device, dtype=y.dtype)
        return y

    __call__ = forward
