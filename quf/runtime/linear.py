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


class FusedProjection:
    """Several QUFLinear modules sharing one input, computed as a single matmul.

    Built from modules whose ``prepared`` weights are plain ``[out, in]`` tensors of one dtype
    (the cpu and torch_gpu Backends, whatever the original precision). The weights and biases
    are concatenated along the output axis; each source module's ``prepared`` and ``bias`` are
    then re-pointed at views of the fused tensors, so the unfused copies are freed and memory
    does not grow. ``forward`` returns a tuple of outputs split at the original widths.
    """

    SUPPORTED_BACKENDS = ("cpu", "torch_gpu")

    @classmethod
    def can_fuse(cls, modules: list[QUFLinear]) -> bool:
        first = modules[0]
        if getattr(first.backend, "name", None) not in cls.SUPPORTED_BACKENDS:
            return False
        for m in modules:
            p = m.prepared
            if not isinstance(p, torch.Tensor) or p.ndim != 2:
                return False
            if p.dtype != first.prepared.dtype or p.device != first.prepared.device:
                return False
            if m.in_features != first.in_features or m.backend is not first.backend:
                return False
        return True

    def __init__(self, modules: list[QUFLinear]):
        if not self.can_fuse(modules):
            raise ValueError("modules cannot be fused")
        self.backend = modules[0].backend
        self.widths = [int(m.out_features) for m in modules]
        self.prepared = torch.cat([m.prepared for m in modules], dim=0).contiguous()
        self.bias = None
        if any(m.bias is not None for m in modules):
            dtype = next(m.bias.dtype for m in modules if m.bias is not None)
            self.bias = torch.cat(
                [
                    m.bias.to(dtype) if m.bias is not None
                    else torch.zeros(w, dtype=dtype, device=self.prepared.device)
                    for m, w in zip(modules, self.widths)
                ]
            ).contiguous()
        # Free the unfused tensors: the modules keep views into the fused storage.
        start = 0
        for m, w in zip(modules, self.widths):
            m.prepared = self.prepared[start : start + w]
            if m.bias is not None:
                m.bias = self.bias[start : start + w]
            start += w

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y = self.backend.matmul(x, self.prepared)
        if self.bias is not None:
            y = y + self.bias.to(device=y.device, dtype=y.dtype)
        return torch.split(y, self.widths, dim=-1)

    __call__ = forward
