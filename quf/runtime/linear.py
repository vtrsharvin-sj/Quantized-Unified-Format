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
        name: str | None = None,
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
        self.name = name
        kwargs = {"name": name} if name and getattr(backend, "name", None) == "triton_int4" else {}
        self.prepared = (
            backend.prepare(qw, device, **kwargs) if g_idx is None else backend.prepare(qw, device, g_idx, **kwargs)
        )
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
    KERNEL_BACKEND = "triton_int4"

    @classmethod
    def can_fuse(cls, modules: list[QUFLinear]) -> bool:
        first = modules[0]
        if getattr(first.backend, "name", None) == cls.KERNEL_BACKEND:
            # fuse_projections decides per group; it never fails (it falls back to separate calls).
            return all(m.backend is first.backend and m.in_features == first.in_features for m in modules)
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
        self._kernel = None
        if getattr(self.backend, "name", None) == self.KERNEL_BACKEND:
            self._init_kernel(modules)
            return
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

    @staticmethod
    def _group_name(modules: list[QUFLinear]) -> str | None:
        """Group name such as `model.layers.3.self_attn.qkv` from member names `...q_proj` etc."""
        names = [m.name for m in modules if getattr(m, "name", None)]
        if len(names) != len(modules):
            return None
        parent = names[0].rpartition(".")[0]
        leaves = [n.rpartition(".")[2] for n in names]
        stems = [l[:-5] if l.endswith("_proj") else l for l in leaves]
        sep = "" if all(len(s) == 1 for s in stems) else "_"
        return (parent + "." if parent else "") + sep.join(stems)

    def _init_kernel(self, modules: list[QUFLinear]) -> None:
        from quf.runtime.kernels.fusion import fuse_projections  # noqa: PLC0415

        self._kernel = fuse_projections([m.prepared for m in modules], name=self._group_name(modules))
        self.prepared = self._kernel.fused
        self.bias = None
        self._part_modules = modules
        if self._kernel.is_fused:
            if any(m.bias is not None for m in modules):
                dtype = next(m.bias.dtype for m in modules if m.bias is not None)
                device = next(m.bias.device for m in modules if m.bias is not None)
                self.bias = torch.cat([
                    m.bias.to(dtype) if m.bias is not None
                    else torch.zeros(w, dtype=dtype, device=device)
                    for m, w in zip(modules, self.widths)
                ]).contiguous()
            # Free the unfused weights: the fused one replaces them.
            for m in modules:
                m.prepared = None
                m.bias = None

    def _forward_kernel(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from quf.runtime.kernels.dispatch import kernel_linear  # noqa: PLC0415
        from quf.runtime.kernels.fusion import split_output  # noqa: PLC0415

        if not self._kernel.is_fused:
            return tuple(m(x) for m in self._part_modules)
        y = kernel_linear(x, self.prepared)
        if self.bias is not None:
            y = y + self.bias.to(device=y.device, dtype=y.dtype)
        return split_output(y, self.widths)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self._kernel is not None:
            return self._forward_kernel(x)
        y = self.backend.matmul(x, self.prepared)
        if self.bias is not None:
            y = y + self.bias.to(device=y.device, dtype=y.dtype)
        return torch.split(y, self.widths, dim=-1)

    __call__ = forward
