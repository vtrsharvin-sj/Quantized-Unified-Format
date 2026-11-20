"""TorchGPUFused backend: keeps Packed_Int4 on the device and dequantizes inside the matmul.

Weights stay as int32 packed values plus fp16 scales and zeros (about 4x less memory than the
cached fp16 weight). Each call unpacks, dequantizes, and multiplies in one function, which can be
wrapped in ``torch.compile``. When compile or Triton is unavailable, or compilation fails, the
same function runs in eager PyTorch. Numerics follow the Reference_Dequantizer:
``w = (q - zero) * scale`` evaluated in fp32 and rounded to fp16.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch

from quf.quant.dequant import expand_groups
from quf.quant.quantize import QuantizedWeight
from quf.runtime.backends.base import Int8Weight

log = logging.getLogger("quf.runtime.backends.torch_fused")

_SHIFTS_CACHE: dict[torch.device, torch.Tensor] = {}


def _shifts(device: torch.device) -> torch.Tensor:
    s = _SHIFTS_CACHE.get(device)
    if s is None:
        s = torch.arange(8, dtype=torch.int32, device=device) * 4
        _SHIFTS_CACHE[device] = s
    return s


def _unpack_rows(packed: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """int32 ``[R, C]`` packed along dim 0 -> int32 ``[R * 8, C]`` with values 0..15."""
    nib = (packed.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF  # [R, 8, C]
    return nib.reshape(packed.shape[0] * 8, packed.shape[1])


def _unpack_cols(packed: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """int32 ``[R, C]`` packed along dim 1 -> int32 ``[R, C * 8]`` with values 0..15."""
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF  # [R, C, 8]
    return nib.reshape(packed.shape[0], packed.shape[1] * 8)


def _fused_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    shifts: torch.Tensor,
    group_size: int,
    matmul_fp32: bool,
    g_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``x @ dequant(W).T``.

    ``qweight`` is int32 ``[in/8, out]``, ``scales`` fp16 ``[groups, out]``, and ``zeros`` is
    packed int32 ``[groups, out/8]`` (asymmetric) or an empty tensor (symmetric, zero = 8).
    """
    q = _unpack_rows(qweight, shifts).to(torch.float32)  # [in, out]
    z = None if zeros.numel() == 0 else _unpack_cols(zeros, shifts)  # [groups, out]
    s, z = expand_groups(scales, z, g_idx, group_size, q.shape[0])  # [in, out] fp32
    w = ((q - z) * s).to(torch.float16)  # [in, out]
    if matmul_fp32:
        return (x.to(torch.float32) @ w.to(torch.float32)).to(x.dtype)
    return (x.to(torch.float16) @ w).to(x.dtype)


def fused_self_test(seed: int = 0) -> bool:
    """Run a small fused matmul on CUDA and compare with the Reference_Dequantizer."""
    from quf.quant.dequant import reference_dequantize  # noqa: PLC0415
    from quf.quant.quantize import quantize_groupwise  # noqa: PLC0415

    if not torch.cuda.is_available():
        return False
    gen = torch.Generator().manual_seed(seed)
    qw = quantize_groupwise(torch.randn(128, 256, generator=gen) * 0.1)
    x = torch.randn(4, 256, generator=gen).to(torch.float16).cuda()
    be = TorchGPUFusedBackend()
    out = be.matmul(x, be.prepare(qw, "cuda"))
    expected = x @ reference_dequantize(qw).to("cuda").t()
    return bool(torch.isfinite(out).all()) and (out.float() - expected.float()).abs().max().item() <= 1e-2


@dataclass
class FusedWeight:
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    group_size: int
    in_features: int
    out_features: int
    g_idx: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.qweight.device


@dataclass
class FusedInt8Weight:
    """Int8 weight kept as int8 on the device; dequantized inside the matmul."""

    q: torch.Tensor  # int8 [out, in]
    scales: torch.Tensor  # fp16 [out]

    @property
    def device(self) -> torch.device:
        return self.q.device


def _int8_linear(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor, matmul_fp32: bool) -> torch.Tensor:
    w = (q.to(torch.float32) * scales.to(torch.float32).unsqueeze(1)).to(torch.float16)  # [out, in]
    if matmul_fp32:
        return (x.to(torch.float32) @ w.to(torch.float32).t()).to(x.dtype)
    return (x.to(torch.float16) @ w.t()).to(x.dtype)


def _fp16_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if w.device.type != "cuda":
        return (x.to(torch.float32) @ w.to(torch.float32).t()).to(x.dtype)
    return (x.to(torch.float16) @ w.t()).to(x.dtype)


class TorchGPUFusedBackend:
    name = "torch_fused"

    def __init__(self, compile: bool = False):
        """``compile=True`` wraps the fused function in ``torch.compile`` when it works."""
        self._eager: Callable[..., torch.Tensor] = _fused_linear
        self._fn: Callable[..., torch.Tensor] = _fused_linear
        self.compiled = False
        self._compile_requested = compile
        if compile:
            self._try_compile()

    def _try_compile(self) -> None:
        try:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available")
            self._fn = torch.compile(_fused_linear, dynamic=True)
            self.compiled = True
        except Exception as exc:  # noqa: BLE001
            log.warning("torch.compile unavailable, using eager fused backend: %s", exc)
            self._fn = self._eager
            self.compiled = False

    def prepare(self, qw: Any, device: Any = "cuda", g_idx: torch.Tensor | None = None):
        dev = torch.device("cuda" if device is None else device)
        if isinstance(qw, Int8Weight):
            return FusedInt8Weight(qw.q.to(dev).contiguous(), qw.scales.to(dev).contiguous())
        if isinstance(qw, torch.Tensor):
            return qw.to(dev, torch.float16).contiguous()
        zeros = (
            torch.empty(0, dtype=torch.int32, device=dev)
            if qw.qzeros is None
            else qw.qzeros.to(dev).contiguous()
        )
        return FusedWeight(
            qweight=qw.qweight.to(dev).contiguous(),
            scales=qw.scales.to(dev).contiguous(),
            zeros=zeros,
            group_size=qw.group_size,
            in_features=qw.in_features,
            out_features=qw.out_features,
            g_idx=None if g_idx is None else g_idx.to(dev).to(torch.long).contiguous(),
        )

    def matmul(self, x: torch.Tensor, prepared) -> torch.Tensor:
        if isinstance(prepared, torch.Tensor):
            return _fp16_linear(x.to(prepared.device), prepared)
        if x.device != prepared.device:
            x = x.to(prepared.device)
        if isinstance(prepared, FusedInt8Weight):
            return _int8_linear(x, prepared.q, prepared.scales, prepared.device.type != "cuda")
        args = (
            x,
            prepared.qweight,
            prepared.scales,
            prepared.zeros,
            _shifts(prepared.device),
            prepared.group_size,
            prepared.device.type != "cuda",
            prepared.g_idx,
        )
        if self.compiled:
            try:
                return self._fn(*args)
            except Exception as exc:  # noqa: BLE001
                log.warning("compiled fused path failed, falling back to eager: %s", exc)
                self._fn = self._eager
                self.compiled = False
        return self._eager(*args)
