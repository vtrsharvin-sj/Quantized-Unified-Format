"""Fused projections for the Triton path: q/k/v and gate/up as one weight (task 28.5).

Layers that read the same input can run as one GEMV with the outputs concatenated along N.
`fuse_projections` fuses `Int4Layout` members that share ``K``, ``group_size`` and the act-order
permutation. Anything else is left as separate members and `fused_linear` runs them one by one
(always correct, just slower). Nothing here is wired into the model; backend integration does that.

Why the permutation must match: the kernel reads the activation in kernel column order
``x[..., perm]``, and a fused weight has a single ``perm``. Rows with different perms cannot be
expressed that way, and re-permuting a member's columns to a common order would move values
across quantization groups, so no common perm is ever picked. Mismatches refuse to fuse.

Requirements 33.8, 33.10
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch

from quf.quant.quantize import SYMMETRIC_ZERO
from quf.runtime.kernels.dispatch import CachedFp16Weight, kernel_linear
from quf.runtime.kernels.triton_kernels import Int4Layout

log = logging.getLogger(__name__)

_LOGGED: set[str] = set()


def reset_fusion_log() -> None:
    """Forget which refusal reasons were already logged (for tests)."""
    _LOGGED.clear()


def _log_once(kind: str, message: str, *args) -> None:
    if kind not in _LOGGED:
        _LOGGED.add(kind)
        log.info(message + " (logged once; later occurrences are silent)", *args)


@dataclass
class FusedKernelWeight:
    """Result of `fuse_projections`.

    ``fused`` is the single weight when fusion succeeded, else ``None`` and ``parts`` holds the
    original members. ``widths`` are the member output widths, in order.
    """

    widths: tuple[int, ...]
    fused: Int4Layout | CachedFp16Weight | None = None
    parts: tuple[Int4Layout | CachedFp16Weight, ...] | None = None
    reason: str = ""

    @property
    def is_fused(self) -> bool:
        return self.fused is not None

    @property
    def in_features(self) -> int:
        src = self.fused if self.fused is not None else self.parts[0]
        return int(src.in_features)


def _perm_of(layout: Int4Layout) -> torch.Tensor | None:
    """The perm, or ``None`` when it is absent or the identity."""
    p = layout.perm
    if p is None:
        return None
    if torch.equal(p, torch.arange(p.numel(), device=p.device, dtype=p.dtype)):
        return None
    return p


def _same_perm(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.device == b.device and torch.equal(a, b)


def _fuse_int4(parts: Sequence[Int4Layout]) -> Int4Layout:
    first = parts[0]
    any_zeros = any(p.zeros is not None for p in parts)
    zeros = None
    if any_zeros:
        zeros = torch.cat([
            p.zeros if p.zeros is not None
            else torch.full_like(p.scales, float(SYMMETRIC_ZERO)) for p in parts
        ], dim=0).contiguous()
    return Int4Layout(
        packed=torch.cat([p.packed for p in parts], dim=0).contiguous(),
        scales=torch.cat([p.scales for p in parts], dim=0).contiguous(),
        zeros=zeros,
        perm=_perm_of(first),
        group_size=first.group_size,
        in_features=first.in_features,
        out_features=sum(p.out_features for p in parts),
    )


def fuse_projections(parts: Sequence[Int4Layout | CachedFp16Weight], name: str | None = None) -> FusedKernelWeight:
    """Fuse prepared projection weights that share their input into one weight when it is exact.

    * All `Int4Layout` with equal ``K``, ``group_size`` and perm: concatenated along N. Symmetric
      and asymmetric members mix by materializing ``zeros = 8`` for the symmetric ones.
    * All `CachedFp16Weight` with equal ``K``: one concatenated fp16 weight.
    * Anything else (different perms, int4 mixed with fp16, different K or group size): not
      fused; the members are kept and run separately. Refusals are logged at INFO once.
    """
    parts = tuple(parts)
    if not parts:
        raise ValueError("fuse_projections needs at least one member")
    widths = tuple(int(p.out_features) for p in parts)

    def refuse(kind: str, why: str) -> FusedKernelWeight:
        _log_once(kind, "not fusing projection group %s: %s; using separate calls", name or "<unnamed>", why)
        return FusedKernelWeight(widths=widths, parts=parts, reason=why)

    if len({int(p.in_features) for p in parts}) != 1:
        return refuse("k", "members have different in_features")
    if len(parts) == 1:
        return FusedKernelWeight(widths=widths, fused=parts[0], reason="single member")

    if all(isinstance(p, Int4Layout) for p in parts):
        if len({p.group_size for p in parts}) != 1:
            return refuse("gs", "members have different group_size")
        ref = _perm_of(parts[0])
        if not all(_same_perm(ref, _perm_of(p)) for p in parts[1:]):
            return refuse("perm", "act-order permutations differ between members")
        return FusedKernelWeight(widths=widths, fused=_fuse_int4(parts), reason="int4")

    if all(isinstance(p, CachedFp16Weight) for p in parts):
        w = torch.cat([p.weight for p in parts], dim=0).contiguous()
        return FusedKernelWeight(widths=widths, fused=CachedFp16Weight(w, "fp16 fused"), reason="fp16")

    return refuse("mixed", "members mix int4 with fp16/int8 weights")


def split_output(y: torch.Tensor, widths: Sequence[int]) -> tuple[torch.Tensor, ...]:
    """Split ``y[..., sum(widths)]`` into views of the original member widths."""
    if y.shape[-1] != sum(widths):
        raise ValueError(f"output width {y.shape[-1]} != sum of member widths {sum(widths)}")
    return tuple(y.split(list(widths), dim=-1))


def fused_linear(x: torch.Tensor, fw: FusedKernelWeight) -> tuple[torch.Tensor, ...]:
    """Run the group on ``x`` and return one output per member (original widths, in order)."""
    if fw.fused is not None:
        return split_output(kernel_linear(x, fw.fused), fw.widths)
    return tuple(kernel_linear(x, p) for p in fw.parts)
