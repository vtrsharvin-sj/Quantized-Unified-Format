"""Property 35: the Int4 kernel equals the fp32 reference. Requirements 33.1-33.5, 33.7, 33.8"""
from __future__ import annotations

import dataclasses

import pytest
import torch
from hypothesis import given, settings, strategies as st

pytestmark = [pytest.mark.gpu, pytest.mark.triton]

from quf.quant.dequant import reference_dequantize  # noqa: E402
from quf.quant.quantize import quantize_groupwise  # noqa: E402
from quf.runtime.kernels import fuse_projections, fused_linear, kernel_linear, prepare_for_kernel  # noqa: E402
from quf.runtime.kernels.triton_kernels import int4_gemv  # noqa: E402

ATOL, RTOL = 1e-2, 2e-3

widths = st.one_of(st.sampled_from([8, 24, 136, 904]), st.integers(1, 130).map(lambda v: v * 8))
ms = st.one_of(st.integers(1, 16), st.sampled_from([17, 33]))


def _close(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape
    err = (out.float() - ref).abs()
    bound = ATOL + RTOL * ref.abs()
    assert bool((err <= bound).all()), f"max excess {(err - bound).max().item():.4g}"


def _make(  # weights ~0.02 std like real models: fp16 weight/output rounding in the reference stays well below the tolerance
n, k, gs, sym, use_gidx, seed):
    g = torch.Generator().manual_seed(seed)
    qw = quantize_groupwise(0.02 * torch.randn(n, k, generator=g), group_size=gs, symmetric=sym)
    if use_gidx:
        gi = (torch.arange(k) // gs)[torch.randperm(k, generator=g)]
        qw = dataclasses.replace(qw, g_idx=gi.to(torch.int32))
    return qw


def _ref(x, qw):
    w = reference_dequantize(qw, qw.g_idx).float().cuda()
    return x.float() @ w.T


# Feature: quf-quantized-format, Property 35: Int4 kernel equals the fp32 reference
# **Validates: Requirements 33.1, 33.2, 33.3, 33.4, 33.5, 33.7, 33.8**
@settings(max_examples=100, deadline=None)
@given(
    n=widths,
    k_groups=st.integers(1, 48),
    gs=st.sampled_from([32, 64, 128]),
    sym=st.booleans(),
    use_gidx=st.booleans(),
    m=ms,
    x_scale=st.floats(0.05, 2.0),
    group_members=st.lists(widths, min_size=2, max_size=3),
    seed=st.integers(0, 2**31 - 1),
)
def test_property_35_kernel_equals_reference(n, k_groups, gs, sym, use_gidx, m, x_scale, group_members, seed):
    k = k_groups * gs
    torch.manual_seed(seed)
    x = (torch.randn(m, k, device="cuda") * x_scale).half()

    qw = _make(n, k, gs, sym, use_gidx, seed)
    layout = prepare_for_kernel(qw, device="cuda")
    ref = _ref(x, qw)
    if m <= 16:
        _close(int4_gemv(x, layout), ref)
    _close(kernel_linear(x, layout), ref)

    members = [_make(w, k, gs, bool((seed >> i) & 1) if i else sym, use_gidx, seed + i) for i, w in enumerate(group_members)]
    if use_gidx:  # fusing requires a shared permutation
        members = [dataclasses.replace(q, g_idx=members[0].g_idx) for q in members]
    fw = fuse_projections([prepare_for_kernel(q, device="cuda") for q in members])
    outs = fused_linear(x, fw)
    assert len(outs) == len(members)
    for o, q in zip(outs, members):
        _close(o, _ref(x, q))
