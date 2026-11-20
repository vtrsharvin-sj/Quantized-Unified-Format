"""Group sizes, MSE clipping, g_idx dequantization, and int8 (task 16.2)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, strategies as st

from quf.quant.clip import CLIP_RATIOS
from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import (
    GroupSizeError,
    dequantize_int8,
    quantize_groupwise,
    quantize_int8,
)


def test_clip_grid_includes_one_and_half():
    assert CLIP_RATIOS[0] == 1.0 and CLIP_RATIOS[-1] == 0.5


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("symmetric", [False, True])
def test_group_sizes_shapes(group_size, symmetric):
    w = torch.randn(16, 256)
    qw = quantize_groupwise(w, group_size, symmetric)
    assert qw.scales.shape == (256 // group_size, 16)
    assert reference_dequantize(qw).shape == w.shape


@pytest.mark.parametrize("group_size", [32, 64])
def test_group_size_error(group_size):
    with pytest.raises(GroupSizeError) as exc:
        quantize_groupwise(torch.zeros(8, 100), group_size)
    assert "100" in str(exc.value) and str(group_size) in str(exc.value)


@given(
    seed=st.integers(0, 10_000),
    group_size=st.sampled_from([32, 64, 128]),
    symmetric=st.booleans(),
)
def test_clipping_no_worse_than_rtn(seed, group_size, symmetric):
    """Feature: quf-quantized-format, Property 26: MSE clipping is no worse than RTN, same layout."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(8, 256, generator=gen) * torch.rand(8, 256, generator=gen) ** 3
    rtn = quantize_groupwise(w, group_size, symmetric)
    clipped = quantize_groupwise(w, group_size, symmetric, clip_search=True)
    assert clipped.qweight.shape == rtn.qweight.shape and clipped.qweight.dtype == torch.int32
    assert clipped.scales.shape == rtn.scales.shape and clipped.scales.dtype == torch.float16
    assert (clipped.qzeros is None) == (rtn.qzeros is None)

    def err(qw):
        d = reference_dequantize(qw).to(torch.float32) - w
        return (d**2).reshape(8, 256 // group_size, group_size).sum(-1)

    # The reference output is rounded to fp16 (relative step 2**-11), so allow that slack.
    slack = group_size * (w.abs().max().item() * 2**-10) ** 2
    assert bool((err(clipped) <= err(rtn) + slack).all())


def test_g_idx_selects_group_params():
    w = torch.randn(8, 128)
    qw = quantize_groupwise(w, 32)
    default = reference_dequantize(qw)
    identity = torch.arange(128) // 32
    assert torch.equal(reference_dequantize(qw, identity), default)
    # reversed group assignment gathers different scales
    rev = (3 - identity).to(torch.int32)
    out = reference_dequantize(qw, rev.numpy())
    assert not torch.equal(out, default)
    q_cols = reference_dequantize(qw, rev)
    assert torch.equal(out, q_cols)


def test_int8_round_trip_error_bound():
    w = torch.randn(16, 64)
    q, s = quantize_int8(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float16 and s.shape == (16,)
    deq = dequantize_int8(q, s)
    assert deq.dtype == torch.float16 and deq.shape == w.shape
    assert bool(((deq.float() - w).abs() <= s.float().unsqueeze(1) * 0.5 + 1e-3).all())
    assert torch.equal(deq, (q.float() * s.float().unsqueeze(1)).half())
