"""Kernel_Layout repack, its pure-torch reference, and the Triton_Gate (task 28.1).

Requirements 33.5, 34.1, 34.5, 36.1, 36.2, 36.3
"""

from __future__ import annotations

import sys

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import quantize_groupwise
from quf.runtime.kernels import (
    dequantize_layout,
    permute_activation,
    repack_quantized_weight,
    triton_available,
)
from quf.runtime.kernels.triton_kernels import act_order_perm


def _case(seed, group_size, symmetric, use_g_idx, out_features, n_groups):
    in_features = group_size * n_groups
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=gen) * 0.1
    qw = quantize_groupwise(w, group_size=group_size, symmetric=symmetric)
    g_idx = None
    if use_g_idx:
        g_idx = (torch.randperm(in_features, generator=gen) // group_size).to(torch.int32)
        qw.g_idx = g_idx
    return qw, g_idx, gen


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("use_g_idx", [False, True])
def test_repack_dequant_equals_reference(group_size, symmetric, use_g_idx):
    qw, g_idx, _ = _case(1, group_size, symmetric, use_g_idx, 24, 3)
    layout = repack_quantized_weight(qw)
    assert layout.packed.shape == (24, qw.in_features // 8) and layout.packed.dtype == torch.int32
    assert layout.scales.shape == (24, 3) and layout.scales.dtype == torch.float16
    assert (layout.zeros is None) == symmetric
    assert (layout.perm is None) == (not use_g_idx)
    assert torch.equal(dequantize_layout(layout), reference_dequantize(qw, g_idx))


@given(
    seed=st.integers(0, 2**16),
    group_size=st.sampled_from([32, 64, 128]),
    symmetric=st.booleans(),
    use_g_idx=st.booleans(),
    out_features=st.sampled_from([8, 24, 64]),
    n_groups=st.integers(1, 4),
)
@settings(max_examples=40, deadline=None)
def test_property_repack_matches_reference(seed, group_size, symmetric, use_g_idx, out_features, n_groups):
    qw, g_idx, _ = _case(seed, group_size, symmetric, use_g_idx, out_features, n_groups)
    layout = repack_quantized_weight(qw)
    assert torch.equal(dequantize_layout(layout), reference_dequantize(qw, g_idx))


def test_act_order_makes_groups_contiguous_and_permutes_activation():
    qw, g_idx, gen = _case(3, 32, False, True, 8, 4)
    layout = repack_quantized_weight(qw)
    assert torch.equal(g_idx.long()[layout.perm], torch.arange(4).repeat_interleave(32))
    x = torch.randn(3, qw.in_features, generator=gen)
    w_ref = reference_dequantize(qw, g_idx).float()
    w_kernel = dequantize_layout(layout, original_order=False).float()
    assert torch.allclose(x @ w_ref.t(), permute_activation(x, layout) @ w_kernel.t(), atol=1e-4)


def test_uneven_g_idx_is_rejected():
    with pytest.raises(ValueError, match="exactly group_size"):
        act_order_perm(torch.zeros(64, dtype=torch.int32), 32)


def test_gate_honours_env_and_never_raises(monkeypatch):
    monkeypatch.setenv("QUF_NO_TRITON", "1")
    ok, reason = triton_available()
    assert ok is False and "QUF_NO_TRITON" in reason


def test_gate_reports_missing_import(monkeypatch):
    monkeypatch.delenv("QUF_NO_TRITON", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "triton", None)  # import raises ImportError
    ok, reason = triton_available()
    assert ok is False and "triton import failed" in reason


def test_gate_does_not_import_triton_when_disabled(monkeypatch):
    monkeypatch.setenv("QUF_NO_TRITON", "1")
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    triton_available()
    assert "triton" not in sys.modules


def test_fast_extra_metadata():
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    fast = data["project"]["optional-dependencies"]["fast"]
    assert "triton-windows; sys_platform == 'win32'" in fast
    assert "triton; sys_platform == 'linux'" in fast
    assert not any(d.startswith("triton") for d in data["project"]["dependencies"])
