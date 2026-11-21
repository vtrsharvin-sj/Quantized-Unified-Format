"""Fused_Projection: fused q/k/v and gate/up equal the separate projections (CPU)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quf.quant.quantize import quantize_groupwise
from quf.runtime.backends.cpu import CPUBackend
from quf.runtime.backends.torch_fused import TorchGPUFusedBackend
from quf.runtime.backends.torch_gpu import TorchGPUBackend
from quf.runtime.generate import generate
from quf.runtime.linear import FusedProjection, QUFLinear
from quf.runtime.model import QUFModel
from test_backend_precisions import _case
from test_model import build_tiny_quf


# Feature: quf-quantized-format, Property 33: Fused projection equals separate projections
@settings(max_examples=100, deadline=None)
@given(
    widths=st.lists(st.integers(1, 12), min_size=2, max_size=3),
    d_in=st.integers(1, 16),
    rows=st.integers(1, 4),
    with_bias=st.booleans(),
    seed=st.integers(0, 2**16),
)
def test_fused_equals_separate(widths, d_in, rows, with_bias, seed):
    """**Validates: Requirements 31.5, 31.7**"""
    g = torch.Generator().manual_seed(seed)
    be = CPUBackend()
    mods = [
        QUFLinear(
            torch.randn(w, d_in, generator=g).half(),
            be,
            "cpu",
            bias=torch.randn(w, generator=g) if with_bias else None,
        )
        for w in widths
    ]
    x = torch.randn(1, rows, d_in, generator=g)
    separate = [m(x) for m in mods]
    fused = FusedProjection(mods)
    for a, b in zip(fused(x), separate):
        assert (a - b).abs().max().item() <= 1e-3
    # The modules now view the fused storage, so no unfused copy stays in memory.
    assert mods[0].prepared.untyped_storage().data_ptr() == fused.prepared.untyped_storage().data_ptr()


_mixed = dict(
    kinds=st.lists(st.sampled_from(["int4", "int8", "fp16"]), min_size=2, max_size=3),
    outs=st.lists(st.sampled_from([8, 24, 40]), min_size=3, max_size=3),
    group_size=st.sampled_from([32, 64]),
    n_groups=st.integers(1, 2),
    use_g_idx=st.booleans(),
    with_bias=st.booleans(),
    rows=st.integers(1, 3),
    seed=st.integers(0, 2**16),
)


def _check_mixed(backend, device, kinds, outs, group_size, n_groups, use_g_idx, with_bias, rows, seed):
    mods, x = [], None
    for i, kind in enumerate(kinds):
        qw, g_idx, x_i, _ = _case(seed + i, kind, group_size, use_g_idx, i % 2 == 0, outs[i], n_groups, rows)
        x = x_i if x is None else x
        bias = torch.randn(outs[i], generator=torch.Generator().manual_seed(seed + 99 + i)) if with_bias else None
        mods.append(QUFLinear(qw, backend, device, bias=bias, g_idx=g_idx))
    x = x.to(device)
    separate = [m(x).clone() for m in mods]
    fused = FusedProjection(mods)
    outs_f = fused(x)
    assert [o.shape[-1] for o in outs_f] == [m.out_features for m in mods]
    for a, b in zip(outs_f, separate):
        assert (a.float() - b.float()).abs().max().item() <= 1e-3
    # The unfused modules still produce the same output after being re-pointed at fused views.
    for m, b in zip(mods, separate):
        assert (m(x).float() - b.float()).abs().max().item() <= 1e-3


# Feature: quf-quantized-format, Property 33: Fused projection equals separate projections
@settings(max_examples=100, deadline=None)
@given(**_mixed)
def test_fused_mixed_precisions_g_idx_bias_cpu(kinds, outs, group_size, n_groups, use_g_idx, with_bias, rows, seed):
    """**Validates: Requirements 31.5, 31.7**"""
    _check_mixed(CPUBackend(), "cpu", kinds, outs, group_size, n_groups, use_g_idx, with_bias, rows, seed)


# Feature: quf-quantized-format, Property 33: Fused projection equals separate projections
@pytest.mark.gpu
@settings(max_examples=50, deadline=None)
@given(**_mixed)
def test_fused_mixed_precisions_g_idx_bias_torch_gpu(kinds, outs, group_size, n_groups, use_g_idx, with_bias, rows, seed):
    """**Validates: Requirements 31.5, 31.7**"""
    _check_mixed(TorchGPUBackend(), "cuda", kinds, outs, group_size, n_groups, use_g_idx, with_bias, rows, seed)


def test_torch_fused_backend_falls_back_to_separate_projections():
    """**Validates: Requirements 31.7**: torch_fused modules are never fused."""
    be = TorchGPUFusedBackend(compile=False)
    mods = [QUFLinear(quantize_groupwise(torch.randn(8, 128) * 0.1), be, "cpu") for _ in range(3)]
    assert not FusedProjection.can_fuse(mods)
    with pytest.raises(ValueError):
        FusedProjection(mods)


@pytest.mark.gpu
def test_torch_fused_model_uses_separate_projections(tmp_path):
    path = tmp_path / "tiny.quf"
    build_tiny_quf(path)
    m = QUFModel.load(path, backend="torch_fused")
    if m.backend_name == "torch_fused":
        assert all(l.qkv is None and l.gate_up is None for l in m.layers)


def test_model_fused_greedy_equals_unfused(tmp_path):
    path = tmp_path / "tiny.quf"
    build_tiny_quf(path)
    a = QUFModel.load(path, backend="cpu", fuse_projections=False)
    b = QUFModel.load(path, backend="cpu")
    assert a.layers[0].qkv is None and b.layers[0].qkv is not None and b.layers[0].gate_up is not None
    for prompt in ([1, 2, 3], [5], [7, 7, 7, 7, 2]):
        assert generate(a, None, prompt, max_new_tokens=12) == generate(b, None, prompt, max_new_tokens=12)
