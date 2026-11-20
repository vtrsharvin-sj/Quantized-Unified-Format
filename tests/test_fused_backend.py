"""Property 20: Fused backend matches the reference.

Feature: quf-quantized-format, Property 20: Fused backend matches the reference

The eager variant runs on CPU. The CUDA variants carry the ``gpu`` marker and also compare
against the TorchGPUBackend (cached dequantization) output, and cover the compiled path
when ``torch.compile`` works.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, strategies as st

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import quantize_groupwise
from quf.runtime.backends.torch_fused import TorchGPUFusedBackend
from quf.runtime.backends.torch_gpu import TorchGPUBackend

ATOL = 1e-2


def _make(seed, out_features, n_groups, batch, symmetric):
    group_size = 128
    in_features = group_size * n_groups
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=gen) * 0.1
    x = (torch.rand(batch, in_features, generator=gen) * 2 - 1).to(torch.float16)
    qw = quantize_groupwise(w, group_size=group_size, symmetric=symmetric)
    return qw, x


def _reference(qw, x):
    """Matmul of x with the Reference_Dequantizer weights, accumulated in fp32."""
    return x.to(torch.float32) @ reference_dequantize(qw).to(torch.float32).t()


_strategies = dict(
    seed=st.integers(0, 2**31 - 1),
    out_features=st.sampled_from([8, 16, 40, 64]),
    n_groups=st.integers(1, 2),
    batch=st.integers(1, 5),
    symmetric=st.booleans(),
)


@given(**_strategies)
def test_fused_eager_cpu_matches_reference(seed, out_features, n_groups, batch, symmetric):
    """**Validates: Requirements 15.2, 15.3, 15.6**"""
    qw, x = _make(seed, out_features, n_groups, batch, symmetric)
    backend = TorchGPUFusedBackend(compile=False)
    prepared = backend.prepare(qw, "cpu")
    y = backend.matmul(x, prepared)

    ref = _reference(qw, x)
    assert y.dtype == torch.float16
    assert y.shape == (batch, out_features)
    assert (y.to(torch.float32) - ref).abs().max().item() <= ATOL
    # Fused weights stay packed: no fp16 [out, in] weight is stored.
    assert prepared.qweight.dtype == torch.int32


@pytest.mark.gpu
@pytest.mark.parametrize("compile_", [False, True], ids=["eager", "compiled"])
@given(**_strategies)
def test_fused_cuda_matches_reference_and_cached(
    compile_, seed, out_features, n_groups, batch, symmetric
):
    """**Validates: Requirements 15.2, 15.3, 15.6**"""
    qw, x = _make(seed, out_features, n_groups, batch, symmetric)
    fused = TorchGPUFusedBackend(compile=compile_)
    cached = TorchGPUBackend()
    y = fused.matmul(x.cuda(), fused.prepare(qw, "cuda")).cpu()
    y_cached = cached.matmul(x.cuda(), cached.prepare(qw, "cuda")).cpu()

    ref = _reference(qw, x)
    assert y.dtype == torch.float16
    assert (y.to(torch.float32) - ref).abs().max().item() <= ATOL
    assert (y.to(torch.float32) - y_cached.to(torch.float32)).abs().max().item() <= ATOL
