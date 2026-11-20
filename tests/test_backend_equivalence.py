"""Property 10: Backend output matches the reference.

Feature: quf-quantized-format, Property 10: Backend output matches the reference

The reference is the Reference_Dequantizer weight followed by a matrix multiply
computed in fp32 (so the reference itself carries no fp16 accumulation error).
The Backend output is fp16, so the 1e-2 tolerance bounds the fp16 output
rounding (half an ulp is below 4e-3 for |y| < 16) plus fp32 summation order.
Activations and weights are bounded so |y| stays in that range.

GPU variants (Torch_GPU, Marlin) carry the ``gpu`` marker. Marlin supports
symmetric quantization only and is skipped when ``marlin_self_test()`` is False.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, strategies as st

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import quantize_groupwise
from quf.runtime.backends.cpu import CPUBackend
from quf.runtime.backends.marlin import MarlinBackend, marlin_self_test
from quf.runtime.backends.torch_gpu import TorchGPUBackend
from quf.runtime.linear import QUFLinear

ATOL = 1e-2


def _backends() -> list:
    return [
        pytest.param(CPUBackend, "cpu", id="cpu"),
        pytest.param(TorchGPUBackend, "cuda", id="torch_gpu", marks=pytest.mark.gpu),
        pytest.param(MarlinBackend, "cuda", id="marlin", marks=pytest.mark.gpu),
    ]


@pytest.mark.parametrize("backend_cls,device", _backends())
@given(
    seed=st.integers(0, 2**31 - 1),
    out_features=st.sampled_from([8, 16, 40, 64]),
    n_groups=st.integers(1, 2),
    batch=st.integers(1, 5),
    symmetric=st.booleans(),
)
def test_backend_output_matches_reference(
    backend_cls, device, seed, out_features, n_groups, batch, symmetric
):
    """**Validates: Requirements 8.2, 9.2, 9.4**"""
    if backend_cls is MarlinBackend:
        if not marlin_self_test():
            pytest.skip("Marlin self-test failed or Marlin is unavailable")
        symmetric = True  # Marlin supports symmetric quantization only
    group_size = 128
    in_features = group_size * n_groups
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=gen) * 0.1
    x = (torch.rand(batch, in_features, generator=gen) * 2 - 1).to(torch.float16)

    qw = quantize_groupwise(w, group_size=group_size, symmetric=symmetric)
    linear = QUFLinear(qw, backend_cls(), device)
    y = linear(x.to(device)).cpu()

    ref = x.to(torch.float32) @ reference_dequantize(qw).to(torch.float32).t()

    assert y.dtype == torch.float16
    assert y.shape == (batch, out_features)
    assert (y.to(torch.float32) - ref).abs().max().item() <= ATOL
