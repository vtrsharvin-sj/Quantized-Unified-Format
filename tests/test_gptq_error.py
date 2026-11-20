"""Property 18: GPTQ error is no worse than round-to-nearest."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quf.quant.dequant import reference_dequantize
from quf.quant.gptq import HessianAccumulator, gptq_quantize
from quf.quant.quantize import quantize_groupwise


def _output_error(w: torch.Tensor, wq: torch.Tensor, x: torch.Tensor) -> float:
    d = (w.to(torch.float64) - wq.to(torch.float64)) @ x.to(torch.float64).T
    return float((d * d).sum())


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    out_features=st.sampled_from([8, 16, 32]),
    in_groups=st.sampled_from([1, 2]),
    n_samples=st.sampled_from([256, 512]),
    symmetric=st.booleans(),
    correlated=st.booleans(),
)
def test_gptq_error_no_worse_than_rtn(
    seed, out_features, in_groups, n_samples, symmetric, correlated
):
    """Feature: quf-quantized-format, Property 18: GPTQ error is no worse than round-to-nearest

    **Validates: Requirements 14.7**
    """
    in_features = 128 * in_groups
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=g)
    x = torch.randn(n_samples, in_features, generator=g)
    if correlated:
        # Correlated, unevenly scaled features make the Hessian non-trivial.
        mix = torch.randn(in_features, in_features, generator=g) * 0.1
        mix += torch.eye(in_features)
        x = (x @ mix) * (0.5 + torch.rand(in_features, generator=g) * 2.0)

    acc = HessianAccumulator(in_features, device="cpu")
    acc.add_batch(x)

    gq = gptq_quantize(w, acc.hessian, symmetric=symmetric, device="cpu")
    rq = quantize_groupwise(w, symmetric=symmetric)

    err_gptq = _output_error(w, reference_dequantize(gq), x)
    err_rtn = _output_error(w, reference_dequantize(rq), x)

    assert err_gptq <= err_rtn * (1 + 1e-3), (err_gptq, err_rtn)
