"""Property 6: Quantization error bound.

**Validates: Requirements 5.5**

Requirement 5.5: |dequantized - original| <= scale/2 plus fp16 rounding error.
The fp16 rounding error is made explicit as follows (eps = 2**-11, half an fp16 ulp):

* the stored scale is rounded to fp16, which shifts a dequantized value
  ``(q - zero) * scale`` by at most ``|q - zero| * scale * eps <= 15 * scale * eps``;
* the dequantized matrix is cast to fp16, which adds at most ``|w_hat| * eps``,
  where ``|w_hat| <= |w| + scale``.

The quantizer already rounds against the stored fp16 scale, so the scale/2 term
is exact; only the two fp16 terms above are allowed on top of it.
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import quantize_groupwise

EPS = 2.0**-11


@st.composite
def weights(draw):
    group_size = 128
    out_features = 8 * draw(st.integers(1, 3))
    n_groups = draw(st.integers(1, 3))
    seed = draw(st.integers(0, 2**32 - 1))
    magnitude = draw(st.sampled_from([1e-3, 0.05, 1.0, 20.0]))
    kind = draw(st.sampled_from(["normal", "uniform", "positive", "sparse"]))
    rng = np.random.default_rng(seed)
    shape = (out_features, n_groups * group_size)
    if kind == "normal":
        w = rng.normal(0, magnitude, shape)
    elif kind == "uniform":
        w = rng.uniform(-magnitude, magnitude, shape)
    elif kind == "positive":
        w = rng.uniform(0, magnitude, shape)
    else:
        w = rng.normal(0, magnitude, shape) * (rng.random(shape) < 0.05)
    return torch.from_numpy(w.astype(np.float32)), group_size


@settings(max_examples=100, deadline=None)
@given(data=weights(), symmetric=st.booleans())
def test_quantization_error_bound(data, symmetric):
    w, group_size = data
    qw = quantize_groupwise(w, group_size=group_size, symmetric=symmetric)
    w_hat = reference_dequantize(qw).to(torch.float32)
    assert w_hat.shape == w.shape

    scale = qw.scales.to(torch.float32).T  # [out, n_groups]
    scale = scale.repeat_interleave(group_size, dim=1)  # [out, in]
    err = (w_hat - w).abs()
    tol = 0.5 * scale + 15 * scale * EPS + (w.abs() + scale) * EPS
    excess = err - tol
    assert bool((excess <= 1e-12).all()), (
        f"max excess {excess.max().item():.3e}, "
        f"err/scale {(err / scale).max().item():.4f}"
    )
