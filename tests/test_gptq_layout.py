"""Property 19: GPTQ output shape, range, and layout.

Feature: quf-quantized-format, Property 19: GPTQ output shape, range, and layout

**Validates: Requirements 14.1, 14.2, 14.8**
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, strategies as st

from quf.quant.dequant import reference_dequantize
from quf.quant.gptq import HessianAccumulator, gptq_quantize
from quf.quant.pack import unpack_int4
from quf.quant.quantize import GROUP_SIZE


@st.composite
def cases(draw):
    out_f = 8 * draw(st.integers(1, 3))
    in_f = GROUP_SIZE * draw(st.integers(1, 2))
    n_samples = draw(st.integers(1, 64))
    seed = draw(st.integers(0, 2**31 - 1))
    mag = draw(st.sampled_from([1e-3, 0.1, 1.0, 10.0]))
    rng = np.random.default_rng(seed)
    w = torch.from_numpy((rng.standard_normal((out_f, in_f)) * mag).astype(np.float32))
    x = torch.from_numpy(rng.standard_normal((n_samples, in_f)).astype(np.float32))
    return w, x


@given(case=cases(), symmetric=st.booleans())
def test_gptq_output_shape_range_and_layout(case, symmetric):
    w, x = case
    out_f, in_f = w.shape
    n_groups = in_f // GROUP_SIZE

    acc = HessianAccumulator(in_f, device="cpu")
    acc.add_batch(x)
    qw = gptq_quantize(w, acc.hessian, symmetric=symmetric, device="cpu")

    # Packed int32 [in/8, out], values in 0..15 once unpacked along the input dim.
    assert qw.qweight.dtype == torch.int32
    assert tuple(qw.qweight.shape) == (in_f // 8, out_f)
    q = unpack_int4(np.ascontiguousarray(qw.qweight.numpy().T))
    assert q.shape == (out_f, in_f)
    assert q.min() >= 0 and q.max() <= 15

    # fp16 scales [in/128, out].
    assert qw.scales.dtype == torch.float16
    assert tuple(qw.scales.shape) == (n_groups, out_f)

    # Zero points only for the asymmetric scheme.
    if symmetric:
        assert qw.qzeros is None
    else:
        assert qw.qzeros is not None
        assert qw.qzeros.dtype == torch.int32
        assert tuple(qw.qzeros.shape) == (n_groups, out_f // 8)
        z = unpack_int4(qw.qzeros.numpy())
        assert z.shape == (n_groups, out_f)
        assert z.min() >= 0 and z.max() <= 15

    deq = reference_dequantize(qw)
    assert deq.dtype == torch.float16
    assert tuple(deq.shape) == (out_f, in_f)
