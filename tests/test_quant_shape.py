"""Property 5: Quantizer output shape and range.

**Validates: Requirements 5.1, 5.4**
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, strategies as st

from quf.quant.dequant import reference_dequantize
from quf.quant.pack import unpack_int4
from quf.quant.quantize import GROUP_SIZE, quantize_groupwise


@st.composite
def weights(draw):
    out_f = 8 * draw(st.integers(1, 4))
    n_groups = draw(st.integers(1, 3))
    in_f = GROUP_SIZE * n_groups
    seed = draw(st.integers(0, 2**31 - 1))
    mag = draw(st.sampled_from([0.0, 1e-4, 0.1, 1.0, 100.0]))
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((out_f, in_f)).astype(np.float32) * mag
    return torch.from_numpy(w)


@given(w=weights(), symmetric=st.booleans())
def test_quantizer_output_shape_and_range(w, symmetric):
    out_f, in_f = w.shape
    n_groups = in_f // GROUP_SIZE
    qw = quantize_groupwise(w, GROUP_SIZE, symmetric=symmetric)

    assert qw.qweight.dtype == torch.int32
    assert tuple(qw.qweight.shape) == (in_f // 8, out_f)
    # Packed along the input dimension: [in/8, out] -> [out, in/8] -> [out, in].
    q = unpack_int4(np.ascontiguousarray(qw.qweight.numpy().T))
    assert q.shape == (out_f, in_f)
    assert q.min() >= 0 and q.max() <= 15

    assert qw.scales.dtype == torch.float16
    assert tuple(qw.scales.shape) == (n_groups, out_f)

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
