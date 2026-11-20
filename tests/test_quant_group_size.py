"""Property 7: Group-size violations rejected.

**Validates: Requirements 5.6**
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, strategies as st

from quf.quant.quantize import GROUP_SIZE, GroupSizeError, quantize_groupwise


@given(
    in_features=st.integers(min_value=1, max_value=600).filter(lambda n: n % GROUP_SIZE != 0),
    out_blocks=st.integers(min_value=1, max_value=2),
    symmetric=st.booleans(),
)
def test_non_multiple_input_dim_raises_with_dim_and_group_size(
    in_features: int, out_blocks: int, symmetric: bool
) -> None:
    w = torch.zeros(out_blocks * 8, in_features)
    with pytest.raises(GroupSizeError) as exc:
        quantize_groupwise(w, symmetric=symmetric)
    msg = str(exc.value)
    assert str(in_features) in msg
    assert str(GROUP_SIZE) in msg
    assert exc.value.dim == in_features
    assert exc.value.group_size == GROUP_SIZE
