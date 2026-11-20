"""Property 4: Pack/unpack round trip."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

from quf.quant.pack import pack_int4, unpack_int4


@st.composite
def nibble_arrays(draw):
    lead = draw(st.lists(st.integers(1, 4), min_size=0, max_size=2))
    words = draw(st.integers(1, 16))
    shape = (*lead, words * 8)
    return draw(hnp.arrays(np.uint8, shape, elements=st.integers(0, 15)))


# Feature: quf-quantized-format, Property 4: Pack/unpack round trip
@given(nibble_arrays())
def test_pack_unpack_round_trip(values):
    """**Validates: Requirements 5.2, 5.3**"""
    packed = pack_int4(values)
    assert packed.dtype == np.int32
    assert packed.shape == (*values.shape[:-1], values.shape[-1] // 8)
    out = unpack_int4(packed)
    assert out.dtype == np.uint8
    assert np.array_equal(out, values)


@given(st.lists(st.integers(0, 15), min_size=8, max_size=8))
def test_canonical_bit_order(vals):
    """Value i of a group occupies bits 4*(i%8), least significant first."""
    packed = pack_int4(np.array(vals, dtype=np.uint8))
    expected = sum(v << (4 * i) for i, v in enumerate(vals))
    assert int(packed.view(np.uint32)[0]) == expected


def test_known_vector_and_sign_bit():
    packed = pack_int4(np.array([0, 1, 2, 3, 4, 5, 6, 15], dtype=np.uint8))
    assert packed.dtype == np.int32
    assert int(packed[0]) == int(np.uint32(0xF6543210).view(np.int32))
    assert int(packed[0]) < 0


def test_rejects_bad_length_and_range():
    with pytest.raises(ValueError):
        pack_int4(np.zeros(7, dtype=np.uint8))
    with pytest.raises(ValueError):
        pack_int4(np.full(8, 16, dtype=np.uint8))
