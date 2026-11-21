"""Property 2: Write/read round trip.

Feature: quf-quantized-format, Property 2: Write/read round trip
**Validates: Requirements 3.5, 4.1, 4.7**
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

from quf.format.constants import GROUP_SIZE
from quf.format.reader import QUFReader
from quf.format.schema import QUFHeader
from quf.format.writer import write_quf

_json_leaf = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(2**53), 2**53),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=12),
)
_json_obj = st.dictionaries(
    st.text(min_size=1, max_size=8),
    st.recursive(_json_leaf, lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=6), c, max_size=3), max_leaves=6),
    max_size=5,
)

_dtypes = st.sampled_from([np.float16, np.float32, np.int32])
_shapes = hnp.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=6)


@st.composite
def _tensor(draw):
    dt = draw(_dtypes)
    if dt is np.int32:
        elems = st.integers(-(2**31), 2**31 - 1)
    else:
        # include NaN/inf/subnormals: round trip must be bit-identical
        elems = None
    return draw(hnp.arrays(dt, draw(_shapes), elements=elems))


# names avoid the special suffixes (.qweight, embed_tokens.weight, lm_head.weight)
_names = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789._"), min_size=1, max_size=20
).filter(lambda n: not n.endswith((".qweight", ".g_idx", ".qweight_int8", "embed_tokens.weight", "lm_head.weight")))


@given(
    model_config=_json_obj,
    tokenizer=_json_obj,
    extra_quant=_json_obj,
    tensors=st.dictionaries(_names, _tensor(), max_size=6),
)
def test_write_read_round_trip(model_config, tokenizer, extra_quant, tensors):
    reserved = {"modules", "bits", "group_size", "symmetric", "act_order"}
    extra_quant = {k: v for k, v in extra_quant.items() if k not in reserved}
    quantization = {**extra_quant, "group_size": GROUP_SIZE, "symmetric": True}
    header = QUFHeader(model_config, tokenizer, quantization)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.quf")
        written = write_quf(path, header, tensors)
        r = QUFReader(path)
        try:
            assert r.header.model_config == model_config
            assert r.header.tokenizer == tokenizer
            assert r.header.quantization == quantization
            assert r.header == written
            assert r.tensor_names() == list(tensors)
            for name, arr in tensors.items():
                got = r.tensor(name)
                assert got.dtype == arr.dtype
                assert got.shape == arr.shape
                assert got.tobytes() == np.ascontiguousarray(arr).tobytes()
                del got  # release the mmap view so the file can be removed on Windows
        finally:
            r.close()
