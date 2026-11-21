"""Property test for the QUF writer's file layout.

Feature: quf-quantized-format, Property 1: Written file layout invariant
"""

from __future__ import annotations

import json
import struct

import numpy as np
from hypothesis import given, strategies as st

from quf.format.constants import ALIGN, DTYPES, GROUP_SIZE, MAGIC, VERSION
from quf.format.schema import QUFHeader
from quf.format.writer import write_quf

_NP = {"fp16": np.float16, "fp32": np.float32, "int32": np.int32}

_json_leaf = st.one_of(
    st.none(), st.booleans(), st.integers(-1000, 1000), st.text(max_size=20)
)
_json_dict = st.dictionaries(st.text(min_size=1, max_size=8), _json_leaf, max_size=4)

_shape = st.lists(st.integers(0, 9), min_size=0, max_size=3)


@st.composite
def _array(draw, dtype=None):
    code = dtype or draw(st.sampled_from(sorted(_NP)))
    shape = tuple(draw(_shape))
    n = int(np.prod(shape)) if shape else 1
    data = draw(st.lists(st.integers(0, 255), min_size=n, max_size=n))
    return np.array(data, dtype=np.int64).astype(_NP[code]).reshape(shape)


@st.composite
def _tensor_sets(draw):
    tensors: dict[str, np.ndarray] = {}
    names = draw(
        st.lists(
            st.text(alphabet="abcdefgh._0123456789", min_size=1, max_size=12),
            unique=True,
            max_size=5,
        )
    )
    for n in names:
        # Avoid names that trigger special rules; those are covered below.
        if n.endswith((".qweight", ".g_idx", ".qweight_int8", "embed_tokens.weight", "lm_head.weight")):
            continue
        tensors[n] = draw(_array())
    symmetric = draw(st.booleans())
    for i in range(draw(st.integers(0, 2))):
        base = f"layers.{i}.proj"
        tensors[base + ".qweight"] = draw(_array("int32"))
        tensors[base + ".scales"] = draw(_array("fp16"))
        if not symmetric:
            tensors[base + ".qzeros"] = draw(_array("int32"))
    if draw(st.booleans()):
        tensors["model.embed_tokens.weight"] = draw(_array("fp16"))
    if draw(st.booleans()):
        tensors["lm_head.weight"] = draw(_array("fp16"))
    return symmetric, tensors


def test_layout_property(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("layout")

    @given(_tensor_sets(), _json_dict, _json_dict, _json_dict)
    def run(ts, model_config, tokenizer, extra_quant):
        symmetric, tensors = ts
        reserved = {"modules", "bits", "group_size", "symmetric", "act_order"}
        quant = {k: v for k, v in extra_quant.items() if k not in reserved}
        quant.update({"group_size": GROUP_SIZE, "symmetric": symmetric, "bits": 4})
        header = QUFHeader(model_config, tokenizer, quant, [])
        path = tmp / "m.quf"
        write_quf(path, header, tensors)
        raw = path.read_bytes()

        # Prefix: magic, version, header length (little-endian)
        assert raw[:4] == MAGIC
        assert struct.unpack("<I", raw[4:8])[0] == VERSION
        hlen = struct.unpack("<Q", raw[8:16])[0]
        assert 16 + hlen <= len(raw)
        parsed = json.loads(raw[16 : 16 + hlen].decode("utf-8"))

        # Required header sections
        assert parsed["model_config"] == model_config
        assert parsed["tokenizer"] == tokenizer
        assert parsed["quantization"]["group_size"] == GROUP_SIZE

        # Tensor table covers every tensor with name/dtype/shape/offset
        entries = parsed["tensors"]
        assert [e["name"] for e in entries] == list(tensors)
        prev_end = 16 + hlen
        for e in entries:
            assert set(e) >= {"name", "dtype", "shape", "offset"}
            arr = tensors[e["name"]]
            assert e["shape"] == list(arr.shape)
            assert e["dtype"] in DTYPES
            assert np.dtype(DTYPES[e["dtype"]][0]) == arr.dtype
            off = e["offset"]
            assert off % ALIGN == 0
            assert off >= prev_end
            # Padding is zero and data is at the stated offset
            assert raw[prev_end:off] == b"\x00" * (off - prev_end)
            nbytes = arr.nbytes
            assert raw[off : off + nbytes] == np.ascontiguousarray(arr).tobytes()
            prev_end = off + nbytes
            if e["name"].endswith(("embed_tokens.weight", "lm_head.weight")):
                assert e["dtype"] == "fp16"

    run()
