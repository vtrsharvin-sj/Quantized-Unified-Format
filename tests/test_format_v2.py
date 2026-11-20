"""Format version 2: group size, g_idx, int8 and fp16 modules, and version 1 compatibility.

Feature: quf-quantized-format, Property 23: Header group size and version round trip
Feature: quf-quantized-format, Property 24: Group-size and g_idx validation
**Validates: Requirements 19.1, 19.2, 19.3, 19.4, 19.5, 19.7, 20.2, 20.3, 20.6, 22.1**
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from quf.format.errors import HeaderSchemaError
from quf.format.reader import QUFReader
from quf.format.schema import QUFHeader
from quf.format.writer import write_quf

FIXTURE = Path(__file__).parent / "fixtures" / "quf_0_1_0.quf"


def _version(path) -> int:
    return struct.unpack("<I", Path(path).read_bytes()[4:8])[0]


def _header(**quant) -> QUFHeader:
    q = {"bits": 4, "symmetric": True, **quant}
    return QUFHeader({"hidden_size": 8}, {"model_id": "t"}, q)


@st.composite
def _model(draw):
    group_size = draw(st.sampled_from([32, 64, 128]))
    in_features = group_size * draw(st.integers(1, 3))
    out = 8 * draw(st.integers(1, 2))
    act_order = draw(st.booleans())
    tensors: dict[str, np.ndarray] = {}
    modules: dict[str, str] = {}
    for i in range(draw(st.integers(1, 3))):
        name = f"layers.{i}.proj"
        prec = draw(st.sampled_from(["int4", "int8", "fp16"]))
        modules[name] = prec
        if prec == "int4":
            tensors[name + ".qweight"] = np.arange(in_features // 8 * out, dtype=np.int32).reshape(
                in_features // 8, out
            )
            tensors[name + ".scales"] = np.ones((in_features // group_size, out), np.float16)
            if act_order:
                g = draw(
                    st.lists(
                        st.integers(0, in_features // group_size - 1),
                        min_size=in_features,
                        max_size=in_features,
                    )
                )
                tensors[name + ".g_idx"] = np.array(g, dtype=np.int32)
        elif prec == "int8":
            tensors[name + ".qweight_int8"] = np.full((out, in_features), -3, np.int8)
            tensors[name + ".scales"] = np.ones((out,), np.float16)
        else:
            tensors[name + ".weight"] = np.ones((out, in_features), np.float16)
    use_modules = draw(st.booleans())
    return group_size, act_order, (modules if use_modules else {}), tensors


@settings(max_examples=100, deadline=None)
@given(_model())
def test_property_23_round_trip_and_version(tmp_path_factory, m):
    group_size, act_order, modules, tensors = m
    quant = {"group_size": group_size}
    if act_order:
        quant["act_order"] = True
    if modules:
        quant["modules"] = modules
    path = tmp_path_factory.mktemp("v2") / "m.quf"
    write_quf(path, _header(**quant), tensors)

    has_g_idx = any(n.endswith(".g_idx") for n in tensors)
    has_int8_fp16 = any(p in ("int8", "fp16") for p in modules.values())
    has_int8_tensor = any(n.endswith(".qweight_int8") for n in tensors)
    expect = 1 if group_size == 128 and not (has_g_idx or has_int8_fp16 or has_int8_tensor) else 2
    assert _version(path) == expect

    with QUFReader(path) as r:
        assert r.version == expect
        assert r.header.group_size == group_size
        assert r.header.modules == modules
        assert r.tensor_names() == list(tensors)
        for n, arr in tensors.items():
            got = r.tensor(n)
            assert got.dtype == arr.dtype and got.shape == arr.shape
            assert got.tobytes() == arr.tobytes()
            del got


def test_group_size_128_without_extras_is_version_1(tmp_path):
    p = tmp_path / "a.quf"
    write_quf(p, _header(group_size=128), {"x": np.zeros(3, np.float16)})
    assert _version(p) == 1


@pytest.mark.parametrize("gs", [32, 64])
def test_other_group_size_is_version_2(tmp_path, gs):
    p = tmp_path / "a.quf"
    write_quf(p, _header(group_size=gs), {"x": np.zeros(3, np.float16)})
    assert _version(p) == 2


@given(bad=st.integers(-10, 300).filter(lambda g: g not in (32, 64, 128)))
@settings(deadline=None)
def test_property_24_bad_group_size(tmp_path_factory, bad):
    d = tmp_path_factory.mktemp("gs")
    with pytest.raises(ValueError) as e:
        write_quf(d / "w.quf", _header(group_size=bad), {"x": np.zeros(3, np.float16)})
    assert str(bad) in str(e.value) and "32, 64, 128" in str(e.value)

    # Reader: patch a valid file's header to carry the bad group size.
    good = d / "g.quf"
    write_quf(good, _header(group_size=128), {"x": np.zeros(3, np.float16)})
    data = good.read_bytes()
    (hlen,) = struct.unpack("<Q", data[8:16])
    hdr = json.loads(data[16 : 16 + hlen])
    hdr["quantization"]["group_size"] = bad
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    bad_file = d / "b.quf"
    bad_file.write_bytes(data[:8] + struct.pack("<Q", len(hb)) + hb + data[16 + hlen :])
    with pytest.raises(HeaderSchemaError) as e2:
        QUFReader(bad_file)
    assert str(bad) in str(e2.value) and "32, 64, 128" in str(e2.value)


def _g_idx_file(path, g_idx, in_features=64, group_size=32):
    tensors = {
        "m.qweight": np.zeros((in_features // 8, 8), np.int32),
        "m.scales": np.ones((in_features // group_size, 8), np.float16),
        "m.g_idx": np.asarray(g_idx, dtype=np.int32),
    }
    write_quf(path, _header(group_size=group_size, act_order=True), tensors)


@pytest.mark.parametrize(
    "g_idx",
    [[0] * 63 + [2], [0] * 63 + [-1], [0] * 63, [0] * 65],
    ids=["too-high", "negative", "too-short", "too-long"],
)
def test_property_24_bad_g_idx_rejected_naming_tensor(tmp_path, g_idx):
    with pytest.raises(ValueError, match="m.g_idx"):
        _g_idx_file(tmp_path / "w.quf", g_idx)


def test_reader_rejects_bad_g_idx_naming_tensor(tmp_path):
    p = tmp_path / "ok.quf"
    _g_idx_file(p, [0, 1] * 32)
    data = bytearray(p.read_bytes())
    with QUFReader(p) as r:
        entry = r._entries["m.g_idx"]
        off = entry.offset
    del r
    data[off : off + 4] = struct.pack("<i", 7)  # group count is 2
    q = tmp_path / "bad.quf"
    q.write_bytes(bytes(data))
    with pytest.raises(HeaderSchemaError, match="m.g_idx"):
        QUFReader(q)


def test_invalid_module_precision_rejected(tmp_path):
    with pytest.raises(ValueError, match="int4, int8, fp16"):
        write_quf(
            tmp_path / "a.quf",
            _header(group_size=128, modules={"m": "int3"}),
            {"x": np.zeros(1, np.float16)},
        )


def test_int8_module_missing_tensor_rejected(tmp_path):
    with pytest.raises(ValueError, match="qweight_int8"):
        write_quf(
            tmp_path / "a.quf",
            _header(group_size=128, modules={"m": "int8"}),
            {"x": np.zeros(1, np.float16)},
        )


def test_reads_0_1_0_fixture_as_group_size_128():
    with QUFReader(FIXTURE) as r:
        assert r.version == 1
        assert r.header.group_size == 128
        assert r.header.modules == {}
        assert r.tensor("l.qweight").shape == (16, 4)
        assert r.tensor("l.scales").dtype == np.float16


def test_missing_group_size_is_read_as_128(tmp_path):
    data = FIXTURE.read_bytes()
    (hlen,) = struct.unpack("<Q", data[8:16])
    hdr = json.loads(data[16 : 16 + hlen])
    del hdr["quantization"]["group_size"]
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    # Keep offsets valid: pad the new header to the old length with spaces.
    assert len(hb) <= hlen
    hb = hb + b" " * (hlen - len(hb))
    p = tmp_path / "nogs.quf"
    p.write_bytes(data[:16] + hb + data[16 + hlen :])
    with QUFReader(p) as r:
        assert r.header.group_size == 128
