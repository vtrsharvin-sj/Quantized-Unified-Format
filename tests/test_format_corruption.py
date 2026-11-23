"""Property 3: Reader rejects corrupted files.

**Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from quf.format.constants import ALIGN, PREFIX_SIZE, align_up
from quf.format.errors import (
    HeaderSchemaError,
    InvalidMagicError,
    TensorBoundsError,
    TruncatedHeaderError,
    UnsupportedVersionError,
)
from quf.format.reader import QUFReader
from quf.format.schema import QUFHeader
from quf.format.writer import write_quf

_names = st.lists(
    st.text(alphabet="abcdefghij._", min_size=1, max_size=12).map(lambda s: "t" + s),
    min_size=1,
    max_size=4,
    unique=True,
)


@st.composite
def valid_files(draw):
    names = draw(_names)
    tensors = {}
    for n in names:
        rows = draw(st.integers(1, 5))
        cols = draw(st.integers(1, 40))
        dtype = draw(st.sampled_from([np.float16, np.float32, np.int32]))
        tensors[n] = np.ones((rows, cols), dtype=dtype)
    return tensors


def _header() -> QUFHeader:
    return QUFHeader(
        model_config={"hidden_size": 8},
        tokenizer={"type": "x"},
        quantization={"group_size": 128, "symmetric": True},
    )


def _build(tmp_path, tensors) -> bytes:
    p = tmp_path / "ok.quf"
    write_quf(p, _header(), tensors)
    return p.read_bytes()


def _split(data: bytes):
    (hlen,) = struct.unpack("<Q", data[8:16])
    header = json.loads(data[PREFIX_SIZE : PREFIX_SIZE + hlen].decode("utf-8"))
    return header, data[PREFIX_SIZE + hlen :]


def _with_header_bytes(data: bytes, header_bytes: bytes) -> bytes:
    (hlen,) = struct.unpack("<Q", data[8:16])
    return (
        data[:8]
        + struct.pack("<Q", len(header_bytes))
        + header_bytes
        + data[PREFIX_SIZE + hlen :]
    )


def _with_header(data: bytes, header: dict) -> bytes:
    return _with_header_bytes(data, json.dumps(header, separators=(",", ":")).encode())


def _open(tmp_path, data: bytes):
    p = tmp_path / "bad.quf"
    p.write_bytes(data)
    QUFReader(p).close()


def _sanity(tmp_path, data):
    p = tmp_path / "sane.quf"
    p.write_bytes(data)
    QUFReader(p).close()


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("corrupt")


@given(tensors=valid_files(), magic=st.binary(min_size=4, max_size=4).filter(lambda b: b != b"QUF1"))
def test_wrong_magic(tmp, tensors, magic):
    data = _build(tmp, tensors)
    _sanity(tmp, data)
    with pytest.raises(InvalidMagicError):
        _open(tmp, magic + data[4:])


@given(tensors=valid_files(), version=st.integers(4, 2**32 - 1))
def test_unsupported_version(tmp, tensors, version):
    data = _build(tmp, tensors)
    with pytest.raises(UnsupportedVersionError):
        _open(tmp, data[:4] + struct.pack("<I", version) + data[8:])


@given(tensors=valid_files(), extra=st.integers(1, 2**40))
def test_header_length_past_eof(tmp, tensors, extra):
    data = _build(tmp, tensors)
    bad_len = len(data) - PREFIX_SIZE + extra
    with pytest.raises(TruncatedHeaderError):
        _open(tmp, data[:8] + struct.pack("<Q", bad_len) + data[16:])


@given(tensors=valid_files(), idx=st.integers(0, 3), delta=st.integers(1, ALIGN - 1))
def test_misaligned_offset_names_tensor(tmp, tensors, idx, delta):
    data = _build(tmp, tensors)
    header, _ = _split(data)
    entry = header["tensors"][idx % len(header["tensors"])]
    entry["offset"] += delta
    with pytest.raises(TensorBoundsError) as ei:
        _open(tmp, _with_header(data, header))
    assert entry["name"] in str(ei.value)


@given(tensors=valid_files(), idx=st.integers(0, 3), slack=st.integers(0, 10))
def test_tensor_past_eof_names_tensor(tmp, tensors, idx, slack):
    data = _build(tmp, tensors)
    header, _ = _split(data)
    entry = header["tensors"][idx % len(header["tensors"])]
    itemsize = {"fp16": 2, "fp32": 4, "int32": 4}[entry["dtype"]]
    nbytes = itemsize * int(np.prod(entry["shape"]))
    entry["offset"] = align_up(len(data) - nbytes + 1) + ALIGN * slack
    assert entry["offset"] % ALIGN == 0
    with pytest.raises(TensorBoundsError) as ei:
        _open(tmp, _with_header(data, header))
    assert entry["name"] in str(ei.value)


@given(
    tensors=valid_files(),
    garbage=st.one_of(
        st.just(b"{not json"),
        st.just(b"\xff\xfe\xfa"),
        st.just(b""),
        st.text(alphabet="abc{[", min_size=1, max_size=20).map(str.encode),
    ),
)
def test_invalid_json(tmp, tensors, garbage):
    data = _build(tmp, tensors)
    with pytest.raises(HeaderSchemaError):
        _open(tmp, _with_header_bytes(data, garbage))


@given(
    tensors=valid_files(),
    field=st.sampled_from(
        ["model_config", "tokenizer", "quantization", "tensors", "t.name", "t.dtype", "t.shape", "t.offset"]
    ),
)
def test_missing_required_field(tmp, tensors, field):
    data = _build(tmp, tensors)
    header, _ = _split(data)
    if field.startswith("t."):
        del header["tensors"][0][field[2:]]
    else:
        del header[field]
    with pytest.raises(HeaderSchemaError):
        _open(tmp, _with_header(data, header))
