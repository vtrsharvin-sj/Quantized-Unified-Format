"""Format version 3: Quantized_Embedding round trip, version rule, validation, row dequant.

Feature: quf-quantized-format, Property 41: Quantized embedding round trip and row dequantization
**Validates: Requirements 38.1, 38.2, 38.3, 38.4, 38.5, 38.11**
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

import quf.format.reader as reader_mod
from quf.format.errors import HeaderSchemaError, UnsupportedVersionError
from quf.format.reader import QUFReader
from quf.format.schema import QUFHeader
from quf.format.writer import write_quf
from quf.quant.embedding import (
    dequantize_embedding,
    dequantize_embedding_rows,
    quantize_embedding_rtn,
)

FIXTURE = Path(__file__).parent / "fixtures" / "quf_0_1_0.quf"
NAME = "model.embed_tokens"


def _version(path) -> int:
    return struct.unpack("<I", Path(path).read_bytes()[4:8])[0]


def _suffix(bits):
    return ".embed_q4" if bits == 4 else ".embed_q8"


def _embed_file(path, rows=5, cols=64, bits=4, gs=32, extra_quant=None):
    w = torch.randn(rows, cols, generator=torch.Generator().manual_seed(0))
    qe = quantize_embedding_rtn(w, bits, gs)
    spec = {"name": NAME, "bits": bits, "group_size": gs, "method": "rtn"}
    q = {"bits": 4, "symmetric": True, "group_size": 128, "embedding": spec, **(extra_quant or {})}
    tensors = {NAME + _suffix(bits): qe.packed, NAME + ".embed_scales": qe.scales}
    write_quf(path, QUFHeader({"hidden_size": cols}, {"model_id": "t"}, q), tensors)
    return qe


@settings(max_examples=60, deadline=None)
@given(
    rows=st.integers(1, 9),
    groups=st.integers(1, 3),
    bits=st.sampled_from([4, 8]),
    gs=st.sampled_from([32, 64]),
    seed=st.integers(0, 2**16),
)
def test_round_trip_and_row_gather(tmp_path_factory, rows, groups, bits, gs, seed):
    cols = gs * groups
    w = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)) * 3
    qe = quantize_embedding_rtn(w, bits, gs)
    path = tmp_path_factory.mktemp("emb") / "m.quf"
    spec = {"name": NAME, "bits": bits, "group_size": gs, "method": "rtn"}
    hdr = QUFHeader({}, {}, {"bits": 4, "symmetric": True, "group_size": 128, "embedding": spec})
    write_quf(path, hdr, {NAME + _suffix(bits): qe.packed, NAME + ".embed_scales": qe.scales})
    assert _version(path) == 3
    with QUFReader(path) as r:
        assert r.version == 3
        packed, scales = r.tensor(NAME + _suffix(bits)), r.tensor(NAME + ".embed_scales")
        assert np.array_equal(packed, qe.packed) and np.array_equal(scales, qe.scales)
        assert packed.shape == qe.packed.shape and scales.shape == qe.scales.shape
        ids = np.array([rows - 1, 0, rows - 1])
        full = dequantize_embedding(qe)
        got = dequantize_embedding_rows(packed, scales, bits, gs, ids)
        assert torch.equal(got, full[torch.as_tensor(ids)])
        # Per-element error bound, with s the stored (fp16-rounded) group scale and qmax 7 / 127:
        #  * rounding to the grid: s / 2;
        #  * the scale is fp16(amax / qmax), so amax / s can exceed qmax by a relative 2**-11 and
        #    the clamp to qmax then costs up to qmax * s * 2**-11;
        #  * the fp16 output rounds a value of magnitude <= qmax * s * (1 + 2**-11) by a relative
        #    2**-11 (half an ulp), i.e. at most qmax * s * 2**-11 * 1.001.
        # The last two never exceed 2 * qmax * s * 2**-11 together. 1e-6 covers fp32 arithmetic.
        qmax = 7 if bits == 4 else 127
        s = np.repeat(qe.scales.astype(np.float32), gs, axis=1)
        bound = s * (0.5 + 2 * qmax * 2.0**-11) + 1e-6
        err = (full.float() - w).abs().numpy()
        assert (err <= bound).all(), float((err - bound).max())


def test_version_selection(tmp_path):
    p = tmp_path / "v3.quf"
    _embed_file(p)
    assert _version(p) == 3
    # v1 and v2 files are written unchanged
    q = {"bits": 4, "symmetric": True, "group_size": 128}
    t = {"model.embed_tokens.weight": np.ones((2, 8), np.float16)}
    write_quf(tmp_path / "v1.quf", QUFHeader({}, {}, q), t)
    assert _version(tmp_path / "v1.quf") == 1
    write_quf(tmp_path / "v2.quf", QUFHeader({}, {}, {**q, "group_size": 64}), t)
    assert _version(tmp_path / "v2.quf") == 2
    for n in ("v1.quf", "v2.quf"):
        with QUFReader(tmp_path / n) as r:
            assert r.version in (1, 2)
    # the 0.1.0 fixture still loads
    with QUFReader(FIXTURE) as r:
        assert r.version == 1


def test_old_reader_rejects_v3(tmp_path, monkeypatch):
    p = tmp_path / "v3.quf"
    _embed_file(p)
    monkeypatch.setattr(reader_mod, "MAX_VERSION", 2)  # a 0.3.0 reader
    with pytest.raises(UnsupportedVersionError):
        QUFReader(p)


def test_v2_file_cannot_declare_embedding(tmp_path):
    p = tmp_path / "v3.quf"
    _embed_file(p)
    data = bytearray(p.read_bytes())
    data[4:8] = struct.pack("<I", 2)
    p.write_bytes(bytes(data))
    with pytest.raises(HeaderSchemaError, match="version 3"):
        QUFReader(p)


def _write(tmp_path, spec_over=None, q=None, s=None, bits=4, cols=64, gs=32):
    spec = {"name": NAME, "bits": bits, "group_size": gs, "method": "rtn", **(spec_over or {})}
    if q is None:
        q = np.zeros((3, cols // 2 if bits == 4 else cols), np.int8)
    if s is None:
        s = np.ones((3, cols // gs), np.float16)
    hdr = QUFHeader({}, {}, {"bits": 4, "symmetric": True, "group_size": 128, "embedding": spec})
    write_quf(tmp_path / "x.quf", hdr, {NAME + _suffix(bits): q, NAME + ".embed_scales": s})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spec_over": {"group_size": 128}},  # unsupported group size
        {"spec_over": {"group_size": 16}},
        {"spec_over": {"bits": 3}},
        {"spec_over": {"method": "awq"}},
        {"cols": 48},  # hidden size not a multiple of the group size
        {"s": np.ones((3, 3), np.float16)},  # wrong scale shape
        {"s": np.ones((4, 2), np.float16)},  # wrong row count in scales
        {"s": np.ones((3, 2), np.float32)},  # wrong scale dtype
        {"q": np.zeros((3, 64), np.float16)},  # wrong packed dtype
        {"q": np.zeros((64,), np.int8)},  # not 2-D
    ],
)
def test_bad_embedding_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        _write(tmp_path, **kwargs)


def test_missing_scales_and_orphan_tensor_rejected(tmp_path):
    spec = {"name": NAME, "bits": 8, "group_size": 64, "method": "rtn"}
    hdr = QUFHeader({}, {}, {"bits": 4, "symmetric": True, "group_size": 128, "embedding": spec})
    with pytest.raises(ValueError, match="missing"):
        write_quf(tmp_path / "a.quf", hdr, {NAME + ".embed_q8": np.zeros((2, 64), np.int8)})
    plain = QUFHeader({}, {}, {"bits": 4, "symmetric": True, "group_size": 128})
    with pytest.raises(ValueError, match="no matching"):
        write_quf(tmp_path / "b.quf", plain, {NAME + ".embed_q8": np.zeros((2, 64), np.int8)})


def test_reference_row_dequant_values():
    # int4: byte 0x9A -> low nibble 10 (col 0), high nibble 9 (col 1); zero point 8
    packed = np.zeros((1, 16), np.int8)
    packed[0, 0] = np.uint8(0x9A).view(np.int8)
    scales = np.full((1, 1), 0.5, np.float16)
    row = dequantize_embedding_rows(packed, scales, 4, 32, [0])
    assert row.dtype == torch.float16 and row.shape == (1, 32)
    assert row[0, 0].item() == 1.0 and row[0, 1].item() == 0.5 and row[0, 2].item() == -4.0
    # int8
    q8 = np.zeros((2, 32), np.int8)
    q8[1, 3] = -100
    out = dequantize_embedding_rows(q8, np.full((2, 1), 0.25, np.float16), 8, 32, [1])
    assert out[0, 3].item() == -25.0
