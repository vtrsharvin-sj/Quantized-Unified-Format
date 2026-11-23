"""A default load and generate prints nothing above DEBUG except the "Selected backend" line.

**Validates: Requirements 37.6**
"""
from __future__ import annotations

import logging

import pytest

from quf.runtime.generate import generate
from quf.runtime.kernels import fusion
from quf.runtime.model import QUFModel


class FakeTok:
    def encode(self, text):
        return [ord(c) % 32 for c in text]

    def decode(self, ids):
        return "".join(chr(97 + int(i) % 26) for i in ids)


def test_default_load_and_generate_is_quiet_on_mixed_int4_int8(tmp_path, caplog):
    from test_int4_greedy import _build

    path = tmp_path / "m.quf"
    _build(path, "tiny", 32, False, "mixed", seed=7)
    fusion.reset_fusion_log()
    with caplog.at_level(logging.DEBUG, logger="quf"):
        model = QUFModel.load(path)  # default device and backend
        model.header.tokenizer["eos_token_id"] = None
        out = generate(model, FakeTok(), [3, 5, 7], max_new_tokens=6)
    assert len(out) == 6
    loud = [
        r.getMessage() for r in caplog.records
        if r.name.startswith("quf") and r.levelno >= logging.INFO
    ]
    assert len(loud) == 1 and loud[0].startswith("Selected backend"), loud


def test_refused_fusion_logs_at_debug_with_group_name(caplog):
    from quf.runtime.linear import FusedProjection

    class M:
        def __init__(self, name):
            self.name = name

    mods = [M(f"model.layers.3.self_attn.{p}_proj") for p in "qkv"]
    assert FusedProjection._group_name(mods) == "model.layers.3.self_attn.qkv"
    mods = [M(f"model.layers.3.mlp.{p}_proj") for p in ("gate", "up")]
    assert FusedProjection._group_name(mods) == "model.layers.3.mlp.gate_up"
    fusion.reset_fusion_log()
    with caplog.at_level(logging.DEBUG, logger="quf"):
        fusion._log_once("k", "not fusing projection group %s: %s", "g", "why")
    recs = [r for r in caplog.records if "not fusing" in r.getMessage()]
    assert recs and all(r.levelno == logging.DEBUG for r in recs)


def test_info_names_clip_embedding_as_rtn_plus_clip_search():
    from quf.cli import _embedding_line

    spec = {"name": "e", "bits": 8, "group_size": 64, "method": "rtn", "clip_search": True}
    assert "(rtn + clip search)" in _embedding_line({"embedding": spec})
    spec = dict(spec, clip_search=False)
    assert "(rtn)" in _embedding_line({"embedding": spec})
