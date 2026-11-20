"""Unit tests for the Benchmark_Suite (Requirements 11.1, 11.2, 11.3, 11.7). No network."""

from __future__ import annotations

import json
import math
import types

import pytest
import torch

from quf.bench import suite
from quf.bench.suite import BenchResult, DatasetLoadError, run_bench

VOCAB = 16


class StubTok:
    def encode(self, text):
        return [ord(c) % VOCAB for c in text]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


class StubModel:
    """Uniform logits: perplexity equals the vocabulary size."""

    max_position_embeddings = 256
    device = "cpu"

    def __init__(self):
        self.header = types.SimpleNamespace(tokenizer={})

    def new_cache(self):
        return None

    def forward(self, ids, cache=None):
        return torch.zeros(1, ids.shape[1], VOCAB)


TEXT = "hello world, this is a benchmark text. " * 10


def _run(tmp_path, **kw):
    quf = tmp_path / "m.quf"
    quf.write_bytes(b"x" * 123)
    out = tmp_path / "sub" / "results.json"
    res = run_bench(
        quf,
        out_json=out,
        device="cpu",
        ctx_len=32,
        max_new_tokens=4,
        text=TEXT,
        model_loader=lambda p, backend: StubModel(),
        tok=StubTok(),
        **kw,
    )
    return res, out, quf


def test_reports_fields_and_writes_json(tmp_path):
    res, out, _ = _run(tmp_path)
    assert res.backend == "cpu"
    assert res.device_name == "cpu"
    assert res.file_size_bytes == 123
    assert res.perplexity == pytest.approx(VOCAB, rel=1e-5)
    assert res.tokens_per_second > 0
    assert res.fp16_perplexity is None and res.gguf_size_bytes is None

    data = json.loads(out.read_text(encoding="utf-8"))
    for key in ("backend", "device_name", "software_versions", "perplexity",
                "file_size_bytes", "tokens_per_second"):
        assert key in data
    assert "python" in data["software_versions"]
    assert BenchResult.from_json(out.read_text(encoding="utf-8")) == res


def test_perplexity_windows_and_short_text():
    tok, model = StubTok(), StubModel()
    assert suite.perplexity(model, tok, ctx_len=8, text="a" * 40, max_windows=2) == pytest.approx(VOCAB)
    # text shorter than ctx_len falls back to a single window
    assert suite.perplexity(model, tok, ctx_len=100, text="abcdef") == pytest.approx(VOCAB)
    with pytest.raises(ValueError):
        suite.perplexity(model, tok, ctx_len=8, text="a")


def test_gguf_size_reported(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"y" * 77)
    monkeypatch.setattr(suite, "_gguf_speed", lambda path, n=32: 12.5)
    res, out, _ = _run(tmp_path, gguf=str(gguf))
    assert res.gguf_size_bytes == 77
    assert res.gguf_tokens_per_second == 12.5
    assert json.loads(out.read_text(encoding="utf-8"))["gguf_size_bytes"] == 77


def test_fp16_reference_relative_increase(tmp_path, monkeypatch):
    class Ref(StubModel):
        def forward(self, ids, cache=None):
            # peaked logits on the true next token would be complex; use a
            # distinct uniform-scale model: all-equal logits again gives VOCAB.
            return torch.zeros(1, ids.shape[1], VOCAB)

    monkeypatch.setattr(suite, "_HFCausalLM", lambda path, device: Ref())
    res, _, _ = _run(tmp_path, fp16_ref="ref")
    assert res.fp16_perplexity == pytest.approx(VOCAB, rel=1e-5)
    assert res.relative_ppl_increase == pytest.approx(0.0, abs=1e-5)
    assert res.ppl_target_percent == 3.0


def test_dataset_load_failure_names_dataset(tmp_path, monkeypatch):
    def boom(dataset=suite.DEFAULT_DATASET, split="test"):
        raise DatasetLoadError(dataset, "offline")

    monkeypatch.setattr(suite, "load_wikitext", boom)
    called = []
    with pytest.raises(DatasetLoadError) as ei:
        run_bench(
            tmp_path / "m.quf",
            out_json=tmp_path / "r.json",
            device="cpu",
            dataset="my-dataset",
            model_loader=lambda p, b: called.append(1),
            tok=StubTok(),
        )
    assert "my-dataset" in str(ei.value)
    assert not called
    assert not (tmp_path / "r.json").exists()


def test_load_wikitext_wraps_errors(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "datasets":
            raise ImportError("no datasets")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DatasetLoadError) as ei:
        suite.load_wikitext("wikitext-2-raw-v1")
    assert "wikitext-2-raw-v1" in str(ei.value)
    assert math.isfinite(suite.PPL_TARGET_PERCENT)
