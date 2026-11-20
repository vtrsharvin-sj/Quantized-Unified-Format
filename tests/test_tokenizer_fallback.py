"""Unit tests for the TokenizerWrapper gigatoken -> Hugging Face fallback (Req 6.2)."""

from __future__ import annotations

import builtins
import logging
import sys
import types

import pytest

tokenizers = pytest.importorskip("tokenizers")

from quf.tokenization.wrapper import TokenizerWrapper  # noqa: E402


def _tiny_hf_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def _block_gigatoken(monkeypatch):
    monkeypatch.delitem(sys.modules, "gigatoken", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gigatoken" or name.startswith("gigatoken."):
            raise ImportError("simulated missing gigatoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_falls_back_to_hf_and_logs_reason_when_gigatoken_import_fails(monkeypatch, caplog):
    _block_gigatoken(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="quf.tokenization"):
        wrapper = TokenizerWrapper(hf_tokenizer=_tiny_hf_tokenizer())

    assert wrapper.backend == "huggingface"
    assert wrapper.fallback_reason is not None
    assert "simulated missing gigatoken" in wrapper.fallback_reason
    messages = [r.getMessage() for r in caplog.records if r.name == "quf.tokenization"]
    assert any("Falling back" in m and "simulated missing gigatoken" in m for m in messages)


def test_fallback_encodes_and_decodes_with_hf(monkeypatch):
    _block_gigatoken(monkeypatch)
    wrapper = TokenizerWrapper(hf_tokenizer=_tiny_hf_tokenizer())

    ids = wrapper.encode("hello world")
    assert ids == [1, 2]
    assert wrapper.decode(ids) == "hello world"


def test_falls_back_when_gigatoken_cannot_wrap_tokenizer(monkeypatch, caplog):
    fake = types.ModuleType("gigatoken")

    class Tokenizer:
        def __init__(self, *_a, **_k):
            raise RuntimeError("unsupported tokenizer")

    fake.Tokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "gigatoken", fake)

    with caplog.at_level(logging.DEBUG, logger="quf.tokenization"):
        wrapper = TokenizerWrapper(hf_tokenizer=_tiny_hf_tokenizer())

    assert wrapper.backend == "huggingface"
    assert "unsupported tokenizer" in wrapper.fallback_reason
    assert any("unsupported tokenizer" in r.getMessage() for r in caplog.records)
    assert wrapper.encode("hello") == [1]
