"""Property test for sampling determinism and greedy decoding (CPU)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from hypothesis import given, strategies as st

from quf.runtime.generate import generate
from quf.runtime.model import QUFModel
from test_model import VOCAB, build_tiny_quf


class _Tok:
    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


@pytest.fixture(scope="module")
def model():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tiny.quf"
        build_tiny_quf(path)
        yield QUFModel.load(path, "cpu")


prompts = st.lists(st.integers(1, VOCAB - 1), min_size=1, max_size=8)


@given(
    prompt=prompts,
    max_new=st.integers(1, 6),
    temperature=st.floats(0.1, 2.0),
    top_k=st.one_of(st.none(), st.integers(1, VOCAB + 4)),
    top_p=st.one_of(st.none(), st.floats(0.05, 1.0)),
    seed=st.integers(0, 2**31 - 1),
)
def test_same_seed_same_tokens(model, prompt, max_new, temperature, top_k, top_p, seed):
    """**Validates: Requirements 8.6, 8.7**"""
    kw = dict(max_new_tokens=max_new, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)
    a = generate(model, _Tok(), prompt, **kw)
    b = generate(model, _Tok(), prompt, **kw)
    assert a == b


@given(prompt=prompts, max_new=st.integers(1, 6), seed=st.integers(0, 1000))
def test_temperature_zero_is_argmax(model, prompt, max_new, seed):
    """**Validates: Requirements 8.6, 8.7**"""
    out = generate(model, _Tok(), prompt, max_new_tokens=max_new, temperature=0.0, seed=seed)
    assert 1 <= len(out) <= max_new
    seq = list(prompt)
    for tok_id in out:
        logits = model.forward(torch.tensor([seq]))[0, -1].float()
        # Cached vs. uncached numerics may differ slightly; allow near-ties.
        assert logits[tok_id] >= logits.max() - 1e-2
        seq.append(tok_id)
