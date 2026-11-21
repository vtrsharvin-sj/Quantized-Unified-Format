"""Property test for KV_Cache equivalence (CPU)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quf.runtime.generate import generate
from quf.runtime.model import CacheLengthError, KVCache, QUFModel
from test_model import VOCAB, build_tiny_quf


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("kv") / "tiny.quf"
    build_tiny_quf(path)
    return QUFModel.load(path, "cpu")


@st.composite
def prompt_and_continuation(draw):
    prompt = draw(st.lists(st.integers(0, VOCAB - 1), min_size=1, max_size=8))
    cont = draw(st.lists(st.integers(0, VOCAB - 1), min_size=1, max_size=6))
    return prompt, cont


# Feature: quf-quantized-format, Property 11: KV_Cache equivalence
@settings(max_examples=30, deadline=None)
@given(data=prompt_and_continuation())
def test_kv_cache_equivalence(tiny_model, data):
    """**Validates: Requirements 8.3, 8.10**"""
    prompt, cont = data
    ids = torch.tensor([prompt + cont])
    n = len(prompt)

    with torch.no_grad():
        full = tiny_model.forward(ids)
        cache = KVCache(tiny_model.num_layers)
        parts = [tiny_model.forward(ids[:, :n], cache)]
        for i in range(n, ids.shape[1]):
            parts.append(tiny_model.forward(ids[:, i : i + 1], cache))
    inc = torch.cat(parts, dim=1)

    assert cache.length == ids.shape[1]
    assert inc.shape == full.shape
    assert torch.allclose(inc.float(), full.float(), atol=1e-2)


# Feature: quf-quantized-format, Static_KV_Cache and fixed-shape Decode_Step
@settings(max_examples=30, deadline=None)
@given(data=prompt_and_continuation())
def test_static_decode_matches_dynamic(tiny_model, data):
    """**Validates: Requirements 31.1, 31.2, 31.10**"""
    prompt, cont = data
    ids = torch.tensor([prompt + cont])
    full = tiny_model.forward(ids)
    cache = tiny_model.new_static_cache(ids.shape[1])
    parts = [tiny_model.forward(ids[:, : len(prompt)], cache)]
    for tok in cont[:-1]:
        parts.append(tiny_model.decode_token(tok, cache))
    inc = torch.cat(parts, dim=1)
    assert (inc - full[:, : inc.shape[1]]).abs().max().item() <= 1e-2
    assert cache.length == inc.shape[1]


def test_static_cache_overflow_names_both_lengths(tiny_model):
    cache = tiny_model.new_static_cache(4)
    tiny_model.forward(torch.tensor([[1, 2, 3, 4]]), cache)
    with pytest.raises(CacheLengthError) as ei:
        tiny_model.decode_token(5, cache)
    assert "5" in str(ei.value) and "4" in str(ei.value)


def test_generate_static_equals_legacy(tiny_model):
    for prompt in ([1, 2, 3], [5], [7, 7, 7, 7, 2]):
        a = generate(tiny_model, None, prompt, max_new_tokens=12, static_cache=False)
        b = generate(tiny_model, None, prompt, max_new_tokens=12)
        assert a == b
