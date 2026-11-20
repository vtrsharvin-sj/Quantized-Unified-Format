"""Property test for KV_Cache equivalence (CPU)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quf.runtime.model import KVCache, QUFModel
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
