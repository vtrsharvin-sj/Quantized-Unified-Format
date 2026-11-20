"""Property test for the generation context length check.

Feature: quf-quantized-format, Property 13: Context length check
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from hypothesis import given, strategies as st

from quf.runtime.generate import ContextLengthError, generate


class _StubModel:
    """Minimal model: the length check runs before any forward pass.

    When generation is accepted, ``forward`` returns logits whose argmax is the
    EOS token, so generation ends after a single step.
    """

    device = "cpu"

    def __init__(self, context_len: int):
        self.max_position_embeddings = context_len
        self.header = SimpleNamespace(tokenizer={"eos_token_id": 0})

    def new_cache(self):
        return None

    def forward(self, ids, cache):
        logits = torch.zeros(1, ids.shape[1], 4)
        logits[..., 0] = 1.0
        return logits


@given(
    context_len=st.integers(min_value=1, max_value=5000),
    prompt_len=st.integers(min_value=0, max_value=6000),
    max_new=st.integers(min_value=0, max_value=6000),
)
def test_context_length_check(context_len, prompt_len, max_new):
    """**Validates: Requirements 8.8**"""
    model = _StubModel(context_len)
    prompt = [1] * prompt_len
    if prompt_len + max_new > context_len:
        with pytest.raises(ContextLengthError) as ei:
            generate(model, None, prompt, max_new_tokens=max_new)
        msg = str(ei.value)
        assert str(prompt_len) in msg and str(max_new) in msg
        assert str(context_len) in msg
        assert (ei.value.prompt_len, ei.value.max_new) == (prompt_len, max_new)
    else:
        generate(model, None, prompt, max_new_tokens=max_new)  # must not raise
