"""Property 8: Tokenizer round trip."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

tokenizers = pytest.importorskip("tokenizers")

from quf.tokenization.wrapper import TokenizerWrapper  # noqa: E402


def _build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>"],
        show_progress=False,
    )
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "quantized unified format stores 4-bit weights",
        "hello world, hello tokenizer! 123 456",
    ]
    tok.train_from_iterator(corpus, trainer)
    return tok


@pytest.fixture(scope="module")
def wrapper():
    return TokenizerWrapper(hf_tokenizer=_build_tokenizer())


# **Validates: Requirements 6.3**
@given(text=st.text())
def test_tokenizer_round_trip(wrapper, text):
    ids = wrapper.encode(text)
    assert all(isinstance(i, int) for i in ids)
    assert wrapper.decode(ids) == text


def test_tokenizer_round_trip_examples(wrapper):
    for text in ["", "hello world", "  leading and trailing  ", "naïve café 日本語 🙂", "a\nb\tc"]:
        assert wrapper.decode(wrapper.encode(text)) == text
