"""Property 9: Calibration shape, determinism, and shortfall.

**Validates: Requirements 6.4, 6.5, 6.6**
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from quf.tokenization.calibration import CalibrationShortfallError, build_calibration


class StubTokenizer:
    """Offline tokenizer: one token per character (its code point)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


texts_st = st.lists(st.text(max_size=60), max_size=15)


@given(
    texts=texts_st,
    n_samples=st.integers(1, 6),
    seq_len=st.integers(1, 20),
    seed=st.integers(0, 2**32 - 1),
)
def test_calibration_shape_determinism_shortfall(texts, n_samples, seq_len, seed):
    tok = StubTokenizer()
    available = sum(len(t) for t in texts)
    needed = n_samples * seq_len

    if available < needed:
        with pytest.raises(CalibrationShortfallError) as ei:
            build_calibration(texts, tok, n_samples, seq_len, seed)
        assert ei.value.needed == needed
        assert ei.value.available == available
        assert ei.value.shortfall == needed - available
        assert str(needed - available) in str(ei.value)
        return

    first = build_calibration(texts, tok, n_samples, seq_len, seed)
    second = build_calibration(list(texts), tok, n_samples, seq_len, seed)

    assert len(first) == n_samples
    assert all(len(w) == seq_len for w in first)
    assert first == second

    # Every token comes from the dataset.
    pool = {ord(c) for t in texts for c in t}
    assert all(i in pool for w in first for i in w)
