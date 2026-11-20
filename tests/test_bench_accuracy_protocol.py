"""Half-window (llama-perplexity) scoring helper in scripts/bench_accuracy.py."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "bench_accuracy", Path(__file__).resolve().parent.parent / "scripts" / "bench_accuracy.py")
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)


def test_score_start():
    assert ba.score_start(2048, "full") == 0
    assert ba.score_start(2048, "llama") == 1024
    with pytest.raises(ValueError):
        ba.score_start(2048, "nope")


def test_half_window_counts_second_half_only():
    n, v = 16, 5
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(n, v)).astype(np.float32)
    ids = rng.integers(0, v, size=n).tolist()
    _, full_cnt = ba.window_nll(logits, ids, "full")
    nll, cnt = ba.window_nll(logits, ids, "llama")
    assert full_cnt == n - 1
    assert cnt == n - 1 - n // 2  # positions n//2 .. n-2, as llama-perplexity
    lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    expect = -sum(lp[j, ids[j + 1]] for j in range(n // 2, n - 1))
    assert math.isclose(nll, float(expect), rel_tol=1e-5)


def test_uniform_logits_give_vocab_size_perplexity():
    ids = list(range(8))
    nll, cnt = ba.window_nll(np.zeros((8, 8), dtype=np.float32), ids, "llama")
    assert math.isclose(math.exp(nll / cnt), 8.0, rel_tol=1e-5)


def test_window_ids_subset_and_pin():
    assert ba.window_ids(list(range(10)), 0, ctx=4) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert len(ba.window_ids(list(range(100)), 3, ctx=4)) == 3
    assert "/b6100/" in ba.CONVERT_SCRIPT_URL
