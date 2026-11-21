"""Property 31: Median timing is order-independent.

**Validates: Requirements 28.1**

CPU-only; time_repeats is driven by a fake run_once, so nothing waits on a real clock.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings, strategies as st

from quf.bench.timing import MIN_REPEATS, median, summarize, time_repeats

samples_st = st.lists(
    st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=MIN_REPEATS,
    max_size=40,
)


@settings(max_examples=100, deadline=None)
@given(samples=samples_st, seed=st.integers(0, 2**32 - 1))
def test_median_is_order_independent(samples, seed):
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    assert median(shuffled) == median(samples)
    a, b = summarize(samples), summarize(shuffled)
    assert (a.median, a.min, a.max, a.repeats) == (b.median, b.min, b.max, b.repeats)


@settings(max_examples=100, deadline=None)
@given(samples=samples_st)
def test_min_le_median_le_max(samples):
    s = summarize(samples)
    assert s.min <= s.median <= s.max
    assert s.min == min(samples) and s.max == max(samples)
    assert s.repeats == len(samples)


@settings(max_examples=50, deadline=None)
@given(repeats=st.integers(-3, MIN_REPEATS - 1))
def test_time_repeats_rejects_fewer_than_five(repeats):
    calls = []
    with pytest.raises(ValueError):
        time_repeats(lambda: calls.append(1) or 1.0, warmup=2, repeats=repeats)
    assert calls == []


@settings(max_examples=100, deadline=None)
@given(
    warmup=st.integers(0, 6),
    repeats=st.integers(MIN_REPEATS, 20),
    values=st.lists(st.floats(0.001, 1e6, allow_nan=False, allow_infinity=False), min_size=26, max_size=26),
)
def test_warmup_runs_excluded_from_samples(warmup, repeats, values):
    it = iter(values)
    calls = []

    def run_once():
        calls.append(1)
        return next(it)

    stats = time_repeats(run_once, warmup=warmup, repeats=repeats)
    assert len(calls) == warmup + repeats
    assert list(stats.samples) == values[warmup : warmup + repeats]
    assert stats.median == median(values[warmup : warmup + repeats])


def test_median_even_and_odd_examples():
    assert median([3.0, 1.0, 2.0, 5.0, 4.0]) == 3.0
    assert median([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == 3.5
