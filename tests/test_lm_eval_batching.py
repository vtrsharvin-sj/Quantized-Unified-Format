"""Batched log-likelihood scoring (no network, no lm_eval needed)."""

import torch
from hypothesis import given, settings, strategies as st

from quf.bench.lm_eval_adapter import score_batched

V = 11
_g = torch.Generator().manual_seed(0)
E = torch.randn(V, V, generator=_g)
W = torch.randn(V, V, generator=_g)


class Calls:
    def __init__(self):
        self.shapes = []

    def __call__(self, ids):
        self.shapes.append(tuple(ids.shape))
        # Causal toy model: logits at t depend on the whole prefix up to t.
        h = torch.cumsum(E[ids], dim=1)
        return torch.tanh(h) @ W


def _single(items, max_length=64):
    f = Calls()
    return [score_batched(f, [it], max_length, batch_size=1)[0] for it in items]


tok = st.integers(0, V - 1)
ctxs = st.lists(tok, min_size=0, max_size=6)
conts = st.lists(tok, min_size=0, max_size=4)


def _close(a, b):
    assert len(a) == len(b)
    for (la, ga), (lb, gb) in zip(a, b):
        assert abs(la - lb) < 1e-3
        assert ga == gb


# Feature: quf-quantized-format, Property 30: Batched log-likelihood equals single-request log-likelihood
# **Validates: Requirements 27.1, 27.2, 27.3**
@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.tuples(ctxs, conts), min_size=1, max_size=8),
    st.integers(1, 5),
    st.integers(1, 40),
    st.integers(4, 20),
)
def test_batched_matches_single(items, batch_size, budget, max_length):
    # Repeat some contexts to exercise sharing.
    items = items + [(items[0][0], c) for _, c in items]
    got = score_batched(Calls(), items, max_length, batch_size=batch_size, max_batch_tokens=budget)
    _close(got, _single(items, max_length))


def test_repeated_contexts_run_once():
    ctx = [1, 2, 3, 4]
    items = [(ctx, [5]), (ctx, [5]), (ctx, [5, 6]), (ctx, [5, 6, 7])]
    f = Calls()
    got = score_batched(f, items, 64)
    assert f.shapes == [(1, 6)]  # one row hosts all four requests
    _close(got, _single(items))


def test_distinct_choices_share_one_batched_pass():
    ctx = [1, 2, 3]
    items = [(ctx, [4]), (ctx, [5, 6]), (ctx, [7, 8, 9])]
    f = Calls()
    got = score_batched(f, items, 64)
    # [4] scores from the shared context rows of the longer inputs: 2 rows, not 3.
    assert len(f.shapes) == 1 and f.shapes[0][0] == 2
    _close(got, _single(items))


def test_empty_continuation_and_empty_context():
    got = score_batched(Calls(), [([1, 2], []), ([], [3])], 16, eos_id=2)
    assert got[0] == (0.0, True)
    _close([got[1]], _single([([2], [3])]))


# --- Real tiny QUFModel variant (CPU) -------------------------------------

import pytest  # noqa: E402

from quf.runtime.model import QUFModel  # noqa: E402
from test_model import VOCAB, build_tiny_quf  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("quf") / "tiny.quf"
    build_tiny_quf(path)
    return QUFModel.load(path, "cpu")


_rtok = st.integers(0, VOCAB - 1)


# Feature: quf-quantized-format, Property 30: Batched log-likelihood equals single-request log-likelihood
# **Validates: Requirements 27.1, 27.2, 27.3**
@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.tuples(st.lists(_rtok, max_size=8), st.lists(_rtok, max_size=5)),
        min_size=1,
        max_size=6,
    ),
    st.integers(1, 5),
    st.integers(1, 30),  # token cap, including caps below a single row's length
    st.integers(3, 16),  # max_length, forces left truncation
)
def test_batched_matches_single_real_model(tiny_model, items, batch_size, budget, max_length):
    items = items + [(items[0][0], c) for _, c in items]  # repeated contexts
    fwd = tiny_model.forward
    got = score_batched(fwd, items, max_length, eos_id=1, batch_size=batch_size, max_batch_tokens=budget)
    single = [score_batched(fwd, [it], max_length, eos_id=1, batch_size=1)[0] for it in items]
    _close(got, single)
