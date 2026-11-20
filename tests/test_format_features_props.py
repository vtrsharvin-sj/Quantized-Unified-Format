"""Properties 28 and 29, plus the defaults of the 0.2.0 calibration (task 16.7).

Feature: quf-quantized-format, Property 28: Size accounting equals tensor bytes
Feature: quf-quantized-format, Property 29: Multi-source calibration shape, shares, and determinism

**Validates: Requirements 22.5, 22.6, 23.2, 23.3, 23.4**
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from quf.convert import converter as C
from quf.convert.precision import PrecisionPlan, SizeReport
from quf.tokenization.calibration import (
    CalibrationShortfallError,
    CalibrationSource,
    build_calibration,
)

_BYTES = {"fp16": 2, "fp32": 4, "int32": 4, "int8": 1}


def _nbytes(dtype, shape):
    n = 1
    for d in shape:
        n *= d
    return n * _BYTES[dtype]


# --- Property 28 -----------------------------------------------------------


@st.composite
def _plan_and_shape(draw):
    group_size = draw(st.sampled_from([32, 64, 128]))
    mods = draw(st.lists(st.sampled_from(["int4", "int8", "fp16"]), min_size=1, max_size=6))
    tensors, expected = [], {"int4": 0, "int8": 0, "fp16": 0}
    modules = {}
    for i, prec in enumerate(mods):
        name = f"model.layers.{i}.mlp.proj"
        in_f = group_size * draw(st.integers(1, 3))
        out_f = 8 * draw(st.integers(1, 3))
        modules[name] = prec
        if prec == "int4":
            ts = [(f"{name}.qweight", "int32", (in_f // 8, out_f)),
                  (f"{name}.scales", "fp16", (in_f // group_size, out_f))]
            if draw(st.booleans()):
                ts.append((f"{name}.qzeros", "int32", (in_f // group_size, out_f // 8)))
            if draw(st.booleans()):
                ts.append((f"{name}.g_idx", "int32", (in_f,)))
        elif prec == "int8":
            ts = [(f"{name}.qweight_int8", "int8", (out_f, in_f)),
                  (f"{name}.scales", "fp16", (out_f,))]
        else:
            ts = [(f"{name}.weight", "fp16", (out_f, in_f))]
        tensors += ts
        expected[prec] += sum(_nbytes(d, s) for _, d, s in ts)
    # tensors outside any module: embeddings and norms count as fp16
    extra = [("model.embed_tokens.weight", "fp16", (draw(st.integers(1, 20)), 8)),
             ("model.norm.weight", "fp16", (8,))]
    tensors += extra
    expected["fp16"] += sum(_nbytes(d, s) for _, d, s in extra)
    return tensors, modules, expected


@settings(max_examples=100, deadline=None)
@given(_plan_and_shape())
def test_property_28_size_accounting(case):
    tensors, modules, expected = case
    rep = SizeReport.from_tensors(tensors, modules)
    assert (rep.int4, rep.int8, rep.fp16) == (expected["int4"], expected["int8"], expected["fp16"])
    assert rep.tensor_bytes == sum(_nbytes(d, s) for _, d, s in tensors)
    text = rep.format()
    for prec in ("int4", "int8", "fp16"):
        assert f"  {prec}: {expected[prec]} bytes" in text


# --- Property 29 -----------------------------------------------------------


class _Tok:
    def encode(self, text):
        return [ord(c) for c in text]


def _expected_counts(n, shares):
    floors = [int(n * s) for s in shares]
    rest = n - sum(floors)
    order = sorted(range(len(shares)), key=lambda i: (-(n * shares[i] - floors[i]), i))
    for i in order[:rest]:
        floors[i] += 1
    return floors


def _source(idx, ntokens, share):
    # Each source uses its own character range so windows can be traced back.
    base = 0x4E00 + idx * 0x100
    text = "".join(chr(base + (j % 200)) for j in range(ntokens))
    return CalibrationSource(f"src{idx}", lambda t=text: [t], share)


@st.composite
def _spec(draw):
    k = draw(st.integers(1, 4))
    denom = draw(st.sampled_from([2, 3, 4, 5, 8, 10]))
    cuts = sorted(draw(st.lists(st.integers(0, denom), min_size=k - 1, max_size=k - 1)))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [denom])]
    shares = [Fraction(p, denom) for p in parts]
    return shares, draw(st.integers(1, 12)), draw(st.integers(1, 16)), draw(st.integers(0, 2**31))


@settings(max_examples=100, deadline=None)
@given(_spec())
def test_property_29_multi_source(spec):
    shares, n, seq_len, seed = spec
    counts = _expected_counts(n, shares)
    assert sum(counts) == n
    sources = [_source(i, c * seq_len + 5, float(s)) for i, (c, s) in enumerate(zip(counts, shares))]

    first = build_calibration(sources, _Tok(), n, seq_len, seed)
    second = build_calibration(sources, _Tok(), n, seq_len, seed)
    assert first == second
    assert len(first) == n and all(len(w) == seq_len for w in first)

    pos = 0
    for i, c in enumerate(counts):
        lo = 0x4E00 + i * 0x100
        for w in first[pos : pos + c]:
            assert all(lo <= t < lo + 0x100 for t in w)
        pos += c


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(2, 10),
    seq_len=st.integers(2, 16),
    short_by=st.integers(1, 5),
)
def test_property_29_shortfall_names_source(n, seq_len, short_by):
    counts = _expected_counts(n, [Fraction(1, 2), Fraction(1, 2)])
    need = counts[1] * seq_len
    srcs = [_source(0, counts[0] * seq_len, 0.5), _source(1, max(need - short_by, 0), 0.5)]
    if counts[1] == 0:
        return
    with pytest.raises(CalibrationShortfallError) as e:
        build_calibration(srcs, _Tok(), n, seq_len, 0)
    assert "src1" in str(e.value)
    assert e.value.shortfall == need - max(need - short_by, 0)


def test_shares_must_sum_to_one():
    srcs = [_source(0, 100, 0.5), _source(1, 100, 0.3)]
    with pytest.raises(ValueError):
        build_calibration(srcs, _Tok(), 4, 4, 0)


# --- Unit tests: defaults and module-pattern selection ----------------------


def test_default_calibration_is_128_by_1024_wikitext_and_multi_source_is_opt_in():
    # Task 16.8: the two-source 128 x 2048 mix measured worse perplexity, so it is opt-in.
    assert (C.DEFAULT_N_SAMPLES, C.DEFAULT_SEQ_LEN) == (128, 1024)
    assert C.MULTI_SOURCE_SEQ_LEN == 2048
    srcs = C.default_calibration_sources()
    assert [s.share for s in srcs] == [0.5, 0.5]
    assert C.LEGACY_SEQ_LEN == 512


def test_module_pattern_selection():
    plan = PrecisionPlan.from_options(["*.v_proj=int8", "lm_head=fp16"])
    assert plan.precision("model.layers.7.self_attn.v_proj") == "int8"
    assert plan.precision("model.layers.7.self_attn.q_proj") == "int4"
    assert plan.precision("lm_head") == "fp16"
