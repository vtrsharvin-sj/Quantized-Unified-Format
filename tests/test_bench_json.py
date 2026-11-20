"""Property 17: Benchmark results JSON round trip.

**Validates: Requirements 11.6**
"""

from __future__ import annotations

import json

from hypothesis import given, settings, strategies as st

from quf.bench.suite import BenchResult

_text = st.text(min_size=1, max_size=40)
_float = st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False)
_opt_float = st.none() | _float
_opt_int = st.none() | st.integers(min_value=0, max_value=2**50)

results = st.builds(
    BenchResult,
    backend=st.sampled_from(["cpu", "torch_gpu", "marlin"]),
    device_name=_text,
    software_versions=st.dictionaries(_text, _text, min_size=1, max_size=8),
    perplexity=_float,
    file_size_bytes=st.integers(min_value=0, max_value=2**50),
    tokens_per_second=_float,
    dataset=_text,
    ppl_target_percent=_float,
    fp16_perplexity=_opt_float,
    relative_ppl_increase=st.none() | st.floats(-1, 10, allow_nan=False),
    gguf_size_bytes=_opt_int,
    gguf_tokens_per_second=_opt_float,
)


@settings(max_examples=100, deadline=None)
@given(results)
def test_bench_results_json_round_trip(result):
    text = result.to_json()
    parsed = BenchResult.from_json(text)
    assert parsed == result
    raw = json.loads(text)
    assert raw["backend"] == result.backend
    assert raw["device_name"] == result.device_name
    assert raw["software_versions"] == result.software_versions
