"""Property test for the Backend_Selector.

Feature: quf-quantized-format, Property 14: Backend selection rules
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from quf.device import DeviceInfo
from quf.runtime.selector import NoGPUError, select_backend

_capability = st.tuples(st.integers(0, 12), st.integers(0, 9))

_gpu_info = st.builds(
    DeviceInfo,
    st.just(True),
    st.just("GPU"),
    _capability,
    st.integers(1, 1 << 36),
)
_no_gpu_info = st.just(DeviceInfo(False, None, None, None))
_any_info = st.one_of(_gpu_info, _no_gpu_info)

# Fused self-test behavior: returns True, returns False, or raises.
_self_test_mode = st.sampled_from(["pass", "fail", "raise"])


@pytest.fixture(autouse=True, scope="module")
def _no_env_override():
    mp = pytest.MonkeyPatch()
    mp.delenv("QUF_NO_TRITON", raising=False)
    yield
    mp.undo()


def _make_self_test(mode: str):
    calls = []

    def self_test() -> bool:
        calls.append(1)
        if mode == "raise":
            raise RuntimeError("boom")
        return mode == "pass"

    return self_test, calls


@given(
    option=st.sampled_from(["auto", "cpu", "cuda"]),
    info=_any_info,
    mode=_self_test_mode,
    prefer_fused=st.booleans(),
    int4_ok=st.booleans(),
)
def test_backend_selection_rules(option, info, mode, prefer_fused, int4_ok):
    """**Validates: Requirements 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 15.1, 15.7**

    Automatic selection returns CPU, triton_int4 (when its probe passes and fused is not preferred)
    or a Torch GPU backend, and never Marlin.
    """
    probe_calls = []

    def probe():
        probe_calls.append(1)
        return int4_ok, "probe"

    self_test, calls = _make_self_test(mode)
    if option == "cuda" and not info.cuda_available:
        with pytest.raises(NoGPUError, match="no NVIDIA GPU was found"):
            select_backend(option, info, prefer_fused=prefer_fused, fused_self_test=self_test, int4_probe=probe)
        assert not calls and not probe_calls
        return
    choice = select_backend(option, info, prefer_fused=prefer_fused, fused_self_test=self_test, int4_probe=probe)
    if option == "cpu" or not info.cuda_available:
        assert choice.backend == "cpu"
        assert not calls and not probe_calls
    elif not prefer_fused and int4_ok:
        assert choice.backend == "triton_int4"
        assert not calls
    elif prefer_fused and mode == "pass":
        assert choice.backend == "torch_fused"
    else:
        assert choice.backend == "torch_gpu"
    assert choice.backend != "marlin"
    assert choice.reason
