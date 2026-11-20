"""Property test for the extended Backend_Selector.

Feature: quf-quantized-format, Property 21: Extended backend selection rules
"""

from __future__ import annotations

from hypothesis import given, strategies as st

from quf.device import DeviceInfo
from quf.runtime.selector import select_backend

_gpu_info = st.just(DeviceInfo(True, "GPU", (12, 0), 1 << 34))
_no_gpu_info = st.just(DeviceInfo(False, None, None, None))
_size = st.one_of(st.none(), st.integers(0, 1 << 36))


def _make_self_test(mode: str):
    calls = []

    def self_test() -> bool:
        calls.append(1)
        if mode == "raise":
            raise RuntimeError("boom")
        return mode == "pass"

    return self_test, calls


@given(
    device=st.sampled_from(["auto", "cpu", "cuda"]),
    info=st.one_of(_gpu_info, _no_gpu_info),
    cached_bytes=_size,
    free_bytes=_size,
    prefer_fused=st.booleans(),
    mode=st.sampled_from(["pass", "fail", "raise"]),
)
def test_extended_backend_selection_rules(device, info, cached_bytes, free_bytes, prefer_fused, mode):
    """**Validates: Requirements 15.1, 15.4, 15.5, 15.7**

    CPU without CUDA; fused when weights do not fit or fused is faster and its
    self-test passes; cached otherwise; Marlin never in automatic mode.
    """
    self_test, calls = _make_self_test(mode)
    if device == "cuda" and not info.cuda_available:
        return  # covered by Property 14 (NoGPUError)
    choice = select_backend(
        device,
        info,
        cached_bytes=cached_bytes,
        free_bytes=free_bytes,
        prefer_fused=prefer_fused,
        fused_self_test=self_test,
    )
    if device == "cpu" or not info.cuda_available:
        assert choice.backend == "cpu"
        assert not calls
        return
    does_not_fit = cached_bytes is not None and free_bytes is not None and cached_bytes > free_bytes
    if does_not_fit or prefer_fused:
        assert len(calls) == 1
        assert choice.backend == ("torch_fused" if mode == "pass" else "torch_gpu")
    else:
        assert not calls
        assert choice.backend == "torch_gpu"
    assert choice.backend != "marlin"
    assert choice.reason
