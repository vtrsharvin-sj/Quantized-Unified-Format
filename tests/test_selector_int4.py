"""Property test for backend selection with the int4 backend (pure, no GPU, no Triton).

Feature: quf-quantized-format, Property 36: Backend selection rules with the int4 backend
**Validates: Requirements 34.2, 34.3, 34.4, 34.5, 34.6, 34.8**
"""
from __future__ import annotations

import sys

import pytest
from hypothesis import given, settings, strategies as st

from quf.device import DeviceInfo
from quf.runtime import selector
from quf.runtime.selector import Int4UnavailableError, NoGPUError, select_backend

_GPU = DeviceInfo(True, "GPU", (8, 0), 1 << 34)
_NO_GPU = DeviceInfo(False, None, None, None)

_sizes = st.one_of(st.none(), st.integers(0, 1 << 36))


class _Calls:
    def __init__(self, mode: str | None, kind: str):
        self.mode, self.kind, self.n = mode, kind, 0

    def __call__(self):
        self.n += 1
        if self.mode == "raise":
            raise RuntimeError("boom")
        ok = self.mode == "pass"
        if self.kind == "probe":
            return ok, ("" if ok else "triton gate or self-test failed")
        return ok


@settings(max_examples=300, deadline=None)
@given(
    device_option=st.sampled_from(["auto", "cpu", "cuda"]),
    gpu=st.booleans(),
    backend_option=st.sampled_from(["auto", "cached", "fused", "marlin", "int4"]),
    probe_mode=st.sampled_from(["pass", "fail", "raise", None]),  # None: no probe supplied
    self_test_mode=st.sampled_from(["pass", "fail", "raise", None]),  # None: no self-test
    no_triton=st.booleans(),
    default_on=st.booleans(),
    prefer_fused=st.booleans(),
    cached_bytes=_sizes,
    free_bytes=_sizes,
)
def test_property_36_selection_rules(
    device_option, gpu, backend_option, probe_mode, self_test_mode, no_triton, default_on,
    prefer_fused, cached_bytes, free_bytes,
):
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(selector, "INT4_DEFAULT_ON", default_on)
        if no_triton:
            mp.setenv("QUF_NO_TRITON", "1")
        else:
            mp.delenv("QUF_NO_TRITON", raising=False)
        probe = None if probe_mode is None else _Calls(probe_mode, "probe")
        self_test = None if self_test_mode is None else _Calls(self_test_mode, "self")
        info = _GPU if gpu else _NO_GPU

        def run():
            return select_backend(
                device_option, info, backend_option=backend_option, cached_bytes=cached_bytes,
                free_bytes=free_bytes, prefer_fused=prefer_fused, fused_self_test=self_test,
                int4_probe=probe,
            )

        probe_calls = lambda: 0 if probe is None else probe.n  # noqa: E731
        test_calls = lambda: 0 if self_test is None else self_test.n  # noqa: E731
        int4_probe_ok = probe_mode == "pass" and not no_triton

        # Device rules (34.2): no CUDA gives CPU, and neither probe runs.
        if device_option == "cpu":
            if backend_option != "auto":
                with pytest.raises(ValueError):
                    run()
            else:
                assert run().backend == "cpu"
            assert probe_calls() == 0 and test_calls() == 0
            return
        if not gpu:
            if device_option == "cuda" or backend_option != "auto":
                with pytest.raises(NoGPUError):
                    run()
            else:
                assert run().backend == "cpu"
            assert probe_calls() == 0 and test_calls() == 0
            return

        # GPU present. Explicit backends (34.4, 34.6).
        if backend_option == "int4":
            if int4_probe_ok:
                assert run().backend == "triton_int4"
            else:
                with pytest.raises(Int4UnavailableError) as ei:
                    run()
                assert str(ei.value)  # carries a reason
            assert test_calls() == 0
            if no_triton:
                assert probe_calls() == 0  # QUF_NO_TRITON never reaches Triton
            return
        if backend_option in ("cached", "fused", "marlin"):
            expected = {"cached": "torch_gpu", "fused": "torch_fused", "marlin": "marlin"}[backend_option]
            assert run().backend == expected
            assert probe_calls() == 0 and test_calls() == 0
            return

        # auto on CUDA (34.3, 34.5, 34.8).
        choice = run()
        int4_wanted = default_on and not prefer_fused
        if no_triton:
            assert probe_calls() == 0
            assert choice.backend != "triton_int4"
        if int4_wanted and int4_probe_ok:
            assert choice.backend == "triton_int4"
            assert test_calls() == 0
            return
        assert choice.backend in ("torch_gpu", "torch_fused")
        assert choice.backend != "triton_int4"
        assert probe_calls() == (1 if int4_wanted and probe is not None and not no_triton else 0)
        if int4_wanted and (probe is not None or no_triton):
            assert "int4 unavailable" in choice.reason  # the failure reason is reported
        does_not_fit = cached_bytes is not None and free_bytes is not None and cached_bytes > free_bytes
        if does_not_fit or prefer_fused:
            assert test_calls() == (1 if self_test is not None else 0)
            fused_ok = self_test is None or self_test_mode == "pass"
            assert choice.backend == ("torch_fused" if fused_ok else "torch_gpu")
        else:
            assert test_calls() == 0
            assert choice.backend == "torch_gpu"
    finally:
        mp.undo()


def test_no_triton_probe_never_imports_triton_backend(monkeypatch):
    monkeypatch.setenv("QUF_NO_TRITON", "1")
    monkeypatch.delitem(sys.modules, "quf.runtime.backends.triton_int4", raising=False)
    ok, why = selector.int4_probe()
    assert not ok and "QUF_NO_TRITON" in why
    assert "quf.runtime.backends.triton_int4" not in sys.modules


def test_default_is_deterministic():
    kw = dict(backend_option="auto", int4_probe=lambda: (True, ""))
    assert select_backend("auto", _GPU, **kw) == select_backend("auto", _GPU, **kw)
