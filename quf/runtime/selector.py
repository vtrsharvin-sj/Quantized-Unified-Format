"""Backend_Selector: pure selection of a Backend from a device option and DeviceInfo.

Automatic selection never returns Marlin. Marlin is chosen only with
``backend_option="marlin"`` (``--backend marlin``). The int4 backend (triton_int4) is the
default on CUDA while ``INT4_DEFAULT_ON`` is True, the CUDA graph is enabled, and Triton and its
self-test work (decision recorded in docs/BACKENDS.md).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Literal, NamedTuple

from quf.device import DeviceInfo

logger = logging.getLogger(__name__)

BACKEND_OPTIONS = ("auto", "cached", "fused", "marlin", "int4")

# Requirements 34.3 and 35.7: the single switch for the CUDA default. When True, ``auto`` picks
# triton_int4 if Triton imports, its self-test passes, and the CUDA graph is enabled. Decided True
# in task 31.2 (faster in graph mode, about half the memory, same perplexity). False would keep
# the int4 backend opt-in through ``--backend int4``.
INT4_DEFAULT_ON = True


class NoGPUError(RuntimeError):
    """Raised when a GPU is required and no NVIDIA GPU is available."""


class Int4UnavailableError(RuntimeError):
    """Raised when ``--backend int4`` is requested but Triton or its self-test is unavailable."""


class BackendChoice(NamedTuple):
    backend: Literal["marlin", "torch_fused", "torch_gpu", "triton_int4", "cpu"]
    reason: str


def _run_self_test(test: Callable[[], bool] | None) -> tuple[bool, str]:
    """Return ``(passed, failure_text)``. ``None`` counts as passed. An exception is a failure."""
    if test is None:
        return True, ""
    try:
        if bool(test()):
            return True, ""
        return False, "fused backend self-test failed"
    except Exception as exc:  # noqa: BLE001 - any self-test failure falls back
        return False, f"fused backend self-test raised {type(exc).__name__}: {exc}"


def int4_probe() -> tuple[bool, str]:
    """Triton_Gate plus Self_Test of the int4 backend. Never imports Triton when QUF_NO_TRITON=1."""
    if os.environ.get("QUF_NO_TRITON", "") == "1":
        return False, "disabled by QUF_NO_TRITON=1"
    try:
        from quf.runtime.backends.triton_int4 import triton_int4_self_test  # noqa: PLC0415

        return triton_int4_self_test()
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _check_int4(probe: Callable[[], tuple[bool, str]] | None) -> tuple[bool, str]:
    if os.environ.get("QUF_NO_TRITON", "") == "1":
        return False, "disabled by QUF_NO_TRITON=1"
    if probe is None:
        return False, "the int4 backend was not checked"
    try:
        ok, reason = probe()
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return bool(ok), str(reason)


def select_backend(
    device_option: Literal["auto", "cpu", "cuda"],
    info: DeviceInfo,
    *,
    backend_option: Literal["auto", "cached", "fused", "marlin", "int4"] = "auto",
    cached_bytes: int | None = None,
    free_bytes: int | None = None,
    prefer_fused: bool = False,
    fused_self_test: Callable[[], bool] | None = None,
    int4_probe: Callable[[], tuple[bool, str]] | None = None,
    cuda_graph: bool = True,
) -> BackendChoice:
    """Choose a Backend. Pure apart from logging and calling the self-tests.

    Rules for ``backend_option="auto"``:
      * CPU when ``--device cpu`` or no GPU is available (``NoGPUError`` for ``cuda``).
        Neither probe is called.
      * triton_int4 when ``INT4_DEFAULT_ON`` is set, ``prefer_fused`` is not, and ``int4_probe``
        (Triton import plus self-test, returning ``(ok, reason)``) passes. ``QUF_NO_TRITON=1`` or a
        missing probe means unavailable, and the probe is then not called. When a probe fails, its
        reason is added to the single "Selected backend" INFO line and the rules below apply.
      * `cuda_graph=False` (`--no-cuda-graph` or `QUF_NO_CUDA_GRAPH`) skips triton_int4, because\n        eager int4 decoding measured slower than torch_gpu. The probe is then not called.\n      * ``cuda_graph=False`` (``--no-cuda-graph`` or ``QUF_NO_CUDA_GRAPH``) skips triton_int4, because
        eager int4 decoding measured slower than torch_gpu. The probe is then not called.
      * The fused backend when the cached fp16 weights (``cached_bytes``) do not fit in
        ``free_bytes``, or ``prefer_fused`` is set (fused measured faster), and its
        self-test passes. A failed or raising self-test selects the cached backend.
      * The cached backend (``torch_gpu``) otherwise, including when sizes are unknown.
    ``cached`` and ``fused`` force that GPU backend; ``marlin`` forces Marlin; ``int4`` forces
    triton_int4 and raises ``Int4UnavailableError`` with the reason when it is unavailable.
    """
    if device_option not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unknown device option: {device_option!r}")
    if backend_option not in BACKEND_OPTIONS:
        raise ValueError(f"unknown backend option: {backend_option!r}")

    if device_option == "cpu":
        if backend_option != "auto":
            raise ValueError(f"--backend {backend_option} needs a GPU but --device cpu was requested")
        choice = BackendChoice("cpu", "--device cpu was requested")
    elif not info.cuda_available:
        if device_option == "cuda" or backend_option != "auto":
            raise NoGPUError("no NVIDIA GPU was found")
        choice = BackendChoice("cpu", "no NVIDIA GPU was found")
    elif backend_option == "int4":
        ok, why = _check_int4(int4_probe)
        if not ok:
            raise Int4UnavailableError(f"--backend int4 is unavailable: {why}")
        choice = BackendChoice("triton_int4", "--backend int4 was requested")
    elif backend_option == "marlin":
        choice = BackendChoice("marlin", "--backend marlin was requested")
    elif backend_option == "cached":
        choice = BackendChoice("torch_gpu", "--backend cached was requested")
    elif backend_option == "fused":
        choice = BackendChoice("torch_fused", "--backend fused was requested")
    else:
        note = ""
        int4_wanted = INT4_DEFAULT_ON and not prefer_fused and cuda_graph
        int4_ok = False
        if INT4_DEFAULT_ON and not prefer_fused and not cuda_graph:
            note = " (int4 skipped: the CUDA graph is disabled and eager int4 is slower than torch_gpu)"
        if int4_wanted:
            int4_ok, int4_why = _check_int4(int4_probe)
            if not int4_ok and (int4_probe is not None or os.environ.get("QUF_NO_TRITON", "") == "1"):
                note = f" (int4 unavailable: {int4_why})"
        if int4_ok:
            choice = BackendChoice("triton_int4", "Triton works and the int4 self-test passed; reads packed 4-bit weights")
        else:
            does_not_fit = cached_bytes is not None and free_bytes is not None and cached_bytes > free_bytes
            if does_not_fit:
                want = f"cached fp16 weights ({cached_bytes} bytes) do not fit in free GPU memory ({free_bytes} bytes)"
            elif prefer_fused:
                want = "fused backend measured faster than the cached backend"
            else:
                want = ""
            if not want:
                choice = BackendChoice(
                    "torch_gpu", "cached fp16 weights fit and the fused backend is not faster" + note
                )
            else:
                passed, failure = _run_self_test(fused_self_test)
                if passed:
                    choice = BackendChoice("torch_fused", want + note)
                else:
                    logger.debug("%s", failure)
                    logger.info("fused backend unavailable, falling back to the cached backend")
                    choice = BackendChoice("torch_gpu", failure + note)

    logger.info("Selected backend %s: %s", choice.backend, choice.reason)
    return choice
