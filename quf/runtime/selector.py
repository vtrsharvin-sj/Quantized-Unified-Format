"""Backend_Selector: pure selection of a Backend from a device option and DeviceInfo.

Automatic selection never returns Marlin. Marlin is chosen only with
``backend_option="marlin"`` (``--backend marlin``).
"""

from __future__ import annotations

import logging
from typing import Callable, Literal, NamedTuple

from quf.device import DeviceInfo

logger = logging.getLogger(__name__)

BACKEND_OPTIONS = ("auto", "cached", "fused", "marlin")


class NoGPUError(RuntimeError):
    """Raised when a GPU is required and no NVIDIA GPU is available."""


class BackendChoice(NamedTuple):
    backend: Literal["marlin", "torch_fused", "torch_gpu", "cpu"]
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


def select_backend(
    device_option: Literal["auto", "cpu", "cuda"],
    info: DeviceInfo,
    *,
    backend_option: Literal["auto", "cached", "fused", "marlin"] = "auto",
    cached_bytes: int | None = None,
    free_bytes: int | None = None,
    prefer_fused: bool = False,
    fused_self_test: Callable[[], bool] | None = None,
) -> BackendChoice:
    """Choose a Backend. Pure apart from logging and calling ``fused_self_test``.

    Rules for ``backend_option="auto"``:
      * CPU when ``--device cpu`` or no GPU is available (``NoGPUError`` for ``cuda``).
      * The fused backend when the cached fp16 weights (``cached_bytes``) do not fit in
        ``free_bytes``, or ``prefer_fused`` is set (fused measured faster), and its
        self-test passes. A failed or raising self-test selects the cached backend.
      * The cached backend (``torch_gpu``) otherwise, including when sizes are unknown.
    ``cached`` and ``fused`` force that GPU backend; ``marlin`` forces Marlin.
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
    elif backend_option == "marlin":
        choice = BackendChoice("marlin", "--backend marlin was requested")
    elif backend_option == "cached":
        choice = BackendChoice("torch_gpu", "--backend cached was requested")
    elif backend_option == "fused":
        choice = BackendChoice("torch_fused", "--backend fused was requested")
    else:
        does_not_fit = cached_bytes is not None and free_bytes is not None and cached_bytes > free_bytes
        if does_not_fit:
            want = f"cached fp16 weights ({cached_bytes} bytes) do not fit in free GPU memory ({free_bytes} bytes)"
        elif prefer_fused:
            want = "fused backend measured faster than the cached backend"
        else:
            want = ""
        if not want:
            choice = BackendChoice("torch_gpu", "cached fp16 weights fit and the fused backend is not faster")
        else:
            passed, failure = _run_self_test(fused_self_test)
            if passed:
                choice = BackendChoice("torch_fused", want)
            else:
                logger.debug("%s", failure)
                logger.info("fused backend unavailable, falling back to the cached backend")
                choice = BackendChoice("torch_gpu", failure)

    logger.info("Selected backend %s: %s", choice.backend, choice.reason)
    return choice
