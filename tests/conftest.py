"""Shared pytest configuration: Hypothesis profile and GPU/convert markers."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "quf",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("quf")


def _nvidia_gpu_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: requires an NVIDIA GPU (skipped when none is available)"
    )
    config.addinivalue_line(
        "markers", "triton: requires Triton (skipped when unavailable)"
    )
    config.addinivalue_line(
        "markers", "realmodel: needs a real converted model file (skipped when missing)"
    )
    config.addinivalue_line(
        "markers", "convert: requires the convert extra and a real model conversion"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _skip_triton(items)
    if not any("gpu" in item.keywords for item in items):
        return
    if _nvidia_gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="no NVIDIA GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


def _skip_triton(items: list[pytest.Item]) -> None:
    if not any("triton" in item.keywords for item in items):
        return
    from quf.runtime.kernels.triton_kernels import triton_available

    ok, reason = triton_available()
    if ok:
        return
    skip = pytest.mark.skip(reason=f"triton unavailable: {reason}")
    for item in items:
        if "triton" in item.keywords:
            item.add_marker(skip)
