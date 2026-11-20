"""Unit tests for package metadata and device detection (Requirements 1.1-1.5)."""

from __future__ import annotations

import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

import quf
from quf.device import DeviceInfo, detect_devices

ROOT = Path(__file__).resolve().parent.parent
NO_GPU = DeviceInfo(False, None, None, None)


def test_version_is_string_and_matches_pyproject():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert isinstance(quf.__version__, str) and quf.__version__
    assert quf.__version__ == data["project"]["version"]


def test_declared_extras_and_console_script():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert {"cuda", "convert"} <= set(extras)
    assert data["project"]["scripts"]["quf"] == "quf.cli:main"
    # torch is a base dependency (Requirement 13.1); the cuda extra is kept empty.
    assert any(d.lower().startswith("torch") for d in data["project"]["dependencies"])
    assert not any("gptqmodel" in d.lower() for d in extras["convert"])


def test_import_with_extras_absent():
    code = (
        "import sys\n"
        "for n in ('torch','gptqmodel','transformers','datasets','safetensors'):\n"
        "    sys.modules[n] = None\n"
        "import quf, quf.device\n"
        "assert quf.device.detect_devices().cuda_available is False\n"
        "print(quf.__version__)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == quf.__version__


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        quf.does_not_exist  # noqa: B018


def _fake_torch(*, available, raises=False):
    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: available

    def boom(*a, **k):
        raise RuntimeError("cuda failure")

    if raises:
        cuda.get_device_properties = boom
    else:
        cuda.get_device_properties = lambda i: types.SimpleNamespace(total_memory=16 * 1024**3)
    cuda.get_device_capability = lambda i: (12, 0)
    cuda.get_device_name = lambda i: "NVIDIA GeForce RTX 5080"
    return types.SimpleNamespace(cuda=cuda)


def test_detect_devices_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True))
    info = detect_devices()
    assert info == DeviceInfo(True, "NVIDIA GeForce RTX 5080", (12, 0), 16 * 1024**3)


def test_detect_devices_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    assert detect_devices() == NO_GPU


def test_detect_devices_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    assert detect_devices() == NO_GPU


def test_detect_devices_query_failure_is_no_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, raises=True))
    assert detect_devices() == NO_GPU
