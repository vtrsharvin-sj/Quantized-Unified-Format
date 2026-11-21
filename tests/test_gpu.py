"""GPU unit and integration tests (Torch_GPU, Marlin self-test, GPUMemoryError mapping)."""

from __future__ import annotations

import hashlib

import pytest
import torch
from test_model import VOCAB, build_tiny_quf

from quf.quant.quantize import quantize_groupwise
from quf.runtime.errors import GPUMemoryError
from quf.runtime.model import QUFModel

pytestmark = pytest.mark.gpu


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gptqmodel_present() -> bool:
    from quf.runtime.backends.marlin import MarlinUnavailableError, _load_gptqmodel

    try:
        _load_gptqmodel()
    except MarlinUnavailableError:
        return False
    return True


def test_torch_gpu_end_to_end_matches_cpu(tmp_path):
    path = tmp_path / "tiny.quf"
    build_tiny_quf(path)
    before = _sha256(path)

    cpu = QUFModel.load(path, "cpu")
    gpu = QUFModel.load(path, "cuda", backend="torch_gpu")
    assert gpu.device.type == "cuda"
    assert gpu.backend_name == "torch_gpu"

    ids = torch.tensor([[3, 7, 1, 9, 4, 2]])
    ref = cpu.forward(ids)
    out = gpu.forward(ids)
    assert out.shape == (1, 6, VOCAB)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
    assert torch.allclose(out.cpu(), ref, atol=0.1, rtol=0.05)
    # Greedy choices agree on at least most positions despite fp16 vs fp32 accumulation.
    agree = (out.cpu().argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree >= 0.5

    # Loading never modifies the QUF_File (Requirement 9.5 / read-only load).
    assert _sha256(path) == before


def test_torch_gpu_cache_equivalence(tmp_path):
    path = tmp_path / "tiny.quf"
    build_tiny_quf(path)
    model = QUFModel.load(path, "cuda", backend="torch_gpu")
    ids = torch.tensor([[3, 7, 1, 9, 4, 2]])
    full = model.forward(ids)
    cache = model.new_cache()
    parts = [model.forward(ids[:, :4], cache)]
    parts += [model.forward(ids[:, i : i + 1], cache) for i in (4, 5)]
    assert cache.length == 6
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=5e-2)


def test_torch_gpu_backend_rejects_cpu_device():
    from quf.runtime.backends.torch_gpu import TorchGPUBackend

    qw = quantize_groupwise(torch.randn(128, 128) * 0.1)
    with pytest.raises(ValueError):
        TorchGPUBackend().prepare(qw, "cpu")


def test_marlin_self_test_returns_false_without_gptqmodel():
    if _gptqmodel_present():
        pytest.skip("gptqmodel is installed")
    from quf.runtime.backends.marlin import marlin_self_test

    assert marlin_self_test() is False


def test_marlin_prepare_raises_unavailable_without_gptqmodel():
    if _gptqmodel_present():
        pytest.skip("gptqmodel is installed")
    from quf.runtime.backends.marlin import MarlinBackend, MarlinUnavailableError

    qw = quantize_groupwise(torch.randn(128, 128) * 0.1, symmetric=True)
    with pytest.raises(MarlinUnavailableError):
        MarlinBackend().prepare(qw, "cuda")


def test_auto_selection_falls_back_to_torch_gpu_when_marlin_unavailable(tmp_path):
    if _gptqmodel_present():
        pytest.skip("gptqmodel is installed")
    path = tmp_path / "tiny.quf"
    build_tiny_quf(path)
    model = QUFModel.load(path, "auto")
    # Automatic selection never picks Marlin: triton_int4 when Triton works, else torch_gpu.
    assert model.backend_name in ("triton_int4", "torch_gpu")


def test_marlin_load_keeps_file_unchanged(tmp_path):
    from quf.runtime.backends.marlin import marlin_self_test

    if not marlin_self_test():
        pytest.skip("Marlin kernels not usable here")
    path = tmp_path / "tiny.quf"
    build_tiny_quf(path)  # asymmetric: Marlin must refuse cleanly, file untouched
    before = _sha256(path)
    with pytest.raises(Exception):
        QUFModel.load(path, "cuda", backend="marlin")
    assert _sha256(path) == before


def test_oom_during_load_maps_to_gpu_memory_error(tmp_path, monkeypatch):
    path = tmp_path / "tiny.quf"
    build_tiny_quf(path)

    def boom(self, reader, be):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(QUFModel, "_load_tensors", boom)
    with pytest.raises(GPUMemoryError) as ei:
        QUFModel.load(path, "cuda", backend="torch_gpu")
    err = ei.value
    assert err.required > 0
    assert err.available > 0  # real GPU: mem_get_info works
    msg = str(err)
    assert "out of memory" in msg
    assert f"{err.required} bytes" in msg
    assert f"{err.available} bytes" in msg
    assert isinstance(err.__cause__, torch.cuda.OutOfMemoryError)


def test_gpu_memory_error_reports_required_and_available():
    err = GPUMemoryError(3 * 1024**2, 1024**2)
    assert "3.0 MiB required" in str(err)
    assert "1.0 MiB available" in str(err)
