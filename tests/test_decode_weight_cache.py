"""Regression (task 22.2, Requirement 28.2): weights are prepared once, decode never re-dequantizes.

The torch_gpu prepared weight is a cached fp16 tensor, the cpu prepared weight is cached in the
fp32 compute dtype, and the tied lm_head needs no per-token conversion.
"""

from __future__ import annotations

import pytest
import torch

import quf.runtime.backends.cpu as cpu_mod
import quf.runtime.backends.torch_gpu as gpu_mod
from quf.runtime.model import QUFModel
from test_backend_precisions import IDS, _build_mixed_quf

_PROJ = ("q", "k", "v", "o", "gate", "up", "down")


def _forbid(monkeypatch, *modules):
    def boom(*a, **k):
        raise AssertionError("dequantize called during decode")

    for mod in modules:
        monkeypatch.setattr(mod, "reference_dequantize", boom)
        monkeypatch.setattr(mod, "dequantize_int8_weight", boom)


def _decode(model, steps=3):
    cache = model.new_cache()
    out = model.forward(IDS, cache)
    for _ in range(steps):
        out = model.forward(out[:, -1:].argmax(-1), cache)
    return out


def _check(model, dtype, precisions):
    seen = set()
    for layer in model.layers:
        for n in _PROJ:
            lin = getattr(layer, n)
            seen.add(lin.precision)
            assert lin.prepared.dtype == dtype
    assert precisions <= seen


def test_cpu_prepared_weight_cached_fp32_and_no_dequant_in_decode(tmp_path, monkeypatch):
    path = tmp_path / "m.quf"
    _build_mixed_quf(path, 64)
    model = QUFModel.load(path, backend="cpu")
    _check(model, torch.float32, {"int4", "int8", "fp16"})
    assert model._lm_head_weight.dtype == torch.float32
    _forbid(monkeypatch, cpu_mod)
    assert torch.isfinite(_decode(model)).all()


@pytest.mark.gpu
def test_gpu_prepared_weight_cached_fp16_and_no_dequant_in_decode(tmp_path, monkeypatch):
    path = tmp_path / "m.quf"
    _build_mixed_quf(path, 64)
    model = QUFModel.load(path, backend="torch_gpu")
    _check(model, torch.float16, {"int4", "int8", "fp16"})
    assert model.layers[0].q.prepared.device.type == "cuda"
    # Tied lm_head is the fp16 embedding itself: no copy, no per-token conversion.
    assert model._lm_head_weight is model.embed and model._lm_head_weight.dtype == torch.float16
    _forbid(monkeypatch, gpu_mod)
    assert torch.isfinite(_decode(model)).all()
