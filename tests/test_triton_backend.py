"""triton_int4 backend and self-test (task 29.1). Requirements 33-36"""
from __future__ import annotations

import pytest
import torch

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import quantize_groupwise
from quf.runtime.backends.triton_int4 import TritonInt4Backend, triton_int4_self_test
from quf.runtime.linear import FusedProjection, QUFLinear

pytestmark = [pytest.mark.gpu, pytest.mark.triton]


def test_self_test_returns_ok_and_reason():
    ok, reason = triton_int4_self_test()
    assert ok, reason
    assert isinstance(reason, str)


def test_self_test_never_raises_when_disabled(monkeypatch):
    monkeypatch.setenv("QUF_NO_TRITON", "1")
    ok, reason = triton_int4_self_test()
    assert ok is False and "QUF_NO_TRITON" in reason


def test_backend_keeps_int4_packed_and_fused_projection_matches():
    be = TritonInt4Backend()
    torch.manual_seed(0)
    qws = [quantize_groupwise(torch.randn(n, 256) * 0.1, 64) for n in (64, 32, 32)]
    biases = [torch.randn(q.out_features).half().cuda() if i == 0 else None for i, q in enumerate(qws)]
    mods = [QUFLinear(q, be, "cuda", bias=b) for q, b in zip(qws, biases)]
    assert all(type(m.prepared).__name__ == "Int4Layout" for m in mods)
    x = torch.randn(1, 1, 256).half().cuda()
    expected = []
    for q, b in zip(qws, biases):
        y = x.float() @ reference_dequantize(q).cuda().float().t()
        expected.append(y + (0 if b is None else b.float()))
    fused = FusedProjection(mods)
    for got, exp in zip(fused(x), expected):
        assert (got.float() - exp).abs().max().item() < 2e-2
