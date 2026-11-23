"""GPU tests for the int8 per-output-channel M=1 Triton GEMV (task 39.1) against the fp32 reference."""

from __future__ import annotations

import pytest
import torch

from quf.quant.quantize import dequantize_int8, quantize_int8
from quf.runtime.kernels.decode_gemv import gemv_int8

pytestmark = [pytest.mark.gpu, pytest.mark.triton]

# (name, N, K): Qwen2.5-0.5B (hidden 896, inter 4864) and 1.5B (hidden 1536, inter 8960).
SHAPES = [
    ("0.5b.kv", 128, 896), ("0.5b.q", 896, 896), ("0.5b.gate", 4864, 896), ("0.5b.down", 896, 4864),
    ("1.5b.kv", 256, 1536), ("1.5b.q", 1536, 1536), ("1.5b.gate", 8960, 1536), ("1.5b.down", 1536, 8960),
]


def _weights(n: int, k: int):
    g = torch.Generator().manual_seed(n * 31 + k)
    q, s = quantize_int8(torch.randn(n, k, generator=g) * 0.05)
    return q.cuda(), s.cuda()


@pytest.mark.parametrize("name,n,k", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("scatter", [False, True], ids=["plain", "scatter"])
def test_matches_fp32_reference(name, n, k, bias, scatter):
    q, s = _weights(n, k)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, k, device="cuda", generator=g).to(torch.float16)
    b = torch.randn(n, device="cuda", generator=g).to(torch.float16) if bias else None
    ref = x.float() @ dequantize_int8(q, s).float().T
    if bias:
        ref = ref + b.float()
    x_index = out_index = None
    out = None
    if scatter:
        # input in a producer-scattered layout and output scattered into a larger shared buffer
        x_index = torch.randperm(k, device="cuda", generator=g).to(torch.int32)
        xs = torch.empty_like(x)
        xs[0, x_index.long()] = x[0]
        x = xs
        out_index = (torch.randperm(n, device="cuda", generator=g) + 7).to(torch.int32)
        out = torch.zeros(1, n + 16, device="cuda", dtype=torch.float16)
    res = gemv_int8(x, q, s, bias=b, out=out, x_index=x_index, out_index=out_index)
    if scatter:
        got = torch.empty(1, n, device="cuda", dtype=torch.float16)
        got[0] = res[0, out_index.long()]
    else:
        got = res
    assert got.dtype == torch.float16 and got.shape == (1, n)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-3)


def test_cuda_graph_replay_with_new_input():
    n, k = 128, 896
    q, s = _weights(n, k)
    b = torch.randn(n, device="cuda").to(torch.float16)
    idx = torch.arange(n, device="cuda", dtype=torch.int32).flip(0).contiguous()
    x = torch.randn(1, k, device="cuda").to(torch.float16)
    out = torch.empty(1, n, device="cuda", dtype=torch.float16)
    gemv_int8(x, q, s, b, out, out_index=idx)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemv_int8(x, q, s, b, out, out_index=idx)
    x.copy_(torch.randn(1, k, device="cuda").to(torch.float16))
    graph.replay()
    ref = x.float() @ dequantize_int8(q, s).float().T + b.float()
    torch.testing.assert_close(out.flip(1).float(), ref, atol=2e-2, rtol=2e-3)
