"""Benchmark int4_gemv vs torch fp16 F.linear at M=1 (run: python -m quf.runtime.kernels.bench_int4)."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from quf.runtime.kernels.triton_kernels import (
    Int4Layout, _sm_count, dequantize_layout, gemv_config, int4_gemv,
)

SHAPES = [
    ("0.5b.q/o", 896, 896), ("0.5b.kv", 128, 896), ("0.5b.qkv", 1152, 896),
    ("0.5b.gate/up", 4864, 896), ("0.5b.gate_up", 9728, 896), ("0.5b.down", 896, 4864),
    ("1.5b.q/o", 1536, 1536), ("1.5b.kv", 256, 1536), ("1.5b.qkv", 2048, 1536),
    ("1.5b.gate/up", 8960, 1536), ("1.5b.gate_up", 17920, 1536), ("1.5b.down", 1536, 8960),
]


def bench(fn, iters: int = 200) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    for _ in range(300):  # warm up clocks
        g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(100):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / (100 * 10))
    return best  # microseconds per call (best of 7 rounds)

def main(gs: int = 128) -> None:
    sms = _sm_count(torch.device("cuda"))
    print(f"group_size={gs} M=1 (CUDA-graph timed)")
    print(f"{'shape':14}{'N':>7}{'K':>6}{'cfg':>14}{'int4 us':>9}{'GB/s':>7}{'fp16 us':>9}{'GB/s':>7}{'speedup':>8}")
    for name, n, k in SHAPES:
        packed = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda")
        sc = (torch.rand(n, k // gs, device="cuda") * 0.02 + 0.001).half()
        zr = torch.randint(0, 16, (n, k // gs), device="cuda").half()
        lay = Int4Layout(packed, sc, zr, None, gs, k, n)
        w = dequantize_layout(lay)
        x = torch.randn(1, k, device="cuda").half()
        t4 = bench(lambda: int4_gemv(x, lay))
        tf = bench(lambda: F.linear(x, w))
        bytes4 = packed.numel() * 4 + sc.numel() * 2 + zr.numel() * 2 + k * 2 + n * 2
        bytesf = w.numel() * 2 + k * 2 + n * 2
        cfg = gemv_config(n, k, gs, sms)
        print(f"{name:14}{n:7}{k:6}{str(cfg):>14}{t4:9.1f}{bytes4 / t4 / 1e3:7.0f}{tf:9.1f}{bytesf / tf / 1e3:7.0f}{tf / t4:8.2f}")


def bench_prefill(m: int = 512, gs: int = 128) -> None:
    """Prefill: dequant-to-buffer + F.linear vs cached fp16 F.linear (``python -m ... prefill``)."""
    from quf.runtime.kernels.dispatch import _int4_prefill, dequantize_into_buffer

    print(f"group_size={gs} M={m} (CUDA-graph timed)")
    print(f"{'shape':14}{'N':>7}{'K':>6}{'dequant us':>12}{'prefill us':>12}{'cached us':>11}{'ratio':>7}")
    for name, n, k in SHAPES:
        packed = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda")
        sc = (torch.rand(n, k // gs, device="cuda") * 0.02 + 0.001).half()
        zr = torch.randint(0, 16, (n, k // gs), device="cuda").half()
        lay = Int4Layout(packed, sc, zr, None, gs, k, n)
        w = dequantize_layout(lay)
        x = torch.randn(m, k, device="cuda").half()
        td = bench(lambda: dequantize_into_buffer(lay))
        tp = bench(lambda: _int4_prefill(x, lay))
        tc = bench(lambda: F.linear(x, w))
        print(f"{name:14}{n:7}{k:6}{td:12.1f}{tp:12.1f}{tc:11.1f}{tp / tc:7.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prefill":
        bench_prefill(int(sys.argv[2]) if len(sys.argv) > 2 else 512)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 128)
