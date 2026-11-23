"""Sweep (BLOCK_N, BLOCK_K, num_warps) of the quantized lm_head GEMV on the Qwen2.5-0.5B shape."""
import itertools
import sys

import torch

from quf.quant.embedding import quantize_embedding_rtn
from quf.runtime.kernels.embed_gemv import embed_gemv

N, K = 151936, 896
gs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
x = torch.randn(1, K, device="cuda").half()
for bits in (4, 8):
    qe = quantize_embedding_rtn(torch.randn(N, K) * 0.05, bits, gs)
    p, s = torch.from_numpy(qe.packed).cuda(), torch.from_numpy(qe.scales).cuda()
    print(f"bits={bits} gs={gs}")
    for bn, bk, w in itertools.product((4, 8, 16, 32), (64, 128, 256), (1, 2, 4, 8)):
        if bk % gs and bk < gs:
            continue
        try:
            embed_gemv(x, p, s, bits, gs, config=(bn, bk, w))
            ts = []
            for _ in range(20):
                flush.zero_()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                embed_gemv(x, p, s, bits, gs, config=(bn, bk, w))
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
            ts.sort()
            print(f"  BN={bn:2d} BK={bk:3d} warps={w}: {ts[len(ts)//2]*1000:7.1f} us")
        except Exception as e:  # noqa: BLE001
            print(f"  BN={bn} BK={bk} warps={w}: fail {type(e).__name__}")
