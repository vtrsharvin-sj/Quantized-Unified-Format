"""Task 29.1: compare the triton_int4 backend with torch_gpu (tokens, GPU memory, tok/s).

    python scripts/verify_triton_backend.py FILE.quf [--repeats 5] [--tokens 128]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

PROMPTS = ["The quick brown fox", "def fibonacci(n):", "Explain why the sky is blue in one sentence."]


def load(path, backend):
    from quf.runtime.model import QUFModel

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    m = QUFModel.load(path, "cuda", backend=backend)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return m, (torch.cuda.memory_allocated() - before) / 2**20


def teacher_logits(model, ids):
    with torch.no_grad():
        return model.forward(torch.tensor([ids]))[0].float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("quf")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()

    from quf.bench.suite import _tokenizer_from_header
    from quf.runtime.generate import generate

    ref, ref_mb = load(a.quf, "torch_gpu")
    tri, tri_mb = load(a.quf, "triton_int4")
    tok = _tokenizer_from_header(ref.header)
    print(f"{Path(a.quf).name}: GPU memory after load  torch_gpu {ref_mb:.0f} MiB | triton_int4 {tri_mb:.0f} MiB")
    kinds = {}
    for layer in tri.layers:
        for name in ("qkv", "gate_up"):
            fp = getattr(layer, name)
            if fp is not None:
                kinds[(name, fp._kernel.is_fused)] = kinds.get((name, fp._kernel.is_fused), 0) + 1
    print("fusion:", kinds)

    all_ok = True
    for graph in (True, False):
        for p in PROMPTS:
            ids = tok.encode(p)
            r = generate(ref, tok, ids, max_new_tokens=48, temperature=0.0, cuda_graph=graph)
            t = generate(tri, tok, ids, max_new_tokens=48, temperature=0.0, cuda_graph=graph)
            same = r == t
            all_ok &= same
            msg = f"graph={graph!s:5} prompt={p[:24]!r:28} identical={same}"
            if not same:
                k = next(i for i, (x, y) in enumerate(zip(r, t)) if x != y)
                ctx = ids + r[:k]
                lr, lt = teacher_logits(ref, ctx)[-1], teacher_logits(tri, ctx)[-1]
                top = torch.topk(lr, 2)
                msg += (f"\n   first divergence at step {k}: ref={r[k]} tri={t[k]}"
                        f" | ref gap(top1-top2)={top.values[0] - top.values[1]:.4f}"
                        f" | ref logit gap(ref[ref_tok]-ref[tri_tok])={lr[r[k]] - lr[t[k]]:.4f}"
                        f" | tri gap(tri[ref_tok]-tri[tri_tok])={lt[r[k]] - lt[t[k]]:.4f}"
                        f" | max|logit diff|={(lr - lt).abs().max():.4f}")
            print(msg, flush=True)

    ids = tok.encode(PROMPTS[0])
    models = {"torch_gpu": ref, "triton_int4": tri}
    # Timed runs must not stop at EOS (the prompt hits it after ~17 tokens, so capture + prefill
    # would dominate). The tokenizer dict is restored afterwards.
    saved = {n: dict(m.header.tokenizer or {}) for n, m in models.items()}
    for m in models.values():
        (m.header.tokenizer or {}).pop("eos_token_id", None)
    speeds = {name: [] for name in models}
    decode_ms = {name: [] for name in models}
    from quf.runtime.cuda_graph import make_decoder

    for i in range(2 + a.repeats):  # alternate the backends every round to control clock/thermal noise
        for name, m in (models.items() if i % 2 == 0 else reversed(list(models.items()))):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = generate(m, tok, ids, max_new_tokens=a.tokens, temperature=0.0, cuda_graph=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            # steady-state decode only: graph already captured, replay a fixed number of steps
            cache = m.new_static_cache(len(ids) + a.tokens + 8)
            m.forward(torch.tensor([ids]), cache)
            dec = make_decoder(m, cache, True)
            dec.decode_token(out[0], cache)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(a.tokens // 2):
                dec.decode_token(out[1], cache)
            torch.cuda.synchronize()
            dd = (time.perf_counter() - t1) / (a.tokens // 2)
            if i >= 2:
                speeds[name].append(len(out) / dt)
                decode_ms[name].append(dd * 1000)
    for name, m in models.items():
        m.header.tokenizer.clear()
        m.header.tokenizer.update(saved[name])
    for name, s in speeds.items():
        print(f"{name}: median {statistics.median(s):.1f} tok/s end-to-end (graph, {a.tokens} tokens, EOS suppressed, "
              f"n={a.repeats}, alternating) all={[round(v) for v in s]}")
        d = statistics.median(decode_ms[name])
        print(f"{name}: steady-state decode {d:.2f} ms/token = {1000 / d:.1f} tok/s")
    print("ALL IDENTICAL" if all_ok else "DIVERGENCE FOUND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
