"""Profile decode time per component for a QUF file (profiling only, runtime code untouched).

    python scripts/profile_decode.py model.quf --backend torch_gpu [--tokens 48] [--repeats 7]
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from quf.runtime import model as model_mod  # noqa: E402
from quf.runtime.linear import QUFLinear  # noqa: E402
from quf.runtime.model import QUFModel  # noqa: E402

PROMPT_IDS = [785, 3974, 13876, 38835, 34208, 916, 279, 17366, 5562, 13, 758, 264, 14861, 2484, 315, 4357, 11]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("quf_file")
    ap.add_argument("--backend", default="torch_gpu")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    gpu = args.backend != "cpu"
    m = QUFModel.load(args.quf_file, "cpu" if not gpu else "cuda", backend=args.backend)
    sync = torch.cuda.synchronize if gpu else (lambda: None)

    # --- inventory of prepared weights
    inv = collections.Counter()
    for layer in m.layers:
        for n in ("q", "k", "v", "o", "gate", "up", "down"):
            lin: QUFLinear = getattr(layer, n)
            inv[(lin.precision, str(lin.prepared.dtype))] += 1
    print("prepared weights (precision, dtype):", dict(inv))
    print("lm_head:", type(m.lm_head).__name__, getattr(m.lm_head, "dtype", ""), "tied" if m.lm_head is m.embed else "")

    # --- count prepare calls / gathers during decode
    calls = collections.Counter()
    import quf.quant.dequant as dq
    import quf.runtime.backends.base as base

    orig_ref, orig_i8 = dq.reference_dequantize, base.dequantize_int8_weight
    import quf.runtime.backends.torch_gpu as tg
    import quf.runtime.backends.cpu as cb

    def wrap(fn, key):
        def w(*a, **k):
            calls[key] += 1
            return fn(*a, **k)
        return w

    for mod in (tg, cb):
        mod.reference_dequantize = wrap(orig_ref, "reference_dequantize")
        mod.dequantize_int8_weight = wrap(orig_i8, "dequantize_int8_weight")

    cache = m.new_cache()
    ids = torch.tensor([PROMPT_IDS], device=m.device)

    def decode(n):
        cache = m.new_cache()
        logits = m.forward(ids, cache)
        nxt = int(logits[0, -1].argmax())
        sync()
        t = time.perf_counter()
        for _ in range(n - 1):
            logits = m.forward(torch.tensor([[nxt]], device=m.device), cache)
            nxt = int(logits[0, -1].argmax())
        sync()
        return (n - 1) / (time.perf_counter() - t)

    decode(8)
    decode(8)
    calls.clear()
    rates = [decode(args.tokens) for _ in range(args.repeats)]
    print(f"decode tok/s repeats: {[round(r, 1) for r in rates]}")
    print(f"MEDIAN {statistics.median(rates):.1f}  min {min(rates):.1f}  max {max(rates):.1f}")
    print("dequant calls during decode:", dict(calls), "(expected {} when cached)")

    # --- per-component wall time with sync (distorting but additive)
    acc = collections.defaultdict(float)
    cnt = collections.Counter()
    orig_fwd = QUFLinear.forward

    def timed_fwd(self, x):
        sync()
        t = time.perf_counter()
        y = orig_fwd(self, x)
        sync()
        acc[f"linear:{self.precision}"] += time.perf_counter() - t
        cnt[f"linear:{self.precision}"] += 1
        return y

    QUFLinear.forward = timed_fwd
    QUFLinear.__call__ = timed_fwd
    orig_norm, orig_rope, orig_att = model_mod._rms_norm, m._rope, m._attention

    def timed(key, fn):
        def w(*a, **k):
            sync()
            t = time.perf_counter()
            r = fn(*a, **k)
            sync()
            acc[key] += time.perf_counter() - t
            cnt[key] += 1
            return r
        return w

    model_mod._rms_norm = timed("rms_norm", orig_norm)
    m._attention = timed("attention_total(incl q/k/v/o linears)", orig_att)
    n = args.tokens
    sync()
    tot0 = time.perf_counter()
    decode_rate = decode(n)
    sync()
    total = time.perf_counter() - tot0
    print(f"\ninstrumented decode {decode_rate:.1f} tok/s (per-call syncs slow it)")
    per_tok = n - 1
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {k:45s} {v / per_tok * 1000:8.2f} ms/token  calls/token {cnt[k] / (per_tok + 1):.0f}")
    QUFLinear.forward = orig_fwd
    QUFLinear.__call__ = orig_fwd
    model_mod._rms_norm = orig_norm
    m._attention = orig_att

    # --- lm_head isolated
    h = torch.randn(1, 1, m.hidden_size, device=m.device, dtype=m.dtype)
    def lm():
        if isinstance(m.lm_head, QUFLinear):
            return m.lm_head(h)
        return h.to(torch.float32) @ m.lm_head.to(torch.float32).t()
    for _ in range(3):
        lm()
    ts = []
    for _ in range(20):
        sync()
        t = time.perf_counter()
        lm()
        sync()
        ts.append(time.perf_counter() - t)
    print(f"\nlm_head alone: median {statistics.median(ts) * 1000:.2f} ms/token (lm_head dtype {m.lm_head.dtype if not isinstance(m.lm_head, QUFLinear) else 'QUFLinear'})")

    # --- torch.profiler
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if gpu else [])
    with profile(activities=acts) as prof:
        decode(16)
        sync()
    sort = "cuda_time_total" if gpu else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort, row_limit=args.top, max_name_column_width=40))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
