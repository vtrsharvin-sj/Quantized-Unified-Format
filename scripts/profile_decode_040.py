"""Per-kernel GPU time per token of graph-mode decode, kernels aggregated by name (tasks 33.5, 39.3).
    python scripts/profile_decode_040.py --quf eval_cache/lm040/q15.quf --out eval_cache/speed040/profile_q15.json
Works on any QUF file (fp16 or quantized embedding). This is the single decode profiler; it replaced
profile_decode_graph.py. It also checks Requirement 42.6: no cuBLAS fp16 GEMV and no neg/cat/index_copy kernel.
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    import torch
    from torch.profiler import ProfilerActivity, profile
    import measure_decode_speed as M
    from quf.bench.suite import _tokenizer_from_header
    from quf.runtime.generate import generate
    from quf.runtime.model import QUFModel
    m = QUFModel.load(str(ROOT / a.quf), "cuda", backend="triton_int4")
    M._suppress_eos(m)
    tok = _tokenizer_from_header(m.header)
    ids = tok.encode(M.PROMPT)
    for _ in range(2):
        generate(m, None, ids, max_new_tokens=32, temperature=0.0, cuda_graph=None)
    n = a.tokens

    def collect(count):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            out = generate(m, None, ids, max_new_tokens=count, temperature=0.0, cuda_graph=None)
            torch.cuda.synchronize()
        agg = collections.defaultdict(lambda: [0.0, 0])
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                agg[e.name][0] += e.device_time
                agg[e.name][1] += 1
        return len(out), agg

    # Steady decode only: profile n tokens and 1 token (prefill + first token), subtract, divide by the
    # n - 1 graph-replayed decode tokens.
    n_full, full = collect(n)
    n_one, one = collect(1)
    toks = n_full - n_one
    agg = {}
    for k in full:
        us = full[k][0] - one.get(k, [0.0, 0])[0]
        cnt = full[k][1] - one.get(k, [0.0, 0])[1]
        if cnt > 0:
            agg[k] = [us, cnt]
    rows = sorted(((k, v[0] / toks, v[1] / toks) for k, v in agg.items()), key=lambda r: -r[1])
    total = sum(r[1] for r in rows)
    rec = {"file": a.quf, "tokens": toks, "gpu_us_per_token_total": total, "fused_decode": bool(m.fused_decode),
           "kernels": [{"name": k, "us_per_token": us, "launches_per_token": c, "share": us / total} for k, us, c in rows[:25]]}
    bad = {"cublas_gemv": [r for r in rows if "gemv" in r[0].lower() and ("cublas" in r[0].lower() or "gemvx" in r[0].lower() or "gemv2" in r[0].lower() or "gemvNSP" in r[0])],
           "neg_cat_index_copy": [r for r in rows if any(t in r[0].lower() for t in ("neg", "catarray", "cat_", "index_copy", "indexcopy", "index_put"))]}
    rec["req42_violations"] = {k: [{"name": r[0], "us_per_token": r[1], "launches_per_token": r[2]} for r in v] for k, v in bad.items()}
    rec["all_kernels"] = [{"name": k, "us_per_token": us, "launches_per_token": c} for k, us, c in rows]
    rec["launches_per_token_total"] = sum(r[2] for r in rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"total GPU {total:.0f} us/token over {toks} tokens; fused={m.fused_decode}")
    print("req42 violations:", {k: [r[0][:80] for r in v] for k, v in bad.items()})
    for k, us, c in rows[:25]:
        print(f"{us:8.1f} us {100 * us / total:5.1f}%  x{c:6.1f}  {k[:110]}")
if __name__ == "__main__":
    main()


