"""Task 26.8: median decode speed of a QUF file (eager vs CUDA graph, CPU) and llama-bench.

    python scripts/measure_decode_speed.py --quf f.quf --mode eager|graph|cpu --out r.json
    python scripts/measure_decode_speed.py --gguf f.gguf --out r.json
    python scripts/measure_decode_speed.py --quf f.quf --compare torch_gpu triton_int4 --out r.json   (task 31.1)
    python scripts/measure_decode_speed.py --quf f.quf --memory --backend triton_int4 --out r.json    (task 31.1)
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = "The quick brown fox"
NEW_TOKENS = 128


def run_quf(a) -> dict:
    from quf.bench.suite import _tokenizer_from_header
    from quf.bench.timing import time_repeats
    from quf.runtime.generate import generate
    from quf.runtime.model import QUFModel

    cpu = a.mode == "cpu"
    m = QUFModel.load(a.quf, "cpu" if cpu else "cuda", backend="cpu" if cpu else a.backend)
    tok = _tokenizer_from_header(m.header)
    ids = tok.encode(PROMPT)
    graph = None if a.mode == "graph" else False
    counts = []

    def once() -> float:
        t = time.perf_counter()
        out = generate(m, tok, ids, max_new_tokens=NEW_TOKENS, temperature=0.0, cuda_graph=graph)
        dt = time.perf_counter() - t
        counts.append(len(out))
        return len(out) / dt

    stats = time_repeats(once, warmup=2, repeats=a.repeats)
    return {"mode": a.mode, "file": Path(a.quf).name, "file_size_bytes": Path(a.quf).stat().st_size,
            "prompt": PROMPT, "new_tokens": NEW_TOKENS, "tokens_generated_per_run": sorted(set(counts)),
            **stats.to_dict()}


def run_gguf(a) -> dict:
    exe = ROOT / "eval_cache" / "llama_cpp" / "llama-bench.exe"
    p = subprocess.run([str(exe), "-m", a.gguf, "-p", "4", "-n", str(NEW_TOKENS), "-ngl", "99",
                        "-r", str(a.repeats), "-o", "json"], capture_output=True, text=True,
                       timeout=1200, cwd=str(exe.parent))
    txt = p.stdout
    rows = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
    tg = next(r for r in rows if r.get("n_gen") == NEW_TOKENS and not r.get("n_prompt"))
    s = [float(x) for x in tg["samples_ts"]]
    return {"mode": "llama.cpp Q4_K_M (-ngl 99, -p 4)", "file": Path(a.gguf).name, "new_tokens": NEW_TOKENS,
            "median": statistics.median(s), "min": min(s), "max": max(s), "repeats": len(s), "samples": s}


def _suppress_eos(m) -> None:
    """Keep generation running for the full NEW_TOKENS (the header EOS id would stop it early)."""
    if m.header.tokenizer:
        m.header.tokenizer["eos_token_id"] = None


def _timed_run(m, ids, graph):
    """One generate call. Returns (end-to-end tok/s, steady-state ms/tok, graph used, n tokens).

    Steady state is the mean gap between consecutive decode_token calls, so it excludes prefill
    and CUDA graph capture. Each gap includes the per-token argmax and host sync.
    """
    import torch

    import quf.runtime.cuda_graph as cg
    from quf.runtime.generate import generate

    stamps: list[float] = []
    used = {"graph": False}
    orig = cg.make_decoder

    class Wrap:
        def __init__(self, inner):
            self.inner = inner

        def decode_token(self, tok_id, cache):
            stamps.append(time.perf_counter())
            return self.inner.decode_token(tok_id, cache)

    def patched(model, cache, enabled=None):
        d = orig(model, cache, enabled)
        used["graph"] = d is not None
        return Wrap(d if d is not None else model)

    cg.make_decoder = patched
    try:
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = generate(m, None, ids, max_new_tokens=NEW_TOKENS, temperature=0.0, cuda_graph=graph)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
    finally:
        cg.make_decoder = orig
    steady = 1000.0 * (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    return len(out) / dt, steady, used["graph"], len(out)


def run_compare(a) -> dict:
    """Alternate backends repeat by repeat, eager and graph, on one file (all models stay loaded)."""
    from quf.bench.suite import _tokenizer_from_header
    from quf.bench.timing import summarize
    from quf.runtime.model import QUFModel

    models = {b: QUFModel.load(a.quf, "cuda", backend=b) for b in a.compare}
    tok = _tokenizer_from_header(models[a.compare[0]].header)
    ids = tok.encode(PROMPT)
    for m in models.values():
        _suppress_eos(m)
    res = {}
    for mode, graph in (("eager", False), ("graph", None)):
        e2e = {b: [] for b in a.compare}
        steady = {b: [] for b in a.compare}
        flags, counts = set(), set()
        for rnd in range(2 + a.repeats):
            for b in a.compare:
                tps, ms, used, n = _timed_run(models[b], ids, graph)
                flags.add((b, used))
                counts.add(n)
                if rnd >= 2:
                    e2e[b].append(tps)
                    steady[b].append(ms)
        for b in a.compare:
            s, st = summarize(e2e[b]), summarize(steady[b])
            key = f"{b}/{mode}"
            res[key] = {"backend": b, "mode": mode, "e2e_tok_s": s.to_dict(), "steady_ms_per_tok": st.to_dict(),
                        "steady_tok_s_median": 1000.0 / st.median,
                        "graph_replay_used": sorted(u for bb, u in flags if bb == b),
                        "tokens_generated_per_run": sorted(counts)}
            print(f"{key}: e2e {s.median:.1f} tok/s, steady {st.median:.3f} ms/tok "
                  f"({1000 / st.median:.1f} tok/s), graph={res[key]['graph_replay_used']}", flush=True)
    return {"file": Path(a.quf).name, "file_size_bytes": Path(a.quf).stat().st_size, "prompt": PROMPT,
            "new_tokens": NEW_TOKENS, "warmup": 2, "repeats": a.repeats, "eos_suppressed": True,
            "order": "backends alternate within every repeat", "results": res}


def run_memory(a) -> dict:
    """GPU memory after load and peak during a graph-mode generation, in a fresh process."""
    import torch

    from quf.bench.suite import _tokenizer_from_header
    from quf.runtime.generate import generate
    from quf.runtime.model import QUFModel

    mib = 1024 * 1024
    torch.cuda.synchronize()
    free0, _total = torch.cuda.mem_get_info()
    m = QUFModel.load(a.quf, "cuda", backend=a.backend)
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    after = {"allocated_mib": torch.cuda.memory_allocated() / mib, "reserved_mib": torch.cuda.memory_reserved() / mib,
             "device_used_delta_mib": (free0 - free1) / mib}
    tok = _tokenizer_from_header(m.header)
    _suppress_eos(m)
    torch.cuda.reset_peak_memory_stats()
    n = len(generate(m, tok, tok.encode(PROMPT), max_new_tokens=NEW_TOKENS, temperature=0.0, cuda_graph=None))
    torch.cuda.synchronize()
    free2, _ = torch.cuda.mem_get_info()
    rec = {"file": Path(a.quf).name, "backend": a.backend, "after_load": after, "tokens_generated": n,
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / mib,
           "device_used_after_generate_delta_mib": (free0 - free2) / mib,
           "note": "generation = graph mode, 128 new tokens, EOS suppressed; includes graph capture and KV cache"}
    print(json.dumps(rec), flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quf")
    ap.add_argument("--gguf")
    ap.add_argument("--mode", choices=("eager", "graph", "cpu"), default="graph")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--backend", default="torch_gpu", choices=("torch_gpu", "triton_int4", "torch_fused", "marlin"))
    ap.add_argument("--compare", nargs="+", help="alternate these backends repeat by repeat (eager and graph)")
    ap.add_argument("--memory", action="store_true", help="measure GPU memory for --backend")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.quf and (a.compare or a.memory):
        rec = run_compare(a) if a.compare else run_memory(a)
        Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return 0
    rec = run_quf(a) if a.quf else run_gguf(a)
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"{rec['mode']}: median {rec['median']:.2f} min {rec['min']:.2f} max {rec['max']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
