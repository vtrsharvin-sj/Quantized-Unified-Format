"""Measure decode tokens/sec and memory for the cpu, torch_gpu and torch_fused backends.

Each backend runs in its own subprocess so memory numbers are not polluted by earlier runs.

    python scripts/bench_backends.py model.quf [--tokens 64] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BACKENDS = ("cpu", "torch_gpu", "torch_fused")
PROMPT = "The quick brown fox jumps over the lazy dog. In a surprising turn of events,"


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return 0


def measure_one(path: str, backend: str, n_tokens: int) -> dict:
    import torch

    from quf.bench.suite import _tokenizer_from_header, software_versions
    from quf.runtime.model import QUFModel

    device = "cpu" if backend == "cpu" else "cuda"
    on_gpu = device == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model = QUFModel.load(path, device, backend=backend)
    if on_gpu:
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    tok = _tokenizer_from_header(model.header)
    prompt_ids = tok.encode(PROMPT)

    def sync():
        if on_gpu:
            torch.cuda.synchronize()

    def run(n: int):
        """Greedy decode of exactly n tokens (EOS ignored). Returns (prefill_s, decode_s, ids)."""
        cache = model.new_cache()
        ids = torch.tensor([prompt_ids], device=model.device)
        sync()
        t = time.perf_counter()
        logits = model.forward(ids, cache)
        nxt = int(logits[0, -1].argmax())
        sync()
        prefill = time.perf_counter() - t
        out = [nxt]
        t = time.perf_counter()
        for _ in range(n - 1):
            logits = model.forward(torch.tensor([[nxt]], device=model.device), cache)
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
        sync()
        return prefill, time.perf_counter() - t, out

    run(8)  # warm-up (compile, allocator, kernels)
    prefill_s, decode_s, out = run(n_tokens)

    res = {
        "backend": backend,
        "tokens": n_tokens,
        "prompt_tokens": len(prompt_ids),
        "load_seconds": load_s,
        "prefill_seconds": prefill_s,
        "decode_seconds": decode_s,
        "decode_tokens_per_second": (n_tokens - 1) / decode_s if decode_s > 0 else 0.0,
        "first_tokens": out[:16],
        "cpu_rss_bytes": _rss_bytes(),
    }
    if on_gpu:
        res["gpu_allocated_bytes"] = int(torch.cuda.memory_allocated())
        res["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated())
        res["device_name"] = torch.cuda.get_device_name()
        res["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability()))
    res["versions"] = software_versions()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("quf_file")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    ap.add_argument("--one", default=None, help="internal: run a single backend in this process")
    args = ap.parse_args(argv)

    if args.one:
        print("RESULT " + json.dumps(measure_one(args.quf_file, args.one, args.tokens)))
        return 0

    results = []
    for b in args.backends:
        proc = subprocess.run(
            [sys.executable, __file__, args.quf_file, "--tokens", str(args.tokens), "--one", b],
            capture_output=True, text=True,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if proc.returncode != 0 or line is None:
            print(f"{b}: FAILED\n{proc.stderr[-1500:]}", file=sys.stderr)
            results.append({"backend": b, "error": proc.stderr[-1500:]})
            continue
        r = json.loads(line[7:])
        results.append(r)
        gpu = f", gpu peak {r['gpu_peak_bytes'] / 2**20:.0f} MiB" if "gpu_peak_bytes" in r else ""
        print(f"{b:12s} {r['decode_tokens_per_second']:8.1f} tok/s  prefill {r['prefill_seconds']*1000:.0f} ms  "
              f"load {r['load_seconds']:.1f} s  rss {r['cpu_rss_bytes'] / 2**20:.0f} MiB{gpu}")

    ok = {r["backend"]: r for r in results if "error" not in r}
    decision = None
    if "torch_gpu" in ok and "torch_fused" in ok:
        a, f = ok["torch_gpu"], ok["torch_fused"]
        fused_faster = f["decode_tokens_per_second"] > a["decode_tokens_per_second"]
        decision = {
            "fused_faster": fused_faster,
            "default_backend": "torch_fused" if fused_faster else "torch_gpu",
            "same_greedy_tokens": a["first_tokens"] == f["first_tokens"],
            "note": "selector uses prefer_fused only when fused decode is faster; otherwise cached is default",
        }
        print("decision:", decision)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"results": results, "decision": decision}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
