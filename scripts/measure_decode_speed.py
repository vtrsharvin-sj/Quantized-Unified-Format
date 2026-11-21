"""Task 26.8: median decode speed of a QUF file (eager vs CUDA graph, CPU) and llama-bench.

    python scripts/measure_decode_speed.py --quf f.quf --mode eager|graph|cpu --out r.json
    python scripts/measure_decode_speed.py --gguf f.gguf --out r.json
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
    m = QUFModel.load(a.quf, "cpu" if cpu else "cuda", backend="cpu" if cpu else "torch_gpu")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quf")
    ap.add_argument("--gguf")
    ap.add_argument("--mode", choices=("eager", "graph", "cpu"), default="graph")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = run_quf(a) if a.quf else run_gguf(a)
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"{rec['mode']}: median {rec['median']:.2f} min {rec['min']:.2f} max {rec['max']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
