"""Task 33.5: QUF 0.4.0 vs 0.3.0 vs llama.cpp decode speed, alternating runs, with power state.
    python scripts/bench_speed040.py --size 05|15 --out eval_cache/speed040/quf05.json
    python scripts/bench_speed040.py --size 05|15 --gguf --out eval_cache/speed040/gguf05.json
Warm-up 2, median of 7, 128 new tokens, EOS suppressed. All numbers are power-capped.
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
Q = "power.draw,clocks.sm,clocks_throttle_reasons.active,temperature.gpu"
class PowerSampler:
    def __init__(self):
        self.rows, self._stop = [], threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._stop.is_set():
            r = subprocess.run(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().split(",")
            try:
                self.rows.append((float(r[0]), float(r[1]), r[2].strip(), float(r[3])))
            except (ValueError, IndexError):
                pass
            self._stop.wait(0.5)
    def __enter__(self):
        self.t.start()
        return self
    def __exit__(self, *a):
        self._stop.set()
        self.t.join()
    def summary(self):
        busy = [r for r in self.rows if r[1] > 300] or self.rows
        reasons = {}
        for r in busy:
            reasons[r[2]] = reasons.get(r[2], 0) + 1
        return {"samples": len(self.rows), "power_w_median": statistics.median(r[0] for r in busy),
                "power_w_max": max(r[0] for r in busy), "sm_mhz_median": statistics.median(r[1] for r in busy),
                "sm_mhz_max": max(r[1] for r in busy), "temp_c_max": max(r[3] for r in busy),
                "throttle_reason_counts_hex": reasons, "label": "power-capped"}
def run_quf(a):
    from quf.bench.suite import _tokenizer_from_header
    from quf.runtime.model import QUFModel
    import measure_decode_speed as M
    s = a.size
    cfgs = {  # name -> (file, QUF_NO_FUSED)
        "quf040": (f"eval_cache/lm040/q{s}.quf", ""),
        "quf030_file_current_code": (f"eval_cache/lm/sb{s}.quf", ""),
        "quf030_path": (f"eval_cache/lm/sb{s}.quf", "1"),
    }
    models = {}
    for k, (f, nf) in cfgs.items():
        os.environ["QUF_NO_FUSED"] = nf
        models[k] = QUFModel.load(str(ROOT / f), "cuda", backend="triton_int4")
        M._suppress_eos(models[k])
    os.environ.pop("QUF_NO_FUSED", None)
    tok = _tokenizer_from_header(models["quf040"].header)
    ids = tok.encode(M.PROMPT)
    e2e = {k: [] for k in cfgs}
    st = {k: [] for k in cfgs}
    info = {k: {"fused_decode": bool(m.fused_decode), "file": cfgs[k][0],
                "file_size_bytes": (ROOT / cfgs[k][0]).stat().st_size} for k, m in models.items()}
    with PowerSampler() as ps:
        for rnd in range(2 + a.repeats):
            for k, m in models.items():
                tps, ms, used, n = M._timed_run(m, ids, None)
                assert used and n == M.NEW_TOKENS, (k, used, n)
                if rnd >= 2:
                    e2e[k].append(tps)
                    st[k].append(ms)
    out = {"size": s, "prompt": M.PROMPT, "new_tokens": 128, "warmup": 2, "repeats": a.repeats,
           "mode": "graph", "backend": "triton_int4", "power": ps.summary(), "results": {}}
    for k in cfgs:
        out["results"][k] = {**info[k], "e2e_tok_s_median": statistics.median(e2e[k]), "e2e_samples": e2e[k],
                             "steady_ms_per_tok_median": statistics.median(st[k]), "steady_samples": st[k],
                             "steady_tok_s_median": 1000 / statistics.median(st[k])}
        print(k, out["results"][k]["e2e_tok_s_median"], out["results"][k]["steady_ms_per_tok_median"], flush=True)
    return out
def run_gguf(a):
    s = a.size
    g = {"Q4_K_M": f"eval_cache/qwen2_5-{s[0]}_{s[1]}b-Q4_K_M.gguf" if False else None}
    name = "0_5b" if s == "05" else "1_5b"
    files = {"Q4_K_M": f"eval_cache/qwen2_5-{name}-Q4_K_M.gguf"}
    for v in ("Q4_K_S", "IQ4_XS", "Q4_K_M-imat", "Q5_K_M"):
        files[v] = f"eval_cache/gguf040/qwen2_5-{name}-{v}.gguf"
    exe = ROOT / "eval_cache" / "llama_cpp" / "llama-bench.exe"
    samples = {k: [] for k in files}
    with PowerSampler() as ps:
        for rnd in range(2 + a.repeats):
            for k, f in files.items():
                p = subprocess.run([str(exe), "-m", str(ROOT / f), "-p", "0", "-n", "128", "-ngl", "99",
                                    "-r", "1", "-o", "json"], capture_output=True, text=True, cwd=str(exe.parent),
                                   timeout=900)
                t = p.stdout
                rows = json.loads(t[t.index("["): t.rindex("]") + 1])
                tg = next(r for r in rows if r.get("n_gen") == 128)
                if rnd >= 2:
                    samples[k].append(float(tg["samples_ts"][0]))
    out = {"size": s, "new_tokens": 128, "warmup": "2 alternating rounds plus llama-bench's own warm-up",
           "repeats": a.repeats, "ngl": 99, "power": ps.summary(), "results": {}}
    for k, f in files.items():
        out["results"][k] = {"file": f, "file_size_bytes": (ROOT / f).stat().st_size,
                             "tg128_tok_s_median": statistics.median(samples[k]), "samples": samples[k]}
        print(k, out["results"][k]["tg128_tok_s_median"], flush=True)
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=("05", "15"))
    ap.add_argument("--gguf", action="store_true")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = run_gguf(a) if a.gguf else run_quf(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()
