"""Task 36.2: llama-protocol perplexity for GGUF variants and re-converted QUF 0.4.0 files.

Each evaluation is its own subprocess with its own log under eval_cache/pareto040/ and a
25-minute cap. Each result is merged into docs/bench_accuracy.json["pareto_040"] on completion.
Finished entries are skipped, so the script can be re-run.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "eval_cache"
EXE = CACHE / "llama_cpp" / "llama-perplexity.exe"
OUT = ROOT / "docs" / "bench_accuracy.json"
LOGS = CACHE / "pareto040"
CAP = 25 * 60
TEXT = {"wikitext2": CACHE / "ppl_text_wikitext2.txt", "c4": CACHE / "ppl_text_c4.txt"}
MODELS = {"0.5B": ("qwen2_5-0_5b", "Qwen/Qwen2.5-0.5B", "q05"),
          "1.5B": ("qwen2_5-1_5b", "Qwen/Qwen2.5-1.5B", "q15")}
GGUFS = ["Q4_K_S", "IQ4_XS", "Q4_K_M-imat", "Q5_K_M"]


def save(model: str, row: str, rec: dict) -> None:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    d.setdefault("pareto_040", {}).setdefault(model, {})[row] = rec
    OUT.write_text(json.dumps(d, indent=2), encoding="utf-8")


def done(model: str, row: str, keys: tuple[str, ...]) -> bool:
    d = json.loads(OUT.read_text(encoding="utf-8")).get("pareto_040", {}).get(model, {}).get(row, {})
    return all(k in d for k in keys)


def run(cmd: list[str], log: Path, cwd: Path) -> tuple[int, str]:
    t = time.perf_counter()
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd))
        try:
            rc = p.wait(timeout=CAP)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -999
    print(f"  rc={rc} {time.perf_counter() - t:.0f}s {log.name}", flush=True)
    return rc, log.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    LOGS.mkdir(exist_ok=True)
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for size, (slug, hf, qtag) in MODELS.items():
        base = d["models_llama"][hf]["gguf_q4_k_m"]
        save(size, "Q4_K_M", {"ppl_wikitext2": base["ppl_wikitext2"], "ppl_c4_heldout": base["ppl_c4_heldout"],
                              "file_size_bytes": base["file_size_bytes"],
                              "source": "reused from models_llama.gguf_q4_k_m (task 20.3)"})
        for v in GGUFS:
            row = v
            if done(size, row, ("ppl_wikitext2", "ppl_c4_heldout")):
                continue
            path = CACHE / "gguf040" / f"{slug}-{v}.gguf"
            rec = {"file": path.name, "file_size_bytes": path.stat().st_size, "protocol": "llama",
                   "backend": "llama-perplexity.exe (cuda, -ngl 99)"}
            for tag, key in (("wikitext2", "ppl_wikitext2"), ("c4", "ppl_c4_heldout")):
                print(f"{size} {v} {tag}", flush=True)
                rc, out = run([str(EXE), "-m", str(path), "-f", str(TEXT[tag]), "-c", "2048", "-ngl", "99"],
                              LOGS / f"{slug}-{v}.{tag}.log", EXE.parent)
                m = re.search(r"Final estimate: PPL = ([0-9.]+)", out)
                if not m:
                    rec[key + "_error"] = f"rc={rc}"
                    break
                rec[key] = float(m.group(1))
            save(size, row, rec)
        if not done(size, "QUF", ("ppl_wikitext2", "ppl_c4_heldout")):
            q = CACHE / "lm040" / f"{qtag}.quf"
            res = LOGS / f"{qtag}.quf.result.json"
            print(f"{size} QUF", flush=True)
            rc, _ = run([sys.executable, "scripts/bench_accuracy.py", "--ppl-quf", str(q), "--protocol", "llama",
                         "--out", str(res)], LOGS / f"{qtag}.quf.log", ROOT)
            rec = json.loads(res.read_text(encoding="utf-8")) if res.exists() and rc == 0 else {"error": f"rc={rc}"}
            rec.update({"file_size_bytes": q.stat().st_size, "protocol": "llama"})
            save(size, "QUF", rec)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
