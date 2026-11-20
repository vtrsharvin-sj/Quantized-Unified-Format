"""Real-model verification (task 12.6, Requirement 16).

Steps per model: convert (unless the output exists), `quf run` on CPU and CUDA
with a fixed prompt and seed, `quf bench` with the fp16 reference. Also times
gigatoken against Hugging Face tokenizers on the WikiText-2 train split.

Usage:
    python scripts/verify_real_model.py --model Qwen/Qwen2.5-0.5B --out q05.quf
Results are written to docs/real_model_results_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT = "The capital of France is"
SEED = 1234


def cli(*args: str) -> tuple[int, str, float]:
    start = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-m", "quf.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, (p.stdout or "") + (p.stderr or ""), time.perf_counter() - start


def tokenizer_speed(model: str) -> dict:
    from tokenizers import Tokenizer

    from quf.bench.suite import load_wikitext

    text = load_wikitext("wikitext-2-raw-v1", "train")
    hf = Tokenizer.from_pretrained(model)
    out: dict = {"chars": len(text)}
    t = time.perf_counter()
    h = hf.encode(text, add_special_tokens=False).ids
    out["hf_seconds"] = time.perf_counter() - t
    out["tokens"] = len(h)
    try:
        import gigatoken as gt

        g = gt.Tokenizer(hf)
        t = time.perf_counter()
        ids = [int(i) for i in g.encode(text)]
        out["gigatoken_seconds"] = time.perf_counter() - t
        out["gigatoken_matches_hf"] = ids == list(h)
        out["speedup"] = out["hf_seconds"] / out["gigatoken_seconds"]
        out["gigatoken_version"] = __import__("importlib.metadata").metadata.version("gigatoken")
    except Exception as exc:  # noqa: BLE001
        out["gigatoken_error"] = repr(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output .quf path (reused if it exists)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--skip-tokenizer", action="store_true")
    a = ap.parse_args()
    tag = a.tag or a.model.split("/")[-1]
    res: dict = {"model": a.model, "samples": a.samples, "seq_len": a.seq_len}

    if not a.skip_tokenizer:
        res["tokenizer"] = tokenizer_speed(a.model)
        print("tokenizer:", res["tokenizer"], flush=True)

    if os.path.exists(a.out):
        res["convert"] = {"reused": True}
    else:
        rc, out, secs = cli(
            "convert", a.model, a.out, "--samples", str(a.samples), "--seq-len", str(a.seq_len)
        )
        res["convert"] = {"rc": rc, "seconds": secs, "tail": out[-1500:]}
        if rc != 0:
            print(out)
            return 1
    res["quf_bytes"] = os.path.getsize(a.out)

    res["run"] = {}
    for dev in ("cpu", "cuda"):
        rc, out, secs = cli(
            "run", a.out, "--prompt", PROMPT, "--max-new-tokens", "32",
            "--temperature", "0", "--seed", str(SEED), "--device", dev, "--no-stream",
        )
        res["run"][dev] = {"rc": rc, "seconds": secs, "output": out[-1500:]}
        print(dev, rc, out[-400:], flush=True)

    bench_json = os.path.join("docs", f"bench_{tag}.json")
    rc, out, secs = cli(
        "bench", a.out, "--fp16-ref", a.model, "--out", bench_json, "--device", "cuda"
    )
    res["bench"] = {"rc": rc, "seconds": secs, "tail": out[-800:]}
    if rc == 0:
        with open(bench_json, encoding="utf-8") as f:
            res["bench"]["result"] = json.load(f)
    print(res["bench"], flush=True)

    with open(os.path.join("docs", f"real_model_results_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
