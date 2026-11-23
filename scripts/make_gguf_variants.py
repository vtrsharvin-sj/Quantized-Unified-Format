"""Produce the GGUF variants for the 0.4.0 comparison (task 36.1). Writes under eval_cache/gguf040/.

Steps per model: Q4_K_S, imatrix (WikiText-2 train, 128 x 1024 tokens), IQ4_XS+imatrix,
Q4_K_M+imatrix, Q5_K_M. Existing f16 and non-imatrix Q4_K_M from task 20 are reused.
Each step logs to its own file; the commands and sizes land in docs/gguf_variants.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "eval_cache"
OUT = CACHE / "gguf040"
BIN = CACHE / "llama_cpp"
CALIB = OUT / "calib_wikitext2_train_128x1024.txt"
MODELS = {"qwen2_5-0_5b": "Qwen/Qwen2.5-0.5B", "qwen2_5-1_5b": "Qwen/Qwen2.5-1.5B"}
TIMEOUT = 20 * 60
IMATRIX_ARGS = ["-c", "1024", "--chunks", "128", "-ngl", "99"]


def run(cmd: list[str], log: Path) -> dict:
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=TIMEOUT)
    return {"command": " ".join(cmd), "log": str(log.relative_to(ROOT)),
            "returncode": p.returncode, "seconds": round(time.time() - t0, 1)}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    record: dict = {"llama_cpp": "b12567 CUDA build (eval_cache/llama_cpp)",
                    "calibration": {"source": "WikiText-2 train (Salesforce/wikitext wikitext-2-raw-v1)",
                                    "tokens": "128 x 1024 = 131072, the QUF calibration tokens "
                                              "(eval_cache/calib_wikitext_128x1024.npy) decoded with the Qwen2.5 tokenizer",
                                    "text_file": rel(CALIB), "imatrix_args": IMATRIX_ARGS},
                    "models": {}}
    q, im = str(BIN / "llama-quantize.exe"), str(BIN / "llama-imatrix.exe")
    for name in MODELS:
        f16 = CACHE / f"{name}-f16.gguf"
        steps: list[tuple[str, Path, list[str]]] = []
        imat = OUT / f"{name}.imatrix.gguf"
        steps.append(("imatrix", imat, [im, "-m", str(f16), "-f", str(CALIB), "-o", str(imat), *IMATRIX_ARGS]))
        for tag, use_imat in [("Q4_K_S", False), ("IQ4_XS", True), ("Q4_K_M", True), ("Q5_K_M", False)]:
            suffix = "-imat" if (use_imat and tag == "Q4_K_M") else ""
            dst = OUT / f"{name}-{tag}{suffix}.gguf"
            cmd = [q]
            if use_imat:
                cmd += ["--imatrix", str(imat)]
            cmd += [str(f16), str(dst), tag]
            steps.append((f"{tag}{suffix}", dst, cmd))
        m: dict = {"f16_source": rel(f16), "files": {}}
        base = CACHE / f"{name}-Q4_K_M.gguf"
        m["files"]["Q4_K_M (no imatrix, reused from task 20)"] = {"path": rel(base), "bytes": base.stat().st_size}
        for label, dst, cmd in steps:
            if not (dst.exists() and dst.stat().st_size > 0):
                r = run(cmd, OUT / f"{name}.{label}.log")
            else:
                r = {"command": " ".join(cmd), "note": "already existed"}
            if dst.exists() and dst.stat().st_size > 0:
                r.update(path=rel(dst), bytes=dst.stat().st_size)
            else:
                r["error"] = "output missing"
            m["files"][label] = r
            print(name, label, r.get("bytes"), r.get("returncode"), flush=True)
            if "error" in r:
                record["models"][name] = m
                (ROOT / "docs" / "gguf_variants.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                return 1
        record["models"][name] = m
    (ROOT / "docs" / "gguf_variants.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
