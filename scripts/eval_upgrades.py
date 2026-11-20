"""Evaluate each Upgrade by WikiText-2 perplexity (task 16.8, Requirements 23.5, 24.1, 24.2).

Every run converts Qwen2.5-0.5B with the 0.1.0 settings (group size 128, no act-order,
no clip search, int4 everywhere, wikitext-2 train calibration 128 x 1024) plus the
Upgrade under test, then measures perplexity, file size and tokens/s with the same
code as ``quf bench``. Calibration tokens are cached on disk and reused.

Driver (each run is a subprocess with a time cap, results are merged into
docs/upgrade_eval.json after every run):

    python scripts/eval_upgrades.py --drive

Single run (what the driver launches):

    python scripts/eval_upgrades.py --run g64 --group-size 64
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = "Qwen/Qwen2.5-0.5B"
FP16_PPL = 13.0703
BASE_PPL = 14.0384  # 0.1.0 settings, 128 x 1024 wikitext calibration (docs/RESULTS.md)
CACHE = ROOT / "eval_cache"
OUT = ROOT / "docs" / "upgrade_eval.json"
TIME_CAP_S = 25 * 60
MEASURABLE = 0.02  # ppl; an Upgrade must beat the reference by more than this to count


# ----------------------------------------------------------------------------- worker
def _calibration(kind: str):
    import numpy as np

    CACHE.mkdir(exist_ok=True)
    if kind == "baseline":
        path, n, seq = CACHE / "calib_wikitext_128x1024.npy", 128, 1024
    else:
        path, n, seq = CACHE / "calib_new_128x2048.npy", 128, 2048
    if path.exists():
        return np.load(path).tolist()
    from quf.convert.converter import (
        _hf_from_info, build_tokenizer_info, default_calibration_sources,
        load_calibration_texts, read_source_config,
    )
    from quf.tokenization.calibration import build_calibration
    from quf.tokenization.wrapper import TokenizerWrapper

    cfg = read_source_config(MODEL)
    tok = TokenizerWrapper(None, hf_tokenizer=_hf_from_info(build_tokenizer_info(MODEL, cfg), MODEL))
    src = load_calibration_texts("wikitext-2-raw-v1") if kind == "baseline" else default_calibration_sources()
    calib = build_calibration(src, tok, n, seq, 0)
    np.save(path, np.asarray(calib, dtype=np.int64))
    return calib


def run_one(a) -> int:
    import gc

    import quf.convert.converter as conv
    from quf.convert.precision import PrecisionPlan

    t0 = time.perf_counter()
    calib = _calibration(a.calib)
    print(f"calibration ready: {len(calib)} x {len(calib[0])} ({time.perf_counter()-t0:.0f}s)", flush=True)
    conv.build_calibration = lambda *args, **kw: calib  # reuse the cached tokens

    plan = None
    if a.preset:
        plan = PrecisionPlan.from_options(presets=[a.preset], num_layers=24)
    CACHE.mkdir(exist_ok=True)
    quf_path = CACHE / f"{a.run}.quf"
    t1 = time.perf_counter()
    res = conv.convert_checkpoint(
        MODEL, quf_path, texts=["x"], overwrite=True, group_size=a.group_size,
        act_order=a.act_order, clip_search=a.clip_search, precision_plan=plan,
        n_samples=len(calib), seq_len=len(calib[0]),
    )
    convert_s = time.perf_counter() - t1
    print(f"converted in {convert_s:.0f}s", flush=True)
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    from quf.bench.suite import run_bench

    t2 = time.perf_counter()
    bench_json = CACHE / f"{a.run}.bench.json"
    r = run_bench(quf_path, out_json=bench_json, device="cuda")
    bench_s = time.perf_counter() - t2
    out = {
        "run": a.run, "status": "done", "group_size": a.group_size, "act_order": a.act_order,
        "clip_search": a.clip_search, "preset": a.preset, "calibration": a.calib,
        "perplexity": r.perplexity, "increase_pct": 100 * (r.perplexity / FP16_PPL - 1),
        "file_size_bytes": r.file_size_bytes, "tokens_per_second": r.tokens_per_second,
        "backend": r.backend, "convert_seconds": round(convert_s), "bench_seconds": round(bench_s),
        "size_report": res.report.format() if res.report else None,
    }
    (CACHE / f"{a.run}.result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("RESULT", json.dumps(out), flush=True)
    quf_path.unlink(missing_ok=True)
    return 0


# ----------------------------------------------------------------------------- driver
def _load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"model": MODEL, "fp16_perplexity": FP16_PPL, "reference_perplexity_0_1_0": BASE_PPL,
            "time_cap_seconds": TIME_CAP_S, "runs": {}}


def _save(d: dict) -> None:
    OUT.write_text(json.dumps(d, indent=2), encoding="utf-8")


def launch(name: str, d: dict, *, group_size=128, act_order=False, clip=False, preset=None, calib="baseline"):
    """Run one configuration in a subprocess with a time cap; skip it if already recorded."""
    if name in d["runs"] and d["runs"][name].get("status") in ("done", "timeout"):
        return d["runs"][name]
    cmd = [sys.executable, "-u", str(Path(__file__)), "--run", name, "--group-size", str(group_size),
           "--calib", calib]
    if act_order:
        cmd.append("--act-order")
    if clip:
        cmd.append("--clip-search")
    if preset:
        cmd += ["--preset", preset]
    log = CACHE / f"{name}.log"
    CACHE.mkdir(exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] start {name}: {' '.join(cmd[3:])}", flush=True)
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
        t = time.time()
        while p.poll() is None and time.time() - t < TIME_CAP_S:
            time.sleep(5)
        if p.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            p.wait()
            rec = {"run": name, "status": "timeout", "group_size": group_size, "act_order": act_order,
                   "clip_search": clip, "preset": preset, "calibration": calib,
                   "note": f"not completed: killed after {TIME_CAP_S // 60} min"}
        else:
            rf = CACHE / f"{name}.result.json"
            if p.returncode == 0 and rf.exists():
                rec = json.loads(rf.read_text(encoding="utf-8"))
            else:
                rec = {"run": name, "status": "error", "returncode": p.returncode,
                       "note": "see eval_cache/" + name + ".log"}
    d["runs"][name] = rec
    _save(d)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {rec.get('status')} ppl={rec.get('perplexity')}", flush=True)
    return rec


def _ppl(d, name):
    r = d["runs"].get(name)
    return r["perplexity"] if r and r.get("status") == "done" else None


def drive() -> int:
    d = _load()
    # Reference: re-measured here with the same calibration, so every comparison is like for like.
    launch("baseline", d)
    base = _ppl(d, "baseline") or BASE_PPL

    def better(name, ref):
        p = _ppl(d, name)
        return p is not None and p < ref - MEASURABLE

    for name, kw in (("g64", dict(group_size=64)), ("g32", dict(group_size=32)),
                     ("clip", dict(clip=True)), ("actorder", dict(act_order=True))):
        launch(name, d, **kw)
    kept_gs = [g for g in (64, 32) if better(f"g{g}", base)]
    best_gs = min(kept_gs, key=lambda g: _ppl(d, f"g{g}")) if kept_gs else 128
    kw = dict(group_size=best_gs, clip=better("clip", base), act_order=better("actorder", base))
    # combinations
    if best_gs != 128:
        launch(f"g{best_gs}_clip", d, group_size=best_gs, clip=True)
        launch(f"g{best_gs}_actorder", d, group_size=best_gs, act_order=True)
    launch("clip_actorder", d, clip=True, act_order=True)
    combos = {"all3": dict(group_size=best_gs, clip=True, act_order=True)}
    for k, v in combos.items():
        launch(f"{k}_g{best_gs}", d, **v)
    # pick best configuration so far
    cands = {n: r for n, r in d["runs"].items() if r.get("status") == "done" and n != "baseline"
             and r.get("preset") is None and r.get("calibration") == "baseline"}
    best = min(cands, key=lambda n: cands[n]["perplexity"])
    b = cands[best]
    d["best_combo_before_mixed"] = best
    _save(d)
    cfg = dict(group_size=b["group_size"], clip=b["clip_search"], act_order=b["act_order"])
    launch("best_down_int8", d, preset="down_proj", **cfg)
    launch("best_sensitive", d, preset="sensitive", **cfg)
    launch("best_newcalib", d, calib="new", **cfg)
    print("driver finished", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--act-order", action="store_true")
    ap.add_argument("--clip-search", action="store_true")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--calib", choices=("baseline", "new"), default="baseline")
    a = ap.parse_args()
    if a.drive:
        return drive()
    if a.run:
        return run_one(a)
    ap.error("use --drive or --run NAME")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
