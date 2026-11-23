"""Size study (task 27.1, Requirement 32.1, 32.2).

Converts the v020 preset and the Size_Candidates A, B and C, then evaluates each with
``scripts/bench_accuracy.py --protocol llama`` (same conversion, scoring and window set as
the other llama-protocol rows), and writes JSON plus a size and perplexity table.

All candidates use group size 32 and act-order, like v020. They differ only in the precision
plan (quf/convert/precision.py PRESETS):

    v020    down_proj int8 + last layer int8      (preset "sensitive")
    size_a  down_proj int8, last layer int4       (candidate A)
    size_b  last layer int8, down_proj int4       (candidate B)
    size_c  every decoder linear int4             (candidate C = A and B)

Usage (manual, each row converts and scores a model, so a full run takes a long time):

    python scripts/size_study.py --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B
    python scripts/size_study.py --models Qwen/Qwen2.5-0.5B --windows 4 --only size_c
    python scripts/size_study.py --table          # print the table from the saved JSON
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "size_study.json"
RUNS = ("v020", "size_a", "size_b", "size_c")
PROTOCOL = "llama"
DESCRIPTION = {
    "v020": "sensitive preset (down_proj and last layer int8)",
    "size_a": "A: last layer int4 g32",
    "size_b": "B: down_proj int4 g32 with act-order",
    "size_c": "C: A and B together",
}


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_accuracy", ROOT / "scripts" / "bench_accuracy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_accuracy"] = mod
    spec.loader.exec_module(mod)
    return mod


def load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"protocol": PROTOCOL, "candidates": DESCRIPTION, "models_llama": {}}


def format_table(d: dict) -> str:
    """Markdown table of size and perplexity per model and run, with the change against v020."""
    lines = ["| Model | Run | Description | File bytes | vs v020 size | wikitext2 ppl | vs v020 | c4 ppl | vs v020 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]

    def rel(a, b):
        return f"{100 * (a / b - 1):+.2f}%" if a is not None and b else "n/a"

    for model, rows in d.get("models_llama", {}).items():
        base = rows.get("v020", {}) if rows.get("v020", {}).get("status") == "done" else {}
        for name in RUNS:
            r = rows.get(name)
            if not r:
                continue
            if r.get("status") != "done":
                lines.append(f"| {model} | {name} | {DESCRIPTION[name]} | {r.get('status')} | | | | | |")
                continue
            w, c, s = r.get("ppl_wikitext2"), r.get("ppl_c4_heldout"), r.get("file_size_bytes")
            lines.append(
                f"| {model} | {name} | {DESCRIPTION[name]} | {s} | {rel(s, base.get('file_size_bytes'))} "
                f"| {w:.4f} | {rel(w, base.get('ppl_wikitext2'))} | {c:.4f} | {rel(c, base.get('ppl_c4_heldout'))} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- v040 search (task 34.4)
# name: (body preset, body group size, embed precision, embed group size, int8 lm_head stored separately)
V040 = {
    "b_e16": ("size_b", 32, "fp16", 64, False),  # = size_b, the 0.3.0 default
    "b_e8g64": ("size_b", 32, "int8", 64, False),
    "b_e8g32": ("size_b", 32, "int8", 32, False),
    "b_e4g32": ("size_b", 32, "int4", 32, False),
    "b_e4g64": ("size_b", 32, "int4", 64, False),
    "b_e4g32_h8": ("size_b", 32, "int4", 32, True),
    "b_e4g64_h8": ("size_b", 32, "int4", 64, True),
    "c_e8g64": ("size_c", 32, "int8", 64, False),
    "c_e4g32": ("size_c", 32, "int4", 32, False),
    "c_e4g32_h8": ("size_c", 32, "int4", 32, True),
    "b_bg64_e8g64": ("size_b", 64, "int8", 64, False),
    "b_bg64_e4g32": ("size_b", 64, "int4", 32, False),
    "b_bg64_e4g32_h8": ("size_b", 64, "int4", 32, True),
    # task 34.4 step 1: budget-constrained mixed precision (all attention int8, plus tail down_proj)
    "d_e8g64": ("size_d", 32, "int8", 64, False),
    "e_e8g64": ("size_e", 32, "int8", 64, False),
    "d_e4g32": ("size_d", 32, "int4", 32, False),
    "d_e4g64": ("size_d", 32, "int4", 64, False),
    "f_e8g64": ("size_f", 32, "int8", 64, False),
    "g_e8g64": ("size_g", 32, "int8", 64, False),
    "f_bg64_e8g64": ("size_f", 64, "int8", 64, False),
    # task 35.2 quantizer improvements on the b_e8g64 base; 6th field = convert_checkpoint overrides
    "q_cal256x1024": ("size_b", 32, "int8", 64, False, {"n_samples": 256, "seq_len": 1024}),
    "q_cal128x2048": ("size_b", 32, "int8", 64, False, {"n_samples": 128, "seq_len": 2048}),
    "q_cal256x2048": ("size_b", 32, "int8", 64, False, {"n_samples": 256, "seq_len": 2048}),
    "q_damp005": ("size_b", 32, "int8", 64, False, {"damp": 0.005}),
    "q_damp05": ("size_b", 32, "int8", 64, False, {"damp": 0.05}),
    "q_cal256x2048_damp05": ("size_b", 32, "int8", 64, False, {"n_samples": 256, "seq_len": 2048, "damp": 0.05}),
}


def run_v040(model: str, name: str, windows: int, protocol: str, result_path: Path) -> int:
    """Convert one v040 candidate, score it, write the row to ``result_path`` (runs in a subprocess)."""
    import gc
    import time

    ba = _load_bench()
    import quf.convert.converter as conv
    from quf.bench.suite import _tokenizer_from_header
    from quf.convert.precision import PrecisionPlan
    from quf.runtime.model import QUFModel

    preset, gs, eprec, egs, head8, *_x = V040[name]
    extra = _x[0] if _x else {}
    nl = conv.read_source_config(model).get("num_hidden_layers")
    mixed = ["lm_head=int8"] if head8 else []
    plan = PrecisionPlan.from_options(mixed, presets=[preset], num_layers=nl)
    path = ba.CACHE / f"{ba.slug(model)}_v040_{name}.quf"
    ba.CACHE.mkdir(exist_ok=True)
    t = time.perf_counter()
    conv.convert_checkpoint(model, path, group_size=gs, act_order=True, clip_search=True, overwrite=True,
                            precision_plan=plan, embed_precision=eprec, embed_group_size=egs,
                            embed_method="clip", **extra)
    print(f"converted in {time.perf_counter() - t:.0f}s", flush=True)
    gc.collect()
    m = QUFModel.load(str(path), device_option="cuda", backend="torch_gpu")
    ppls = ba._domain_ppls(m, _tokenizer_from_header(m.header), protocol, windows)
    rec = {"run": name, "status": "done", "model": model, "body_preset": preset, "body_group_size": gs,
           "embed_precision": eprec, "embed_group_size": egs if eprec != "fp16" else None,
           "lm_head_int8": head8, "overrides": extra, "protocol": protocol, "windows": windows or "all",
           "file_size_bytes": path.stat().st_size, **ppls}
    result_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("RESULT", json.dumps(rec), flush=True)
    path.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--only", nargs="+", choices=RUNS, help="run only these rows (default: all four)")
    ap.add_argument("--windows", type=int, default=0, help="score only the first N 2048-token windows (0 = all)")
    ap.add_argument("--table", action="store_true", help="print the table from the saved JSON and exit")
    ap.add_argument("--v040-run", choices=list(V040), help="(internal) convert and score one v040 candidate")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--result", help="(internal) result JSON path for --v040-run")
    a = ap.parse_args(argv)
    if a.v040_run:
        return run_v040(a.model, a.v040_run, a.windows, PROTOCOL, Path(a.result))
    if a.table:
        print(format_table(load()))
        return 0

    ba = _load_bench()
    ba.OUT = OUT  # keep study rows out of docs/bench_accuracy.json
    d = load()
    d.setdefault("models", {})  # bench_accuracy._finalize expects this key
    d["environment"] = ba._env()
    d["windows"] = a.windows or "all"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2), encoding="utf-8")
    for model in a.models:
        for name in RUNS:
            if a.only and name not in a.only:
                continue
            # bench_accuracy.launch runs `--run NAME --protocol llama` in a subprocess and merges the row
            ba.launch(model, name, d, PROTOCOL, a.windows)
            OUT.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(format_table(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
