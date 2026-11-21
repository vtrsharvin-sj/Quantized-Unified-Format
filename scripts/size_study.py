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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--only", nargs="+", choices=RUNS, help="run only these rows (default: all four)")
    ap.add_argument("--windows", type=int, default=0, help="score only the first N 2048-token windows (0 = all)")
    ap.add_argument("--table", action="store_true", help="print the table from the saved JSON and exit")
    a = ap.parse_args(argv)
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
