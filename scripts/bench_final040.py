"""Task 39.5 part B: interleaved QUF 0.4.0 vs llama.cpp decode speed in ONE session.
    python scripts/bench_final040.py [--repeats 7] [--warmup 2]
Each round runs 12 rows (2 models x [QUF, Q4_K_M, Q4_K_S, IQ4_XS-imat, Q5_K_M]) in a rotated order.
QUF: triton_int4, CUDA graph, 128 new tokens, EOS suppressed, prompt of 33.5. llama-bench: -p 0 -n 128 -ngl 99 -r 1.
Writes docs/bench_accuracy.json["speed_040_final"] and the docs/BACKENDS.md section.
"""
from __future__ import annotations
import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_speed040 import PowerSampler  # noqa: E402

SEC = "Final decode speed, QUF 0.4.0 vs llama.cpp (task 39.5)"
EXE = ROOT / "eval_cache" / "llama_cpp" / "llama-bench.exe"
GG = {"Q4_K_M": "eval_cache/qwen2_5-{n}-Q4_K_M.gguf", "Q4_K_S": "eval_cache/gguf040/qwen2_5-{n}-Q4_K_S.gguf",
      "IQ4_XS-imat": "eval_cache/gguf040/qwen2_5-{n}-IQ4_XS.gguf", "Q5_K_M": "eval_cache/gguf040/qwen2_5-{n}-Q5_K_M.gguf"}
NAMES = {"05": "0_5b", "15": "1_5b"}


def power_source():
    def ps(c):
        return subprocess.run(["powershell", "-NoProfile", "-Command", c], capture_output=True, text=True).stdout.strip()
    online = ps("(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus).PowerOnline")
    return {"power_online_ac": online.lower() == "true", "raw": online,
            "battery_pct": ps("(Get-CimInstance Win32_Battery).EstimatedChargeRemaining")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()
    import measure_decode_speed as M
    from quf.bench.suite import _tokenizer_from_header
    from quf.runtime.model import QUFModel
    pw = power_source()
    label = "AC power" if pw["power_online_ac"] else "BATTERY (not on AC)"
    print("power source:", pw, label, flush=True)
    quf = {}
    for s in NAMES:
        quf[s] = QUFModel.load(str(ROOT / f"eval_cache/lm040/q{s}.quf"), "cuda", backend="triton_int4")
        M._suppress_eos(quf[s])
    tok = _tokenizer_from_header(quf["05"].header)
    ids = tok.encode(M.PROMPT)
    rows = [(s, e) for s in NAMES for e in ["QUF", *GG]]

    def run(s, e):
        with PowerSampler() as p:
            if e == "QUF":
                tps, ms, used, n = M._timed_run(quf[s], ids, None)
                assert used and n == M.NEW_TOKENS, (s, used, n)
                r = {"steady": 1000.0 / ms, "e2e": tps}
            else:
                f = ROOT / GG[e].format(n=NAMES[s])
                o = subprocess.run([str(EXE), "-m", str(f), "-p", "0", "-n", "128", "-ngl", "99", "-r", "1", "-o", "json"],
                                   capture_output=True, text=True, cwd=str(EXE.parent), timeout=900).stdout
                js = json.loads(o[o.index("["): o.rindex("]") + 1])
                r = {"steady": float(next(x for x in js if x.get("n_gen") == 128)["samples_ts"][0])}
        sm = p.summary()
        r.update(power_w=sm["power_w_median"], sm_mhz=sm["sm_mhz_median"], throttle=sm["throttle_reason_counts_hex"])
        return r

    data = {k: [] for k in rows}
    t0 = time.time()
    for rnd in range(a.warmup + a.repeats):
        k = rnd % len(rows)
        order = rows[k:] + rows[:k]
        for s, e in order:
            r = run(s, e)
            print(f"round {rnd} {s} {e} {r['steady']:.1f} tok/s {r['power_w']:.1f} W {r['sm_mhz']:.0f} MHz t={time.time()-t0:.0f}s", flush=True)
            if rnd >= a.warmup:
                data[(s, e)].append(r)

    def agg(rs, key):
        v = [x[key] for x in rs]
        return {"median": st.median(v), "min": min(v), "max": max(v)}

    res = {}
    for (s, e), rs in data.items():
        th = {}
        for x in rs:
            for h, c in x["throttle"].items():
                th[h] = th.get(h, 0) + c
        d = {"steady_tok_s": agg(rs, "steady"), "samples_steady": [x["steady"] for x in rs],
             "power_w_median": st.median(x["power_w"] for x in rs), "sm_mhz_median": st.median(x["sm_mhz"] for x in rs),
             "throttle_reason_counts_hex": th}
        if e == "QUF":
            d["e2e_tok_s"] = agg(rs, "e2e")
        res.setdefault(s, {})[e] = d

    def verdict(q, o):
        if q["min"] > o["max"]:
            return "faster"
        if q["max"] < o["min"]:
            return "slower"
        return "equal within noise"

    for s in NAMES:
        q = res[s]["QUF"]["steady_tok_s"]
        for e in GG:
            res[s][e]["quf_vs_this"] = verdict(q, res[s][e]["steady_tok_s"])
            res[s][e]["quf_median_ratio"] = q["median"] / res[s][e]["steady_tok_s"]["median"]
    t05 = res["05"]["QUF"]["steady_tok_s"]["median"] >= res["05"]["Q4_K_M"]["steady_tok_s"]["median"]
    t15 = res["15"]["QUF"]["steady_tok_s"]["median"] >= 398
    out = {"power_source": pw, "label": label, "prompt": M.PROMPT, "new_tokens": 128, "warmup_rounds": a.warmup,
           "measured_rounds": a.repeats, "order": "rotated one row per round", "quf_backend": "triton_int4",
           "mode": "CUDA graph", "llama_bench_args": "-p 0 -n 128 -ngl 99 -r 1", "wall_s": time.time() - t0,
           "results": {"0.5B": res["05"], "1.5B": res["15"]},
           "targets": {"0.5B QUF median >= Q4_K_M median": t05, "1.5B QUF median >= 398 tok/s": t15,
                       "1.5B vs Q4_K_M": res["15"]["Q4_K_M"]["quf_vs_this"]}}
    dj = ROOT / "docs" / "bench_accuracy.json"
    doc = json.loads(dj.read_text(encoding="utf-8"))
    doc["speed_040_final"] = out
    dj.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    L = [f"## {SEC}", "",
         f"Power source during the run: **{label}** (PowerOnline={pw['raw']}, battery {pw['battery_pct']}%). "
         f"All numbers below are {'AC' if pw['power_online_ac'] else 'battery'} numbers.", "",
         f"One session, rows interleaved in rotated order, {a.warmup} warm-up rounds discarded, {a.repeats} measured rounds. "
         "QUF: triton_int4, CUDA graph, 128 new tokens, EOS suppressed, prompt \"The quick brown fox\". "
         "llama-bench: `-p 0 -n 128 -ngl 99 -r 1`. Compared figure is QUF steady tok/s (no prefill, no graph capture) "
         "against llama-bench tg128. Verdict: faster/slower when the min-max ranges do not overlap, otherwise equal within noise.", "",
         "| Model | Engine | tok/s median | min | max | QUF e2e median | power W | SM MHz | QUF vs row |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s, mn in (("05", "0.5B"), ("15", "1.5B")):
        for e, d in res[s].items():
            t = d["steady_tok_s"]
            L.append(f"| {mn} | {'QUF 0.4.0' if e == 'QUF' else e} | {t['median']:.1f} | {t['min']:.1f} | {t['max']:.1f} | "
                     f"{d['e2e_tok_s']['median']:.1f} | " if e == "QUF" else
                     f"| {mn} | {e} | {t['median']:.1f} | {t['min']:.1f} | {t['max']:.1f} | - | ")
            L[-1] += f"{d['power_w_median']:.1f} | {d['sm_mhz_median']:.0f} | " + \
                     ("-" if e == "QUF" else f"{d['quf_vs_this']} ({d['quf_median_ratio']:.2f}x)") + " |"
    L += ["", f"Targets: 0.5B QUF >= Q4_K_M: **{'met' if t05 else 'not met'}**; "
          f"1.5B QUF >= 398 tok/s: **{'met' if t15 else 'not met'}**; 1.5B QUF vs Q4_K_M: {res['15']['Q4_K_M']['quf_vs_this']}.", ""]
    bp = ROOT / "docs" / "BACKENDS.md"
    txt = bp.read_text(encoding="utf-8")
    if f"## {SEC}" in txt:
        txt = txt[: txt.index(f"## {SEC}")].rstrip() + "\n"
    bp.write_text(txt.rstrip() + "\n\n" + "\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
