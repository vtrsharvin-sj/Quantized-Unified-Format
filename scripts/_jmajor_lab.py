"""Task 39.5 lab: tune _gemv_jmajor_kernel for the in-model shapes (not a shipped tool).

    python scripts/_jmajor_lab.py micro --quf eval_cache/lm040/q15.quf --out eval_cache/speed040/jm_micro_q15.json
    python scripts/_jmajor_lab.py graph --quf eval_cache/lm040/q15.quf --cfg "17920x1536=2,16,1,2;1536x8960=1,32,2,2"

``micro`` times each candidate config per shape on cold weights (one graph that walks all layers' weights).
``graph`` loads the model with the given per-shape configs and reports whole-graph steady ms/token (median of 5,
alternating is left to the caller: run it once per candidate set).
"""
from __future__ import annotations
import argparse, itertools, json, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def parse_cfg(s):
    out = {}
    for part in filter(None, s.split(";")):
        shape, cfg = part.split("=")
        n, k = map(int, shape.split("x"))
        out[(n, k)] = tuple(int(v) for v in cfg.split(","))
    return out


def collect(m):
    from quf.runtime.kernels.decode_gemv import FastGroup, FastLinear
    by = {}
    for lay in m._fast_layers:
        if lay is None:
            continue
        for f in lay:
            for mem in (f.members if isinstance(f, FastGroup) else [f]):
                if isinstance(mem, FastLinear):
                    by.setdefault((mem.out_features, mem.in_features), []).append(mem)
    return by


def micro(a):
    import torch
    from quf.runtime.kernels.decode_gemv import gemv_jmajor, splitk_workspace
    from quf.runtime.model import QUFModel
    m = QUFModel.load(str(ROOT / a.quf), "cuda", backend="triton_int4")
    res = {}
    for (n, k), mods in collect(m).items():
        if a.only and f"{n}x{k}" not in a.only:
            continue
        x = torch.randn(k, device="cuda").half().reshape(1, k)
        out = torch.empty((1, n), dtype=torch.float16, device="cuda")
        graphs = {}
        cfgs = list(itertools.product(a.bn, a.ngb, a.warps, a.stages, a.split))
        default = (1, 32, 2, 2, 1) if k >= 4096 else (2, 16, 1, 2, 1)
        if default not in cfgs:
            cfgs.append(default)
        for cfg in cfgs:  # phase 1: compile and capture every candidate
            try:
                ws = splitk_workspace(n, cfg[0], cfg[4], "cuda") if cfg[4] > 1 else None
                for f in mods:
                    gemv_jmajor(x, f.layout, f.bias, out=out, config=cfg, workspace=ws)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for f in mods:
                        gemv_jmajor(x, f.layout, f.bias, out=out, config=cfg, workspace=ws)
                graphs[cfg] = g
            except Exception:  # noqa: BLE001
                pass
        t0 = time.perf_counter()  # sustained load so the clocks ramp before timing
        while time.perf_counter() - t0 < 1.0:
            for g in graphs.values():
                g.replay()
            torch.cuda.synchronize()
        times = {c: [] for c in graphs}
        for _ in range(a.rounds):  # phase 2: candidates interleaved, so a clock drift hits all of them
            for c, g in graphs.items():
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                times[c].append((time.perf_counter() - t) / 3 / len(mods) * 1e6)
        rows = sorted((statistics.median(v), c) for c, v in times.items())
        res[f"{n}x{k}"] = {"count": len(mods), "best": rows[:8], "default_us": next((r[0] for r in rows if r[1] == default), None)}
        print(f"{n}x{k} x{len(mods)}:", [(round(u, 2), c) for u, c in rows[:5]], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def graph(a):
    import torch
    import quf.runtime.kernels.decode_gemv as D
    import measure_decode_speed as M
    from quf.bench.suite import _tokenizer_from_header
    from quf.runtime.model import QUFModel
    m = QUFModel.load(str(ROOT / a.quf), "cuda", backend="triton_int4")
    M._suppress_eos(m)
    ids = _tokenizer_from_header(m.header).encode(M.PROMPT)
    by = collect(m)
    base = {id(mem): mem.config for mems in by.values() for mem in mems}
    sets = {name: parse_cfg(s) for name, s in (p.split(":", 1) for p in a.cfg.split("@"))}

    def apply(cfgmap):  # the graph is captured by each generate call, so it picks the config up
        for shape, mems in by.items():
            for mem in mems:
                mem.config = cfgmap.get(shape, base[id(mem)])
    st = {n: [] for n in sets}
    for r in range(2 + a.repeats):
        for n, cm in sets.items():
            apply(cm)
            tps, ms, used, cnt = M._timed_run(m, ids, None)
            if r >= 2:
                st[n].append(ms)
    for n in sets:
        med = statistics.median(st[n])
        print(json.dumps({"name": n, "steady_ms": round(med, 4), "steady_tok_s": round(1000 / med, 1),
                          "samples": [round(v, 3) for v in st[n]]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("micro", "graph"))
    ap.add_argument("--quf", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--bn", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ngb", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--stages", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--split", type=int, nargs="+", default=[1])
    a = ap.parse_args()
    micro(a) if a.mode == "micro" else graph(a)
