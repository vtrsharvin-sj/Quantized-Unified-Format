"""Budget-constrained mixed precision study (task 34.4 part B, Requirement 39).

Step 1 of the study: measure, for each decoder linear, how much perplexity improves when that
one module is int8 instead of int4 (all other modules stay at the size_b int4 GPTQ result),
divide by the extra bytes, and rank. Everything is done in memory on one GPTQ run, so a
module probe costs one scoring pass, not a conversion.

    python scripts/mixed_precision_study.py --model Qwen/Qwen2.5-0.5B --out docs/mixed_precision_0.5B.json

Ranking windows are wikitext-2 test windows 16..31 and C4 held-out windows 16..31 (llama
protocol), disjoint from the first 16 windows used by the quick screens.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def _bench():
    spec = importlib.util.spec_from_file_location("bench_accuracy", ROOT / "scripts" / "bench_accuracy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_accuracy"] = mod
    spec.loader.exec_module(mod)
    return mod


def delta_bytes(out_f: int, in_f: int, group_size: int = 32) -> int:
    """Extra bytes of int8 (per-channel scales) over int4 (g_idx, fp16 scales, packed zeros)."""
    groups = in_f // group_size
    int4 = out_f * in_f // 2 + out_f * groups * 2 + out_f * groups // 2 + in_f * 4
    int8 = out_f * in_f + out_f * 2
    return int8 - int4


class Scorer:
    def __init__(self, model, wins: list[list[int]], protocol: str):
        self.model, self.protocol = model, protocol
        self.ids = [torch.tensor([w], dtype=torch.long, device="cuda") for w in wins]
        self.wins = wins

    @torch.no_grad()
    def ppl(self) -> float:
        nll, cnt = 0.0, 0
        for ids in self.ids:
            n = ids.shape[1]
            first = n // 2
            logits = self.model(input_ids=ids, use_cache=False).logits[0, first : n - 1].float()
            lp = torch.log_softmax(logits, -1)
            tgt = ids[0, first + 1 : n]
            nll += float(-lp.gather(1, tgt[:, None]).sum())
            cnt += tgt.numel()
        return math.exp(nll / cnt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--plans", help="JSON {name: [glob, ...]} of int8 rules to score together instead of the per-module probe")
    ap.add_argument("--skip", type=int, default=16, help="skip the first N windows (used by the quick screens)")
    a = ap.parse_args()
    ba = _bench()
    from transformers import AutoModelForCausalLM
    import quf.convert.converter as conv
    from quf.bench.suite import load_wikitext
    from quf.convert.precision import PrecisionPlan
    from quf.quant.gptq import quantize_model_sequential
    from quf.quant.quantize import dequantize_int8, quantize_int8
    from quf.tokenization.calibration import build_calibration
    from quf.tokenization.wrapper import TokenizerWrapper

    cfg = conv.read_source_config(a.model)
    nl = cfg["num_hidden_layers"]
    info = conv.build_tokenizer_info(a.model, cfg)
    tok = TokenizerWrapper(None, hf_tokenizer=conv._hf_from_info(info, a.model))
    calib = build_calibration(conv.load_calibration_texts("wikitext-2-raw-v1"), tok, 128, 1024, 0)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).eval()
    fp16_w = {n: m.weight.detach().clone() for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and ".layers." in n}
    plan = PrecisionPlan.from_options(presets=["size_b"], num_layers=nl)  # int4 body, last layer int8
    t = time.perf_counter()
    quantize_model_sequential(model, calib, device="cuda", group_size=32, act_order=True, clip_search=True,
                              precision=plan.precision, int8_out={})
    print(f"GPTQ done in {time.perf_counter() - t:.0f}s", flush=True)
    model.to("cuda")
    mods = dict(model.named_modules())
    base_w = {n: mods[n].weight.detach().clone() for n in fp16_w}  # int4 (and last-layer int8) dequantized

    wiki = ba.window_ids(tok.encode(load_wikitext()), 0)[a.skip : a.skip + a.windows]
    c4 = ba.window_ids(tok.encode(ba.c4_heldout_text()), 0)[a.skip : a.skip + a.windows]
    sw, sc = Scorer(model, wiki, "llama"), Scorer(model, c4, "llama")
    base = (sw.ppl(), sc.ppl())
    full = {}
    print(f"base (size_b int4 body, {len(wiki)}+{len(c4)} windows): wiki {base[0]:.4f} c4 {base[1]:.4f}", flush=True)

    @torch.no_grad()
    def set_int8(names):
        for n in names:
            q, s = quantize_int8(fp16_w[n].to("cuda"))
            mods[n].weight.copy_(dequantize_int8(q, s).to("cuda", mods[n].weight.dtype))

    @torch.no_grad()
    def reset():
        for n, w in base_w.items():
            mods[n].weight.copy_(w)

    rows = []
    last = nl - 1
    if a.plans:
        import fnmatch
        plans = json.loads(Path(a.plans).read_text())
        res = {}
        for pname, globs in plans.items():
            names = [n for n in fp16_w if any(fnmatch.fnmatchcase(n, g.replace("{last}", str(last))) for g in globs)]
            names = [n for n in names if f".layers.{last}." not in n]
            set_int8(names)
            w, c = sw.ppl(), sc.ppl()
            reset()
            extra = sum(delta_bytes(*fp16_w[n].shape) for n in names)
            res[pname] = {"globs": globs, "n_modules": len(names), "extra_mb": extra / 1e6, "wiki": w, "c4": c,
                          "wiki_vs_base_pct": 100 * (w / base[0] - 1), "c4_vs_base_pct": 100 * (c / base[1] - 1)}
            print(pname, json.dumps(res[pname]), flush=True)
        Path(a.out).write_text(json.dumps({"model": a.model, "base_wiki": base[0], "base_c4": base[1], "plans": res}, indent=1))
        print("done", flush=True)
        return 0
    for n in fp16_w:
        if f".layers.{last}." in n:
            continue  # already int8 in size_b
        set_int8([n])
        w, c = sw.ppl(), sc.ppl()
        reset()
        W = fp16_w[n]
        db = delta_bytes(W.shape[0], W.shape[1])
        # gain is the mean relative perplexity improvement over the two domains
        gain = 0.5 * ((1 - w / base[0]) + (1 - c / base[1]))
        layer = int(n.split(".")[2])
        rows.append({"name": n, "layer": layer, "kind": n.rsplit(".", 1)[-1], "wiki": w, "c4": c,
                     "gain": gain, "delta_bytes": db, "gain_per_mb": gain / (db / 1e6)})
        print(f"{n:48s} gain {100 * gain:+.4f}%  +{db / 1e6:.3f} MB  {100 * gain / (db / 1e6):+.4f}%/MB", flush=True)
    rows.sort(key=lambda r: -r["gain_per_mb"])
    Path(a.out).write_text(json.dumps({"model": a.model, "base_wiki": base[0], "base_c4": base[1],
                                       "windows": f"{a.skip}..{a.skip + a.windows - 1}", "rows": rows}, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
