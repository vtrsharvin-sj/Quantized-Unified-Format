"""Accuracy diagnosis (task 15.4, Requirements 18.1-18.4).

Prints a ranked table of per-module relative output error and the WikiText-2
perplexity of each ablation subset (only that subset quantized to 4 bits, the
rest fp16), for RTN and for GPTQ, and reports whether the tied embedding is fp16
in a converted file.

    python scripts/diagnose_accuracy.py --model Qwen/Qwen2.5-0.5B --out docs/accuracy_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from quf.analysis.sensitivity import (  # noqa: E402
    Scheme,
    ablate,
    decoder_linear_names,
    module_errors,
    standard_subsets,
)
from quf.bench.suite import load_wikitext  # noqa: E402
from quf.convert.converter import (  # noqa: E402
    _coerce_other,
    _hf_from_info,
    build_tokenizer_info,
    load_calibration_texts,
    read_source_config,
)
from quf.tokenization.calibration import build_calibration  # noqa: E402
from quf.tokenization.wrapper import TokenizerWrapper  # noqa: E402


def tied_embedding_report(model) -> dict:
    """Whether the tied embedding is stored fp16 (and lm_head not duplicated) by the converter."""
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    name = "model.embed_tokens.weight"
    quantized_names = set(decoder_linear_names(model))
    emb_is_decoder_linear = name.rsplit(".", 1)[0] in quantized_names
    emb = model.state_dict()[name].detach().cpu().float().numpy()
    stored = _coerce_other(name, emb)  # same coercion the converter applies to non-quantized tensors
    return {
        "tie_word_embeddings": tied,
        "embedding_quantized": emb_is_decoder_linear,
        "embedding_stored_dtype": str(stored.dtype),
        "embedding_shape": list(stored.shape),
        "lm_head_stored_separately": not tied,
        "embedding_fp16_in_converted_file": (not emb_is_decoder_linear) and stored.dtype == np.float16,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--schemes", default="rtn,gptq")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()
    cfg = read_source_config(a.model)
    info = build_tokenizer_info(a.model, cfg)
    tok = TokenizerWrapper(None, hf_tokenizer=_hf_from_info(info, a.model))
    calib = build_calibration(load_calibration_texts("wikitext-2-raw-v1"), tok, a.samples, a.seq_len, 0)
    calib = torch.as_tensor(calib, dtype=torch.long)
    text = load_wikitext()
    eval_cfg = {"text": text}
    subsets = standard_subsets(model)
    result: dict = {"model": a.model, "samples": a.samples, "seq_len": a.seq_len}
    result["tied_embedding"] = tied_embedding_report(model)
    print("tied embedding:", json.dumps(result["tied_embedding"]), flush=True)

    t = time.perf_counter()
    result["fp16_perplexity"] = ablate(model, tok, [], eval_cfg=eval_cfg)
    print(f"fp16 perplexity {result['fp16_perplexity']:.4f} ({time.perf_counter()-t:.0f}s)", flush=True)
    result["subset_sizes"] = {k: len(v) for k, v in subsets.items()}

    for sname in a.schemes.split(","):
        scheme = Scheme(method=sname, calib=calib)
        res: dict = {}
        t = time.perf_counter()
        errs = module_errors(model, calib, scheme)
        res["module_errors"] = [e.__dict__ for e in errs]
        print(f"\n[{sname}] module_errors done ({time.perf_counter()-t:.0f}s); top {a.top} by relative output error", flush=True)
        for e in sorted(errs, key=lambda e: -e.rel_output_error)[: a.top]:
            print(f"  {e.name:48s} {e.rel_output_error:.4f}", flush=True)
        by_kind: dict[str, list[float]] = {}
        for e in errs:
            by_kind.setdefault(e.kind, []).append(e.rel_output_error)
        res["mean_error_by_kind"] = {k: float(np.mean(v)) for k, v in by_kind.items()}
        by_layer: dict[int, list[float]] = {}
        for e in errs:
            by_layer.setdefault(e.layer, []).append(e.rel_output_error)
        res["mean_error_by_layer"] = {str(k): float(np.mean(v)) for k, v in sorted(by_layer.items())}
        print("  mean error by kind:", json.dumps(res["mean_error_by_kind"]), flush=True)

        res["ablation_perplexity"] = {}
        everything = set().union(*subsets.values())
        for key, names in list(subsets.items()) + [("all", everything)]:
            t = time.perf_counter()
            ppl = ablate(model, tok, names, scheme, eval_cfg)
            res["ablation_perplexity"][key] = ppl
            print(f"[{sname}] ablation {key:9s} n={len(names):3d} ppl={ppl:.4f} "
                  f"(+{100*(ppl/result['fp16_perplexity']-1):.2f}%) ({time.perf_counter()-t:.0f}s)", flush=True)
            torch.cuda.empty_cache()
        result[sname] = res
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    print("\nranking by perplexity impact (increase over fp16):")
    for sname in a.schemes.split(","):
        ab = result[sname]["ablation_perplexity"]
        rank = sorted(((k, v - result["fp16_perplexity"]) for k, v in ab.items() if k != "all"), key=lambda kv: -kv[1])
        print(f"  {sname}: " + ", ".join(f"{k} +{d:.4f}" for k, d in rank))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
