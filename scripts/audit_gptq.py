"""Audit helper for task 15.3: WikiText-2 perplexity of in-memory GPTQ/RTN variants.

Quantizes the decoder linears of an HF model with ``quf.quant.gptq`` (or RTN),
swaps the dequantized weights in, and measures perplexity with the same
windowing as ``quf.bench.suite.perplexity`` (2048-token windows, test split).

    python scripts/audit_gptq.py --mode gptq --samples 128 --seq-len 1024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from quf.bench.suite import load_wikitext, perplexity  # noqa: E402
from quf.convert.converter import build_tokenizer_info, load_calibration_texts, read_source_config, _hf_from_info  # noqa: E402
from quf.quant.dequant import reference_dequantize  # noqa: E402
from quf.quant.gptq import quantize_model_sequential  # noqa: E402
from quf.quant.quantize import quantize_groupwise  # noqa: E402
from quf.tokenization.calibration import build_calibration  # noqa: E402
from quf.tokenization.wrapper import TokenizerWrapper  # noqa: E402


class _M:
    def __init__(self, m, dev):
        self.m, self.dev = m, dev

    def forward(self, ids):
        with torch.no_grad():
            return self.m(ids.to(self.dev)).logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--mode", choices=["fp16", "rtn", "gptq"], default="gptq")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    dev = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32 if a.fp32 else torch.float16)
    cfg = read_source_config(a.model)
    info = build_tokenizer_info(a.model, cfg)
    tok = TokenizerWrapper(None, hf_tokenizer=_hf_from_info(info, a.model))

    t0 = time.perf_counter()
    if a.mode == "gptq":
        calib = build_calibration(load_calibration_texts("wikitext-2-raw-v1"), tok, a.samples, a.seq_len, 0)
        quantize_model_sequential(model, calib, device=dev, symmetric=a.symmetric, damp=a.damp)
    elif a.mode == "rtn":
        for layer in model.model.layers:
            for m in layer.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.data.copy_(reference_dequantize(quantize_groupwise(m.weight.data, symmetric=a.symmetric)))
    qsecs = time.perf_counter() - t0
    model.to(torch.float16).to(dev).eval()
    ppl = perplexity(_M(model, dev), tok, text=load_wikitext())
    res = {"mode": a.mode, "samples": a.samples, "seq_len": a.seq_len, "damp": a.damp,
           "symmetric": a.symmetric, "fp32_compute": a.fp32, "perplexity": ppl, "quantize_seconds": qsecs}
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

