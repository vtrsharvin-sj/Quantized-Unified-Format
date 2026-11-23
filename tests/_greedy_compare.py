"""Fused fast path against the unfused path on a real file, greedy tokens and logits.

    python tests/_greedy_compare.py eval_cache/lm040/q05.quf [more.quf ...]

Loads the file twice (``QUF_NO_FUSED=1`` for the reference, default for the fused path), runs 4
prompts x 64 greedy tokens through the CUDA-graph generate loop on both, then replays the
reference tokens through ``decode_token`` on both models and compares the logits. Real files carry
fp16 noise of order 0.1 on logits of size 10 to 20 (the all-int4 0.3.0 file sb05 shows 0.078), so
the bound is LOGIT_BOUND, not the 1e-2 + 2e-3 of the tiny-model tests. A token difference passes only where the top two reference logits differ by less than
NEAR_TIE_GAP. Exit code 0 means all files pass.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = ["The quick brown fox", "Explain what a hash table is.", "def fibonacci(n):", "In 1969, the first"]
NEW = 64
LOGIT_BOUND = 0.25
NEAR_TIE_GAP = 5e-3


def _load(path: str, no_fused: bool):
    from quf.runtime.model import QUFModel

    os.environ["QUF_NO_FUSED"] = "1" if no_fused else ""
    try:
        m = QUFModel.load(path, "cuda", backend="triton_int4")
    finally:
        os.environ.pop("QUF_NO_FUSED", None)
    m.header.tokenizer["eos_token_id"] = None  # never stop early
    return m


def _teacher_forced_logits(model, ids, tokens):
    cache = model.new_static_cache(len(ids) + len(tokens) + 1)
    model.forward(torch.tensor([ids], device=model.device), cache)
    out = []
    for t in tokens:
        out.append(model.decode_token(t, cache)[0, -1].float().clone())
    return torch.stack(out)


def compare(path: str) -> bool:
    from quf.bench.suite import _tokenizer_from_header
    from quf.runtime.generate import generate

    ref, fast = _load(path, True), _load(path, False)
    assert fast.fused_decode and not ref.fused_decode, (fast.fused_decode, ref.fused_decode)
    n_fast = sum(f is not None for f in fast._fast_layers)
    print(f"{path}: fast layers {n_fast}/{len(fast._fast_layers)}")
    tok = _tokenizer_from_header(ref.header)
    ok = True
    flips = 0
    worst = 0.0
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        a = generate(ref, tok, ids, max_new_tokens=NEW, temperature=0.0, cuda_graph=True)
        b = generate(fast, tok, ids, max_new_tokens=NEW, temperature=0.0, cuda_graph=True)
        la = _teacher_forced_logits(ref, ids, a)
        lb = _teacher_forced_logits(fast, ids, a)
        worst = max(worst, float((la - lb).abs().max()))
        if float((la - lb).abs().max()) > LOGIT_BOUND:
            print(f"  logits differ by more than {LOGIT_BOUND} for {prompt!r}")
            ok = False
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None:
            print(f"  {prompt!r}: {len(a)} tokens equal")
            continue
        flips += 1
        top2 = torch.topk(la[first], 2).values
        gap = float(top2[0] - top2[1])
        print(f"  {prompt!r}: first difference at step {first}, reference top-2 gap {gap:.5f}")
        ok &= gap < NEAR_TIE_GAP
    print(f"  max |logit diff| {worst:.4g}, prompts with a token difference {flips}, {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [compare(p) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
