"""Whole-model check: triton_int4 vs torch_gpu logits on the real 0.5B size_b file.

Skips cleanly when the file is missing, so CI and other machines are unaffected.
Requirements 35.2
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

from quf.runtime.model import QUFModel

pytestmark = [pytest.mark.gpu, pytest.mark.triton, pytest.mark.realmodel]

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [
    Path(os.environ.get("TEMP", "")) / "quf_user" / "out.quf",
    ROOT / "eval_cache" / "lm" / "sb05.quf",
]
ATOL = 5e-2
DECODE_STEPS = 16
PROMPTS = [
    [9707, 11, 847, 829, 374, 264, 1273],
    [785, 6722, 315, 9625, 374],
    [3838, 374, 279, 1850, 1616, 311, 3960, 311, 2038, 30],
    [641, 374, 264, 2613, 315, 1995, 429, 5707, 311, 8830, 279, 12720, 315, 1251, 13],
]


def _find() -> Path | None:
    for p in CANDIDATES:
        if str(p) and p.is_file():
            return p
    return None


@pytest.fixture(scope="module")
def models():
    path = _find()
    if path is None:
        pytest.skip("real 0.5B .quf file not found")
    ref = QUFModel.load(path, "cuda", backend="torch_gpu")
    fast = QUFModel.load(path, "cuda", backend="triton_int4")
    yield ref, fast
    del ref, fast
    gc.collect()
    torch.cuda.empty_cache()


def _run(model: QUFModel, prompt: list[int]):
    cache = model.new_static_cache(len(prompt) + DECODE_STEPS)
    logits = model.forward(torch.tensor([prompt]), cache)[0, -1].float().cpu()
    prefill = logits
    tokens, steps = [int(logits.argmax())], []
    for _ in range(DECODE_STEPS):
        logits = model.decode_token(tokens[-1], cache)[0, -1].float().cpu().clone()
        steps.append(logits)
        tokens.append(int(logits.argmax()))
    return prefill, steps, tokens


@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: f"len{len(p)}")
def test_triton_int4_matches_torch_gpu_on_real_model(models, prompt):
    ref, fast = models
    r_pre, r_steps, r_tok = _run(ref, prompt)
    f_pre, f_steps, f_tok = _run(fast, prompt)
    assert (r_pre - f_pre).abs().max().item() <= ATOL, "prefill logits differ"
    assert r_tok == f_tok, f"greedy tokens differ: {r_tok} vs {f_tok}"
    for i, (a, b) in enumerate(zip(r_steps, f_steps)):
        err = (a - b).abs().max().item()
        assert err <= ATOL, f"decode step {i} logits differ by {err}"
