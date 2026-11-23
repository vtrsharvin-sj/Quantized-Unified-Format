"""Task 33.2 part A: fused residual-add + RMSNorm and the on-device greedy loop (GPU + Triton).

**Validates: Requirements 37.4, 37.6, 37.7**
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quf.runtime.kernels.triton_kernels import triton_available

pytestmark = [pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "eval_cache" / "lm" / "sb05.quf", ROOT / "eval_cache" / "lm" / "sb15.quf"]
PROMPTS = ["The quick brown fox", "Once upon a time", "def fibonacci(n):", "The capital of France is"]

needs_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available()[0], reason="needs CUDA and Triton"
)


def _torch_ref(x, res, w, eps):
    s = x + res if res is not None else x
    sf = s.float()
    y = (sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return w * y, s


@needs_triton
@pytest.mark.parametrize("n", [128, 896, 1536, 1000])
@pytest.mark.parametrize("with_res", [True, False])
def test_add_rmsnorm_matches_torch(n, with_res):
    from quf.runtime.kernels.fused import add_rmsnorm

    g = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn(1, 1, n, device="cuda", dtype=torch.float16, generator=g)
    res = torch.randn(1, 1, n, device="cuda", dtype=torch.float16, generator=g) if with_res else None
    w = torch.rand(n, device="cuda", dtype=torch.float16, generator=g) + 0.5
    normed, new_res = add_rmsnorm(x, res, w, 1e-6)
    ref_n, ref_r = _torch_ref(x, res, w, 1e-6)
    assert (normed.float() - ref_n.float()).abs().max().item() < 1e-3
    assert (new_res.float() - ref_r.float()).abs().max().item() < 1e-3


@needs_triton
def test_add_rmsnorm_is_graph_capturable():
    from quf.runtime.kernels.fused import add_rmsnorm

    x = torch.randn(1, 1, 896, device="cuda", dtype=torch.float16)
    res = torch.randn_like(x)
    w = torch.ones(896, device="cuda", dtype=torch.float16)
    add_rmsnorm(x, res, w, 1e-6)  # compile outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        normed, new_res = add_rmsnorm(x, res, w, 1e-6)
    x.copy_(torch.randn_like(x))
    g.replay()
    ref_n, ref_r = _torch_ref(x, res, w, 1e-6)
    assert (normed.float() - ref_n.float()).abs().max().item() < 1e-3
    assert (new_res.float() - ref_r.float()).abs().max().item() < 1e-3


def _gen(model, tok, prompt, n, **kw):
    from quf.runtime.generate import generate

    return generate(model, tok, tok.encode(prompt), max_new_tokens=n, temperature=0.0, **kw)


@needs_triton
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_greedy_tokens_equal_030_path_and_eos_trims(path):
    if not path.exists():
        pytest.skip(f"{path} not built")
    from quf.bench.suite import _tokenizer_from_header
    from quf.runtime.model import QUFModel

    model = QUFModel.load(path, "cuda", backend="triton_int4")
    assert model.fused_decode
    tok = _tokenizer_from_header(model.header)
    eos = model.header.tokenizer.get("eos_token_id")
    model.header.tokenizer["eos_token_id"] = None  # run the full length first
    new = {p: _gen(model, tok, p, 40) for p in PROMPTS}
    streamed = []
    assert _gen(model, tok, PROMPTS[0], 40, stream=streamed.append) == new[PROMPTS[0]]
    assert "".join(streamed) == tok.decode(new[PROMPTS[0]]).rstrip("\ufffd")
    # EOS in the middle of a chunk: output ends at the first occurrence, nothing after it.
    ref = new[PROMPTS[1]]
    stop = ref[11]
    model.header.tokenizer["eos_token_id"] = stop
    cut = ref.index(stop) + 1
    assert _gen(model, tok, PROMPTS[1], 40) == ref[:cut]
    model.header.tokenizer["eos_token_id"] = eos

    model.fused_decode = False  # the 0.3.0 path
    model.header.tokenizer["eos_token_id"] = None
    for p in PROMPTS:
        assert _gen(model, tok, p, 40) == new[p], p
    del model
    torch.cuda.empty_cache()
