"""Property 43: fused fast path equals the unfused path on mixed-precision modules (GPU + Triton).

Whole-model part: tiny models with random per-module int4/int8/fp16 mixes (plus the size_f layout:
k/v int8 and an all-int8 last layer), group sizes 32/64 and act-order. The fused model (default)
and the unfused model (``QUF_NO_FUSED=1``) are loaded from the same file and compared with the
near-tie helper ``tests/_greedy_compare.py``. Kernel part: ``gemv_int8`` against the fp32 reference.
"""

from __future__ import annotations

import gc
import zlib

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import _greedy_compare as gcmp
import test_int4_greedy as tig
from quf.quant.quantize import dequantize_int8, quantize_int8
from quf.runtime.generate import generate
from quf.runtime.kernels.decode_gemv import gemv_int8

pytestmark = [pytest.mark.gpu, pytest.mark.triton]

ATOL, RTOL = 1e-2, 2e-3  # Property 43
MAX_NEAR_TIE_FRACTION = 0.05
_CACHE: dict[tuple, tuple] = {}
_STATS = {"examples": 0, "near_ties": 0}


def _pair(root, key):
    if key not in _CACHE:
        while len(_CACHE) >= 4:
            _CACHE.pop(next(iter(_CACHE)))
            gc.collect()
            torch.cuda.empty_cache()
        shape, gs, act_order, mix = key
        path = root / f"{shape}_{gs}_{int(act_order)}_{mix}.quf"
        tig._build(path, shape, gs, act_order, mix, seed=zlib.crc32(repr(key).encode()) % 1000)
        ref, fast = gcmp._load(str(path), True), gcmp._load(str(path), False)
        assert fast.fused_decode and not ref.fused_decode
        _CACHE[key] = (ref, fast)
    return _CACHE[key]


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    _STATS.update(examples=0, near_ties=0)
    yield tmp_path_factory.mktemp("int8_fast_props")
    _CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


@settings(max_examples=100, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    shape=st.sampled_from(["tiny", "mha", "qwen0.5b"]),
    gs=st.sampled_from([32, 64]),
    act_order=st.booleans(),
    mix=st.sampled_from(["mixed", "size_f"]),
    prompt=st.lists(st.integers(1, tig.VOCAB - 1), min_size=1, max_size=16),
    new=st.integers(2, 16),
)
def test_fused_equals_unfused_on_mixed_modules(model_root, shape, gs, act_order, mix, prompt, new):
    """Feature: quf-quantized-format, Property 43: Fused fast path equals the unfused path on mixed-precision modules

    **Validates: Requirements 42.3, 42.4, 42.5**
    """
    ref, fast = _pair(model_root, (shape, gs, act_order, mix))
    _STATS["examples"] += 1
    near_tie = False
    for graph in (True, False):
        a = generate(ref, None, prompt, max_new_tokens=new, temperature=0.0, cuda_graph=graph)
        b = generate(fast, None, prompt, max_new_tokens=new, temperature=0.0, cuda_graph=graph)
        assert len(a) == len(b) == new
        # teacher-forced logits on the reference tokens, same prompt, both models
        la = gcmp._teacher_forced_logits(ref, prompt, a).cpu()
        lb = gcmp._teacher_forced_logits(fast, prompt, a).cpu()
        torch.testing.assert_close(lb, la, atol=ATOL, rtol=RTOL)
        if a != b:
            i = next(j for j in range(new) if a[j] != b[j])
            top2 = torch.topk(la[i], 2).values
            gap = float(top2[0] - top2[1])
            # A flip needs the top-2 gap to be closed by logit noise on both logits, each within the Property 43`n            # tolerance, so gaps up to twice that tolerance (and never below the helper gap) are near-ties.`n            limit = max(gcmp.NEAR_TIE_GAP, 2 * (ATOL + RTOL * float(la[i].abs().max())))`n            assert gap < limit, f"step {i}: unfused {a[i]} != fused {b[i]}, top-2 gap {gap} (graph={graph})"
            near_tie = True
    _STATS["near_ties"] += int(near_tie)


def test_mixed_near_ties_are_rare(model_root):
    """Runs after the property test (definition order); the cap is a rate over 100 examples."""
    n = _STATS["examples"]
    if n == 0:
        pytest.skip("property test did not run")
    print(f"near-ties: {_STATS['near_ties']} of {n}")
    assert _STATS["near_ties"] <= MAX_NEAR_TIE_FRACTION * n


@settings(max_examples=100, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    n=st.integers(1, 1200),
    k=st.integers(1, 1500),
    bias=st.booleans(),
    scatter_in=st.booleans(),
    scatter_out=st.booleans(),
    seed=st.integers(0, 10**6),
)
def test_gemv_int8_equals_fp32_reference(n, k, bias, scatter_in, scatter_out, seed):
    """Feature: quf-quantized-format, Property 43: Fused fast path equals the unfused path on mixed-precision modules

    **Validates: Requirements 42.1, 42.2**
    """
    cpu = torch.Generator().manual_seed(seed)
    q, s = quantize_int8(torch.randn(n, k, generator=cpu) * 0.05)
    q, s = q.cuda(), s.cuda()
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(1, k, device="cuda", generator=g).to(torch.float16)
    b = torch.randn(n, device="cuda", generator=g).to(torch.float16) if bias else None
    ref = x.float() @ dequantize_int8(q, s).float().T
    if bias:
        ref = ref + b.float()
    x_index = out_index = out = None
    if scatter_in:  # weight column c reads x[x_index[c]]
        x_index = torch.randperm(k, device="cuda", generator=g).to(torch.int32)
        xs = torch.empty_like(x)
        xs[0, x_index.long()] = x[0]
        x = xs
    if scatter_out:
        out_index = (torch.randperm(n, device="cuda", generator=g) + 3).to(torch.int32)
        out = torch.zeros(1, n + 8, device="cuda", dtype=torch.float16)
    res = gemv_int8(x, q, s, bias=b, out=out, x_index=x_index, out_index=out_index)
    got = res[:, out_index.long()] if scatter_out else res
    assert got.dtype == torch.float16 and got.shape == (1, n)
    torch.testing.assert_close(got.float(), ref, atol=ATOL, rtol=RTOL)
