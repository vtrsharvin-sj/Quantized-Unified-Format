"""Task 33.4: property tests of the Fused_Decode_Kernels (GPU + Triton).

Property 38: each fused kernel equals its unfused torch reference (kernel level, 100 examples each).
Property 40: greedy tokens of the fused decode path equal the 0.3.0 path (model level, 25 examples,
             near-tie rule and near-tie cap of test_int4_greedy.py).
Property 42: seeded sampling is deterministic across two runs, graph on and off (model level,
             20 examples; sampler level, 100 examples).
Property 39 (quantized lm_head GEMV) lives in test_embed_runtime.py.

**Validates: Requirements 37.2, 37.5, 37.6, 37.7**
"""
from __future__ import annotations

import gc
import math
import zlib

import pytest
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quf.runtime.kernels.triton_kernels import Int4Layout, dequantize_layout, triton_available

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.triton,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not triton_available()[0], reason="needs CUDA and Triton"
    ),
]

ATOL, RTOL = 1e-2, 2e-3  # Property 38 tolerance
KERNEL = settings(max_examples=100, deadline=None, suppress_health_check=list(HealthCheck))
MODEL = settings(max_examples=25, deadline=None, suppress_health_check=list(HealthCheck))


def _close(got, ref):
    err = (got.float() - ref.float()).abs() - (ATOL + RTOL * ref.float().abs())
    assert err.max().item() <= 0, f"max excess {err.max().item()}"


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


# -- Property 38: fused add + RMSNorm ---------------------------------------------------------

@KERNEL
@given(n=st.integers(16, 2048), with_res=st.booleans(), scatter=st.booleans(), seed=st.integers(0, 10**6))
def test_add_rmsnorm_equals_reference(n, with_res, scatter, seed):
    """Feature: quf-quantized-format, Property 38: Fused kernel equals unfused reference

    **Validates: Requirements 37.2, 37.6**
    """
    from quf.runtime.kernels.fused import add_rmsnorm

    g = _gen(seed)
    x = torch.randn(1, 1, n, device="cuda", dtype=torch.float16, generator=g)
    res = torch.randn(1, 1, n, device="cuda", dtype=torch.float16, generator=g) if with_res else None
    w = torch.rand(n, device="cuda", dtype=torch.float16, generator=g) + 0.5
    s = x + res if with_res else x
    sf = s.float()
    ref = w.float() * (sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + 1e-6))
    if scatter:
        idx = torch.randperm(n, device="cuda", generator=g).to(torch.int32)
        normed, new_res = add_rmsnorm(x, res, w, 1e-6, out_index=idx)
        exp = torch.empty_like(ref.view(1, n))
        exp[0, idx.long()] = ref.view(n)
        _close(normed.view(1, n), exp)
    else:
        normed, new_res = add_rmsnorm(x, res, w, 1e-6)
        _close(normed, ref)
    _close(new_res, s)


# -- Property 38: fused RoPE + KV write ---------------------------------------------------------

def _rope_tables(t, d):
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, dtype=torch.float32, device="cuda") / d))
    freqs = torch.arange(t, device="cuda", dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().half(), emb.sin().half()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


gqa = st.tuples(st.integers(1, 4), st.integers(1, 8)).map(lambda rk: (rk[0] * rk[1], rk[1]))  # (heads, kv)
head_dim = st.sampled_from([32, 64, 128])


@KERNEL
@given(hk=gqa, d=head_dim, t=st.integers(1, 300), data=st.data(), seed=st.integers(0, 10**6))
def test_rope_kv_write_equals_reference(hk, d, t, data, seed):
    """Feature: quf-quantized-format, Property 38: Fused kernel equals unfused reference

    **Validates: Requirements 37.2, 37.6**
    """
    from quf.runtime.kernels.attention import rope_kv_write

    h, kv = hk
    p = data.draw(st.integers(0, t - 1))
    g = _gen(seed)
    cos, sin = _rope_tables(t, d)
    qkv = torch.randn(1, (h + 2 * kv) * d, device="cuda", dtype=torch.float16, generator=g)
    kc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16, generator=g)
    vc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16, generator=g)
    kc0, vc0 = kc.clone(), vc.clone()
    q = rope_kv_write(qkv, cos, sin, torch.tensor([p], device="cuda"), kc, vc, h, kv)

    qq = qkv[:, : h * d].view(1, 1, h, d).transpose(1, 2).float()
    kk = qkv[:, h * d : (h + kv) * d].view(1, 1, kv, d).transpose(1, 2).float()
    vv = qkv[:, (h + kv) * d :].view(1, 1, kv, d).transpose(1, 2)
    c, s = cos[p].float(), sin[p].float()
    _close(q.view(1, h, 1, d), qq * c + _rotate_half(qq) * s)
    kc0[:, :, p] = (kk * c + _rotate_half(kk) * s)[:, :, 0].half()
    vc0[:, :, p] = vv[:, :, 0]
    _close(kc, kc0)
    assert torch.equal(vc, vc0)  # v is a plain copy and every other slot is untouched


# -- Property 38: flash decode ------------------------------------------------------------------

def _attn_ref(q, kc, vc, pos, h, kv, d):
    rep = h // kv
    k = kc[:, :, : pos + 1].float().repeat_interleave(rep, dim=1)
    v = vc[:, :, : pos + 1].float().repeat_interleave(rep, dim=1)
    sc = (q.view(1, h, 1, d).float() @ k.transpose(-1, -2)) / math.sqrt(d)
    return (torch.softmax(sc, dim=-1) @ v).reshape(1, h * d)


@KERNEL
@given(
    hk=gqa, d=head_dim, t=st.integers(1, 1300), data=st.data(),
    chunk=st.sampled_from([64, 128, 256, 512]), seed=st.integers(0, 10**6),
)
def test_flash_decode_equals_reference_and_ignores_chunking(hk, d, t, data, chunk, seed):
    """Feature: quf-quantized-format, Property 38: Fused kernel equals unfused reference

    **Validates: Requirements 37.2, 37.6**
    """
    from quf.runtime.kernels.attention import flash_decode

    h, kv = hk
    p = data.draw(st.integers(0, t - 1))
    g = _gen(seed)
    q = torch.randn(h, d, device="cuda", dtype=torch.float16, generator=g)
    kc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16, generator=g)
    vc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16, generator=g)
    pos = torch.tensor([p], device="cuda")
    out = flash_decode(q, kc, vc, pos, h, kv, chunk=chunk)
    _close(out, _attn_ref(q, kc, vc, p, h, kv, d))
    _close(flash_decode(q, kc, vc, pos, h, kv, chunk=512), out)  # split-K count does not matter


# -- Property 38: j-major int4 GEMV (bias, scatter epilogues) and SiLU * mul -------------------

@KERNEL
@given(
    n=st.integers(1, 700), gs=st.sampled_from([32, 64, 128]), groups=st.integers(1, 12),
    perm=st.booleans(), zeros=st.booleans(), bias=st.booleans(), seed=st.integers(0, 10**6),
    config=st.one_of(st.none(), st.tuples(st.sampled_from([1, 2, 4, 8, 16]), st.sampled_from([4, 8, 16, 32]),
                                          st.sampled_from([1, 2]), st.sampled_from([2, 3, 4]), st.integers(1, 5))),
)
def test_decode_gemv_equals_dequantized_reference(n, gs, groups, perm, zeros, bias, seed, config):
    """Feature: quf-quantized-format, Property 38: Fused kernel equals unfused reference

    **Validates: Requirements 37.2, 37.6**
    """
    from quf.runtime.kernels.decode_gemv import FastLinear, activation_index, to_jmajor

    k = gs * groups
    g = _gen(seed)
    packed = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda", generator=g)
    sc = (torch.rand(n, k // gs, device="cuda", generator=g) * 0.02 + 0.001).half()
    zr = torch.randint(0, 16, (n, k // gs), device="cuda", generator=g).half() if zeros else None
    pm = torch.randperm(k, device="cuda", generator=g) if perm else None
    lay = Int4Layout(packed, sc, zr, pm, gs, k, n)
    x = torch.randn(1, k, device="cuda", dtype=torch.float16, generator=g)
    b = torch.randn(n, device="cuda", dtype=torch.float16, generator=g) if bias else None
    fl = FastLinear(lay, b)
    if config is not None:  # any tuned shape config, including split-K (task 39.5)
        fl.config = config
    y = fl(to_jmajor(x, lay))
    ref = x.float() @ dequantize_layout(lay).float().t()
    if bias:
        ref = ref + b.float()
    assert y.shape == (1, n)
    _close(y, ref)
    # the scatter epilogue: activation_index sends element i to the j-major slot of the next GEMV
    idx = activation_index(lay)
    assert idx.numel() == k
    assert torch.equal(torch.sort(idx.long()).values, torch.arange(k, device="cuda"))


@KERNEL
@given(half=st.integers(1, 3000), perm=st.booleans(), seed=st.integers(0, 10**6))
def test_silu_mul_scatter_equals_reference(half, perm, seed):
    """Feature: quf-quantized-format, Property 38: Fused kernel equals unfused reference

    **Validates: Requirements 37.2, 37.6**
    """
    from quf.runtime.kernels.decode_gemv import silu_mul_scatter

    g = _gen(seed)
    gu = torch.randn(1, 2 * half, device="cuda", dtype=torch.float16, generator=g)
    idx = torch.randperm(half, device="cuda", generator=g) if perm else torch.arange(half, device="cuda")
    out = silu_mul_scatter(gu, idx.to(torch.int32))
    act = F.silu(gu[:, :half].float()) * gu[:, half:].float()
    ref = torch.empty_like(act)
    ref[0, idx] = act[0]
    _close(out, ref)


# -- Property 40 / 42: tiny models --------------------------------------------------------------

NEAR_TIE_GAP = 1e-3
MAX_NEAR_TIE_FRACTION = 0.05
_CACHE: dict[tuple, tuple] = {}
_STATS = {"examples": 0, "near_ties": 0}
MODEL_SHAPES = ["tiny", "mha", "qwen0.5b"]


def _pair(root, key):
    """(fused model, 0.3.0 path model) for one tiny random quantized model."""
    if key not in _CACHE:
        while len(_CACHE) >= 4:
            _CACHE.pop(next(iter(_CACHE)))
            gc.collect()
            torch.cuda.empty_cache()
        import test_int4_greedy as tig
        from quf.runtime.model import QUFModel

        shape, gs, act_order, mix = key
        path = root / f"{shape}_{gs}_{int(act_order)}_{mix}.quf"
        tig._build(path, shape, gs, act_order, mix, seed=zlib.crc32(repr(key).encode()) % 1000)
        fused = QUFModel.load(path, "cuda", backend="triton_int4")
        old = QUFModel.load(path, "cuda", backend="triton_int4")
        for m in (fused, old):
            m.header.tokenizer["eos_token_id"] = None
        assert fused.fused_decode, "the Fused_Decode_Kernels must be enabled for this property"
        old.fused_decode = False  # the 0.3.0 path
        _CACHE[key] = (fused, old)
    return _CACHE[key]


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    _STATS.update(examples=0, near_ties=0)
    yield tmp_path_factory.mktemp("fused_props")
    _CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _prefix_logits(model, prompt, toks):
    cache = model.new_static_cache(len(prompt) + len(toks) + 1)
    return model.forward(torch.tensor([prompt + toks]), cache)[0, -1].float().cpu()


model_args = dict(
    shape=st.sampled_from(MODEL_SHAPES),
    gs=st.sampled_from([32, 64]),
    act_order=st.booleans(),
    mix=st.sampled_from(["int4", "mixed"]),
)


@MODEL
@given(**model_args, prompt=st.lists(st.integers(1, 15), min_size=1, max_size=24), new=st.integers(2, 24))
def test_fused_greedy_tokens_equal_030_path(model_root, shape, gs, act_order, mix, prompt, new):
    """Feature: quf-quantized-format, Property 40: Fused decode greedy tokens equal the 0.3.0 path

    **Validates: Requirements 37.2, 37.4, 37.6, 37.7**
    """
    from quf.runtime.generate import generate

    fused, old = _pair(model_root, (shape, gs, act_order, mix))
    _STATS["examples"] += 1
    near_tie = False
    for graph in (True, False):
        a = generate(fused, None, prompt, max_new_tokens=new, cuda_graph=graph)
        b = generate(old, None, prompt, max_new_tokens=new, cuda_graph=graph)
        assert len(a) == len(b) == new
        if a != b:
            i = next(j for j in range(new) if a[j] != b[j])
            top2 = torch.topk(_prefix_logits(old, prompt, b[:i]), 2).values
            gap = float(top2[0] - top2[1])
            assert gap < NEAR_TIE_GAP, f"step {i}: fused {a[i]} != 0.3.0 {b[i]}, top-2 gap {gap} (graph={graph})"
            near_tie = True
    _STATS["near_ties"] += int(near_tie)


def test_fused_near_ties_are_rare(model_root):
    """Runs after the Property 40 test (definition order)."""
    n = _STATS["examples"]
    if n == 0:
        pytest.skip("property test did not run")
    print(f"near-ties: {_STATS['near_ties']} of {n}")
    assert _STATS["near_ties"] <= MAX_NEAR_TIE_FRACTION * n


@settings(max_examples=20, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    **model_args,
    prompt=st.lists(st.integers(1, 15), min_size=1, max_size=16),
    new=st.integers(2, 16),
    temperature=st.floats(0.2, 2.0),
    top_k=st.one_of(st.none(), st.integers(1, 16)),
    seed=st.integers(0, 10**6),
)
def test_seeded_sampling_is_deterministic_on_model(
    model_root, shape, gs, act_order, mix, prompt, new, temperature, top_k, seed
):
    """Feature: quf-quantized-format, Property 42: Seeded sampling is deterministic

    **Validates: Requirements 37.5**
    """
    from quf.runtime.generate import generate

    fused, _ = _pair(model_root, (shape, gs, act_order, mix))
    for graph in (True, False):
        kw = dict(max_new_tokens=new, temperature=temperature, top_k=top_k, seed=seed, cuda_graph=graph)
        first = generate(fused, None, prompt, **kw)
        assert generate(fused, None, prompt, **kw) == first, f"graph={graph}"


@KERNEL
@given(
    vocab=st.integers(2, 4000), temperature=st.floats(0.05, 3.0), seed=st.integers(0, 10**6),
    steps=st.integers(1, 6), top_k=st.one_of(st.none(), st.integers(1, 50)),
    top_p=st.one_of(st.none(), st.floats(0.05, 1.0)),
)
def test_seeded_sampler_is_deterministic(vocab, temperature, seed, steps, top_k, top_p):
    """Feature: quf-quantized-format, Property 42: Seeded sampling is deterministic

    **Validates: Requirements 37.5**
    """
    from quf.runtime.generate import sample_next

    logits = torch.randn(steps, vocab, device="cuda", generator=_gen(seed + 1)) * 3
    runs = []
    for _ in range(2):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        runs.append([
            sample_next(logits[i], temperature=temperature, top_k=top_k, top_p=top_p, generator=gen)
            for i in range(steps)
        ])
    assert runs[0] == runs[1]
    assert all(0 <= t < vocab for t in runs[0])
