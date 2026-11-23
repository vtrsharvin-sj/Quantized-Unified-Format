"""Task 33.2 part B: j-major int4 GEMV, SiLU*mul scatter, fused RoPE + KV write, flash-decode (GPU + Triton).

Each kernel is compared with its torch reference and must be CUDA-graph capturable.

**Validates: Requirements 37.2, 37.6, 37.7**
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from quf.runtime.kernels.triton_kernels import Int4Layout, dequantize_layout, triton_available

pytestmark = [pytest.mark.gpu]

needs_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available()[0], reason="needs CUDA and Triton"
)


def _layout(n, k, gs, perm=True, zeros=True, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda", generator=g)
    sc = (torch.rand(n, k // gs, device="cuda", generator=g) * 0.02 + 0.001).half()
    zr = torch.randint(0, 16, (n, k // gs), device="cuda", generator=g).half() if zeros else None
    p = torch.randperm(k, device="cuda", generator=g) if perm else None
    return Int4Layout(packed, sc, zr, p, gs, k, n)


@needs_triton
@pytest.mark.parametrize("n,k,gs", [(1152, 896, 32), (896, 4864, 32), (130, 512, 128), (257, 640, 64), (64, 256, 32)])
@pytest.mark.parametrize("perm,zeros,bias", [(True, True, True), (False, False, False), (True, False, True)])
def test_gemv_jmajor_matches_dequantized_reference(n, k, gs, perm, zeros, bias):
    from quf.runtime.kernels.decode_gemv import FastLinear, to_jmajor

    lay = _layout(n, k, gs, perm, zeros, seed=n + k)
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16) if bias else None
    fl = FastLinear(lay, b)
    y = fl(to_jmajor(x, lay))
    ref = x.float() @ dequantize_layout(lay).float().t()
    if bias:
        ref = ref + b.float()
    assert y.shape == (1, n) and y.dtype == torch.float16
    assert (y.float() - ref).abs().max().item() < 1e-2 * ref.abs().max().item() + 1e-3


@needs_triton
def test_gemv_jmajor_agrees_with_old_kernel_and_is_graph_capturable():
    from quf.runtime.kernels.decode_gemv import FastLinear, to_jmajor
    from quf.runtime.kernels.triton_kernels import int4_gemv

    lay = _layout(896, 896, 32, True, True, seed=3)
    fl = FastLinear(lay, None)
    x = torch.randn(1, 896, device="cuda", dtype=torch.float16)
    xj = to_jmajor(x, lay)
    fl(xj)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fl(xj)
    xj.copy_(to_jmajor(2 * x, lay))
    g.replay()
    ref = int4_gemv(2 * x, lay)
    assert (y.float() - ref.float()).abs().max().item() < 2e-2 * ref.abs().max().item()


@needs_triton
@pytest.mark.parametrize("config", [(8, 32, 2, 2, 4), (4, 16, 1, 4, 3), (1, 8, 1, 2, 7), (2, 16, 1, 2, 1)])
def test_gemv_jmajor_split_k_matches_reference_over_graph_replays(config):
    """Split-K with the in-kernel reduce (task 39.5): equal to the reference, deterministic across
    replays, and the per-block counter resets itself so later replays stay correct."""
    from quf.runtime.kernels.decode_gemv import FastLinear, to_jmajor

    lay = _layout(260, 4864, 32, True, True, seed=11)
    b = torch.randn(260, device="cuda", dtype=torch.float16)
    fl = FastLinear(lay, b)
    fl.config = config
    x = torch.randn(1, 4864, device="cuda", dtype=torch.float16)
    xj = to_jmajor(x, lay)
    first = fl(xj).clone()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fl(xj)
    for scale in (1.0, -3.0, 0.5):
        xj.copy_(to_jmajor(scale * x, lay))
        g.replay()
        g.replay()
        ref = (scale * x).float() @ dequantize_layout(lay).float().t() + b.float()
        assert (y.float() - ref).abs().max().item() < 1e-2 * ref.abs().max().item() + 1e-3
    xj.copy_(to_jmajor(x, lay))
    g.replay()
    assert torch.equal(y, first)  # fixed-order reduce: bitwise equal to the eager call


@needs_triton
def test_silu_mul_scatter_and_add_rmsnorm_index():
    from quf.runtime.kernels.decode_gemv import activation_index, silu_mul_scatter, to_jmajor
    from quf.runtime.kernels.fused import add_rmsnorm

    lay = _layout(64, 4864, 32, True, True)
    idx = activation_index(lay)
    gu = torch.randn(1, 2 * 4864, device="cuda", dtype=torch.float16)
    out = silu_mul_scatter(gu, idx)
    ref = to_jmajor(F.silu(gu[:, :4864]) * gu[:, 4864:], lay)
    assert (out.float() - ref.float()).abs().max().item() <= 2e-3 * ref.abs().max().item()

    lay2 = _layout(64, 896, 32, True, True, seed=9)
    x = torch.randn(1, 1, 896, device="cuda", dtype=torch.float16)
    res = torch.randn_like(x)
    w = torch.rand(896, device="cuda", dtype=torch.float16) + 0.5
    plain, new_res = add_rmsnorm(x, res, w, 1e-6)
    scat, new_res2 = add_rmsnorm(x, res, w, 1e-6, out_index=activation_index(lay2))
    assert torch.equal(new_res, new_res2)
    assert torch.equal(scat.view(1, -1), to_jmajor(plain.view(1, -1), lay2))


def _rope_tables(t, d):
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, dtype=torch.float32, device="cuda") / d))
    freqs = torch.arange(t, device="cuda", dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().half(), emb.sin().half()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


@needs_triton
@pytest.mark.parametrize("h,kv,d", [(14, 2, 64), (12, 2, 128)])
def test_rope_kv_write_matches_torch_exactly(h, kv, d):
    from quf.runtime.kernels.attention import rope_kv_write

    t = 64
    cos, sin = _rope_tables(t, d)
    qkv = torch.randn(1, (h + 2 * kv) * d, device="cuda", dtype=torch.float16)
    kc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16)
    vc = torch.randn_like(kc)
    kc0, vc0 = kc.clone(), vc.clone()
    pos = torch.tensor([37], device="cuda")
    q = rope_kv_write(qkv, cos, sin, pos, kc, vc, h, kv)

    qq = qkv[:, : h * d].view(1, 1, h, d).transpose(1, 2)
    kk = qkv[:, h * d : (h + kv) * d].view(1, 1, kv, d).transpose(1, 2)
    vv = qkv[:, (h + kv) * d :].view(1, 1, kv, d).transpose(1, 2)
    c, s = cos[37], sin[37]
    q_ref = qq * c + _rotate_half(qq) * s
    k_ref = kk * c + _rotate_half(kk) * s
    assert torch.equal(q.view(1, h, 1, d), q_ref)
    kc0.index_copy_(2, pos, k_ref)
    vc0.index_copy_(2, pos, vv)
    assert torch.equal(kc, kc0) and torch.equal(vc, vc0)


def _attn_ref(q, kc, vc, pos, h, kv, d):
    rep = h // kv
    k = kc[:, :, : pos + 1].repeat_interleave(rep, dim=1)
    v = vc[:, :, : pos + 1].repeat_interleave(rep, dim=1)
    sc = (q.view(1, h, 1, d).float() @ k.float().transpose(-1, -2)) / math.sqrt(d)
    p = torch.softmax(sc, dim=-1).to(v.dtype)
    return (p @ v).reshape(1, h * d)


@needs_triton
@pytest.mark.parametrize("h,kv,d,t", [(14, 2, 64, 132), (12, 2, 128, 132), (14, 2, 64, 1500), (12, 2, 128, 2100)])
@pytest.mark.parametrize("pos", [0, 5, 131, 1030])
def test_flash_decode_matches_torch(h, kv, d, t, pos):
    from quf.runtime.kernels.attention import flash_decode

    if pos >= t:
        pytest.skip("position outside this cache")
    g = torch.Generator(device="cuda").manual_seed(h + t + pos)
    q = torch.randn(h, d, device="cuda", dtype=torch.float16, generator=g)
    kc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16, generator=g)
    vc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16, generator=g)
    p = torch.tensor([pos], device="cuda")
    out = flash_decode(q, kc, vc, p, h, kv)
    ref = _attn_ref(q, kc, vc, pos, h, kv, d)
    assert (out.float() - ref.float()).abs().max().item() < 4e-3 + 4e-3 * ref.abs().max().item()
    # scattered store equals the torch scatter of the same result
    idx = torch.randperm(h * d, device="cuda", dtype=torch.int64).to(torch.int32)
    out2 = flash_decode(q, kc, vc, p, h, kv, out_index=idx)
    exp = torch.empty_like(out)
    exp[0, idx.long()] = out[0]
    assert torch.equal(out2, exp)


@needs_triton
def test_attention_kernels_are_graph_capturable_and_follow_the_position_tensor():
    from quf.runtime.kernels.attention import flash_decode, rope_kv_write

    h, kv, d, t = 14, 2, 64, 700
    cos, sin = _rope_tables(t, d)
    qkv = torch.randn(1, (h + 2 * kv) * d, device="cuda", dtype=torch.float16)
    kc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16)
    vc = torch.randn_like(kc)
    pos = torch.tensor([10], device="cuda")

    def step():
        q = rope_kv_write(qkv, cos, sin, pos, kc, vc, h, kv)
        return flash_decode(q, kc, vc, pos, h, kv)

    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    for p in (10, 600):  # 600 crosses into the second split
        pos.fill_(p)
        g.replay()
        torch.cuda.synchronize()
        qq = qkv[:, : h * d].view(1, 1, h, d).transpose(1, 2)
        q_ref = (qq * cos[p] + _rotate_half(qq) * sin[p]).reshape(h, d)
        ref = _attn_ref(q_ref, kc, vc, p, h, kv, d)
        assert (out.float() - ref.float()).abs().max().item() < 4e-3 + 4e-3 * ref.abs().max().item()
