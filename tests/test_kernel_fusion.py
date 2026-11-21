"""Fused projection layouts (task 28.5). Requirements 33.8, 33.10"""
from __future__ import annotations

import dataclasses
import logging

import pytest
import torch

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import quantize_groupwise, quantize_int8
from quf.runtime.backends.base import Int8Weight
from quf.runtime.kernels import (
    CachedFp16Weight,
    Int4Layout,
    fuse_projections,
    fused_linear,
    prepare_for_kernel,
    split_output,
)
from quf.runtime.kernels.fusion import reset_fusion_log
from quf.runtime.kernels.triton_kernels import dequantize_layout, int4_gemv


def _qw(n, k, gs=128, symmetric=True, g_idx=None):
    qw = quantize_groupwise(torch.randn(n, k), group_size=gs, symmetric=symmetric)
    if g_idx is not None:
        qw = dataclasses.replace(qw, g_idx=g_idx.to(torch.int32))
    return qw


def _shuffled_g_idx(k, gs, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.arange(k) // gs)[torch.randperm(k, generator=g)]


def test_fuse_concatenates_and_splits_cpu():
    torch.manual_seed(0)
    parts = [prepare_for_kernel(_qw(n, 256, 64)) for n in (128, 32, 32)]
    fw = fuse_projections(parts)
    assert fw.is_fused and fw.parts is None and fw.widths == (128, 32, 32)
    assert fw.fused.out_features == 192 and fw.fused.packed.shape == (192, 32)
    ref = torch.cat([dequantize_layout(p) for p in parts], dim=0)
    torch.testing.assert_close(dequantize_layout(fw.fused), ref, atol=0, rtol=0)
    y = torch.randn(3, 192)
    a, b, c = split_output(y, fw.widths)
    assert (a.shape[-1], b.shape[-1], c.shape[-1]) == (128, 32, 32) and torch.equal(torch.cat([a, b, c], -1), y)


def test_mixed_symmetric_asymmetric_materializes_zeros():
    sym = prepare_for_kernel(_qw(64, 256, 64, symmetric=True))
    asym = prepare_for_kernel(_qw(32, 256, 64, symmetric=False))
    assert sym.zeros is None and asym.zeros is not None
    fw = fuse_projections([sym, asym, sym])
    assert fw.is_fused and fw.fused.zeros is not None
    assert torch.all(fw.fused.zeros[:64] == 8) and torch.equal(fw.fused.zeros[64:96], asym.zeros)
    ref = torch.cat([dequantize_layout(sym), dequantize_layout(asym), dequantize_layout(sym)], dim=0)
    torch.testing.assert_close(dequantize_layout(fw.fused), ref, atol=0, rtol=0)


def test_all_symmetric_keeps_zeros_none():
    parts = [prepare_for_kernel(_qw(32, 256, 64)) for _ in range(2)]
    assert fuse_projections(parts).fused.zeros is None


def test_matching_perms_fuse_and_keep_perm():
    g = _shuffled_g_idx(256, 64, 1)
    parts = [prepare_for_kernel(_qw(n, 256, 64, False, g)) for n in (64, 32)]
    fw = fuse_projections(parts)
    assert fw.is_fused and fw.fused.perm is not None
    assert torch.equal(fw.fused.perm, parts[0].perm)
    ref = torch.cat([dequantize_layout(p) for p in parts], 0)
    torch.testing.assert_close(dequantize_layout(fw.fused), ref, atol=0, rtol=0)


def test_perm_mismatch_refuses_and_logs_once(caplog):
    reset_fusion_log()
    a = prepare_for_kernel(_qw(32, 256, 64, True, _shuffled_g_idx(256, 64, 1)))
    b = prepare_for_kernel(_qw(32, 256, 64, True, _shuffled_g_idx(256, 64, 2)))
    with caplog.at_level(logging.INFO, logger="quf.runtime.kernels.fusion"):
        fw1 = fuse_projections([a, b], name="layer0.qkv")
        fw2 = fuse_projections([a, b], name="layer1.qkv")
    assert not fw1.is_fused and fw1.parts == (a, b) and not fw2.is_fused
    assert len([r for r in caplog.records if "permutations differ" in r.getMessage()]) == 1


def test_perm_vs_no_perm_refuses():
    a = prepare_for_kernel(_qw(32, 256, 64, True, _shuffled_g_idx(256, 64, 1)))
    b = prepare_for_kernel(_qw(32, 256, 64, True))
    assert not fuse_projections([a, b]).is_fused


def test_identity_perm_equals_no_perm():
    ident = torch.arange(256) // 64
    a = prepare_for_kernel(_qw(32, 256, 64, True, ident))
    b = prepare_for_kernel(_qw(32, 256, 64, True))
    fw = fuse_projections([a, b])
    assert fw.is_fused and fw.fused.perm is None


def test_other_mismatches_refuse():
    a = prepare_for_kernel(_qw(32, 256, 64))
    assert not fuse_projections([a, prepare_for_kernel(_qw(32, 256, 128))]).is_fused  # group size
    assert not fuse_projections([a, prepare_for_kernel(_qw(32, 128, 64))]).is_fused  # K


def test_fp16_members_fuse_and_int8_mix_falls_back():
    w1, w2 = torch.randn(16, 64).half(), torch.randn(8, 64).half()
    fw = fuse_projections([prepare_for_kernel(w1), prepare_for_kernel(w2)])
    assert fw.is_fused and isinstance(fw.fused, CachedFp16Weight) and torch.equal(fw.fused.weight, torch.cat([w1, w2]))
    q, s = quantize_int8(torch.randn(8, 256))
    p8 = prepare_for_kernel(Int8Weight(q, s))
    p4 = prepare_for_kernel(_qw(32, 256, 64))
    fw = fuse_projections([p4, p8])
    assert not fw.is_fused and fw.parts == (p4, p8)
    assert fw.widths == (32, 8)


def test_fp16_fused_linear_matches_separate():
    w1, w2 = torch.randn(16, 64).half(), torch.randn(8, 64).half()
    x = torch.randn(3, 64).half()
    fw = fuse_projections([prepare_for_kernel(w1), prepare_for_kernel(w2)])
    a, b = fused_linear(x, fw)
    torch.testing.assert_close(a, x @ w1.T)
    torch.testing.assert_close(b, x @ w2.T)


# ------------------------------- GPU -------------------------------------------------------

# (K, widths): Qwen2.5-0.5B qkv, gate/up, and a 1.5B-like qkv
SHAPES = [(896, (896, 128, 128)), (896, (4864, 4864)), (1536, (1536, 256, 256))]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("k,widths", SHAPES)
@pytest.mark.parametrize("act_order", [False, True])
def test_gpu_fused_matches_separate_gemv(m, k, widths, act_order):
    torch.manual_seed(k + m)
    g = _shuffled_g_idx(k, 128, 3) if act_order else None
    parts = [prepare_for_kernel(_qw(n, k, 128, symmetric=False, g_idx=g), device="cuda") for n in widths]
    fw = fuse_projections(parts)
    assert fw.is_fused
    x = torch.randn(m, k, device="cuda").half()
    sep = [int4_gemv(x, p) for p in parts]
    outs = fused_linear(x, fw)
    for o, s in zip(outs, sep):
        assert o.shape == s.shape
        assert (o.float() - s.float()).abs().max() <= 1e-3 * max(1.0, s.float().abs().max().item())


@pytest.mark.gpu
@pytest.mark.triton
def test_gpu_mixed_sym_asym_correct():
    torch.manual_seed(5)
    k = 896
    qws = [_qw(896, k, 128, True), _qw(128, k, 128, False), _qw(128, k, 128, True)]
    parts = [prepare_for_kernel(q, device="cuda") for q in qws]
    fw = fuse_projections(parts)
    assert fw.is_fused and fw.fused.zeros is not None
    x = torch.randn(4, k, device="cuda").half()
    outs = fused_linear(x, fw)
    for o, q in zip(outs, qws):
        ref = x.float() @ reference_dequantize(q).float().cuda().T
        assert (o.float() - ref).abs().max() <= 2e-3 * ref.abs().max()


@pytest.mark.gpu
@pytest.mark.triton
def test_gpu_perm_mismatch_falls_back_and_stays_correct():
    torch.manual_seed(6)
    k = 896
    qws = [_qw(128, k, 128, False, _shuffled_g_idx(k, 128, s)) for s in (1, 2, 3)]
    parts = [prepare_for_kernel(q, device="cuda") for q in qws]
    fw = fuse_projections(parts)
    assert not fw.is_fused
    x = torch.randn(2, k, device="cuda").half()
    for o, q in zip(fused_linear(x, fw), qws):
        ref = x.float() @ reference_dequantize(q, q.g_idx).float().cuda().T
        assert (o.float() - ref).abs().max() <= 2e-3 * ref.abs().max()
