"""Kernel dispatcher routing (task 28.3). Requirements 33.6, 33.11"""
from __future__ import annotations

import dataclasses
import logging

import pytest
import torch
import torch.nn.functional as F

from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import dequantize_int8, quantize_groupwise, quantize_int8
from quf.runtime.backends.base import Int8Weight
from quf.runtime.kernels import CachedFp16Weight, Int4Layout, kernel_linear, prepare_for_kernel


def _int8(n=32, k=256):
    w = torch.randn(n, k)
    q, s = quantize_int8(w)
    return Int8Weight(q, s)


def test_int8_routes_to_cached_fp16_and_logs(caplog):
    w8 = _int8()
    with caplog.at_level(logging.DEBUG, logger="quf.runtime.kernels.dispatch"):
        p = prepare_for_kernel(w8, name="model.layers.23.mlp.down_proj")
    assert isinstance(p, CachedFp16Weight) and p.weight.dtype == torch.float16
    assert "model.layers.23.mlp.down_proj" in caplog.text
    torch.testing.assert_close(p.weight, dequantize_int8(w8.q, w8.scales))


def test_fp16_routes_to_cached_fp16():
    w = torch.randn(16, 64).half()
    p = prepare_for_kernel(w, name="lm_head")
    assert isinstance(p, CachedFp16Weight) and torch.equal(p.weight, w)


def test_int4_routes_to_layout():
    qw = quantize_groupwise(torch.randn(64, 256), group_size=128, symmetric=True)
    assert isinstance(prepare_for_kernel(qw), Int4Layout)


def test_unsupported_group_size_falls_back(caplog):
    qw = quantize_groupwise(torch.randn(16, 48), group_size=16, symmetric=True)
    with caplog.at_level(logging.DEBUG, logger="quf.runtime.kernels.dispatch"):
        p = prepare_for_kernel(qw, name="odd")
    assert isinstance(p, CachedFp16Weight) and "odd" in caplog.text
    torch.testing.assert_close(p.weight, reference_dequantize(qw))


def test_int8_output_matches_reference_dequant():
    w8 = _int8()
    x = torch.randn(5, 256).half()
    out = kernel_linear(x, prepare_for_kernel(w8))
    ref = F.linear(x.float(), dequantize_int8(w8.q, w8.scales).float())
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_int8_large_m_works():
    p = prepare_for_kernel(_int8())
    assert kernel_linear(torch.randn(40, 256).half(), p).shape == (40, 32)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.parametrize("m", [17, 64, 512])
@pytest.mark.parametrize("n,k", [(896, 896), (1152, 896), (896, 4864), (1536, 1536), (1536, 8960)])
@pytest.mark.parametrize("symmetric", [True, False])
def test_gpu_int4_prefill_matches_fp32_reference(m, n, k, symmetric):
    torch.manual_seed(m + n)
    qw = quantize_groupwise(torch.randn(n, k), group_size=128, symmetric=symmetric)
    x = torch.randn(m, k, device="cuda").half()
    out = kernel_linear(x, prepare_for_kernel(qw, device="cuda"))
    ref = F.linear(x.float(), reference_dequantize(qw).float().cuda())
    assert out.shape == (m, n)
    # fp16 output rounding: error bounded relative to the largest output magnitude
    assert (out.float() - ref).abs().max() <= 2e-3 * ref.abs().max()


@pytest.mark.gpu
@pytest.mark.triton
def test_gpu_int4_prefill_act_order_and_buffer_reuse():
    from quf.runtime.kernels import dispatch

    torch.manual_seed(1)
    n, k, gs = 64, 256, 64
    qw = quantize_groupwise(torch.randn(n, k), group_size=gs, symmetric=False)
    g_idx = torch.arange(k) % (k // gs)  # interleaved groups: non-trivial perm
    qw = dataclasses.replace(qw, g_idx=g_idx.to(torch.int32))
    layout = prepare_for_kernel(qw, device="cuda")
    assert layout.perm is not None
    w = dispatch.dequantize_into_buffer(layout)
    torch.testing.assert_close(w, reference_dequantize(qw, qw.g_idx).cuda(), atol=0, rtol=0)
    x = torch.randn(20, k, device="cuda").half()
    torch.testing.assert_close(
        kernel_linear(x, layout).float(), F.linear(x.float(), reference_dequantize(qw, qw.g_idx).float().cuda()),
        atol=5e-2, rtol=5e-3,
    )
    dispatch.release_prefill_buffers()
    big = prepare_for_kernel(quantize_groupwise(torch.randn(128, 256), group_size=64, symmetric=True), device="cuda")
    small = prepare_for_kernel(quantize_groupwise(torch.randn(32, 256), group_size=64, symmetric=True), device="cuda")
    x = torch.randn(20, 256, device="cuda").half()
    kernel_linear(x, big)
    nbytes = dispatch.prefill_buffer_bytes()
    kernel_linear(x, small)
    kernel_linear(x, big)
    assert dispatch.prefill_buffer_bytes() == nbytes == 128 * 256 * 2


@pytest.mark.gpu
@pytest.mark.triton
def test_gpu_mixed_routing_matches_reference():
    torch.manual_seed(0)
    w8 = _int8(64, 256)
    qw = quantize_groupwise(torch.randn(64, 256), group_size=128, symmetric=False)
    x = torch.randn(3, 256, device="cuda").half()
    p8 = prepare_for_kernel(w8, device="cuda")
    p4 = prepare_for_kernel(qw, device="cuda")
    assert isinstance(p8, CachedFp16Weight) and isinstance(p4, Int4Layout)
    ref8 = F.linear(x.float(), dequantize_int8(w8.q, w8.scales).float().cuda())
    torch.testing.assert_close(kernel_linear(x, p8).float(), ref8, atol=2e-2, rtol=2e-2)
    ref4 = F.linear(x.float(), reference_dequantize(qw).float().cuda())
    torch.testing.assert_close(kernel_linear(x, p4).float(), ref4, atol=2e-2, rtol=5e-3)


def test_default_info_level_has_no_per_module_lines(caplog):
    qw = quantize_groupwise(torch.randn(16, 48), group_size=16, symmetric=True)
    with caplog.at_level(logging.INFO, logger="quf.runtime.kernels.dispatch"):
        prepare_for_kernel(_int8(), name="a")
        prepare_for_kernel(qw, name="b")
        prepare_for_kernel(torch.randn(4, 8).half(), name="c")
    assert caplog.text == ""


def test_route_summary_counts():
    from quf.runtime.kernels.dispatch import route_summary

    p4 = prepare_for_kernel(quantize_groupwise(torch.randn(64, 256), group_size=128, symmetric=True))
    p8 = prepare_for_kernel(_int8())
    assert route_summary([p4, p4, p8]) == "int4 kernel: 2 modules; cached fp16: 1 (int8)"
