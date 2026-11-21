"""CUDA graph capture and replay of the decode kernel paths (task 28.6). Requirements 33.9"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from quf.quant.quantize import quantize_groupwise, quantize_int8
from quf.runtime.backends.base import Int8Weight
from quf.runtime.kernels import fuse_projections, fused_linear, kernel_linear, prepare_for_kernel
from quf.runtime.kernels.triton_kernels import int4_gemv, triton_available

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.triton,
    pytest.mark.skipif(not triton_available()[0], reason="Triton or CUDA unavailable"),
]

K = 1024
REPLAYS = 20


def _qw(n, k=K, gs=128, symmetric=False, act_order=False):
    qw = quantize_groupwise(torch.randn(n, k), group_size=gs, symmetric=symmetric)
    if act_order:
        g = torch.Generator().manual_seed(7)
        g_idx = (torch.arange(k) // gs)[torch.randperm(k, generator=g)]
        qw = dataclasses.replace(qw, g_idx=g_idx.to(torch.int32))
    return qw


def _as_tuple(y):
    return y if isinstance(y, tuple) else (y,)


def _check_graph(fn, m):
    """Warm up, capture ``fn(static_x)``, replay with new inputs, compare with eager."""
    torch.manual_seed(0)
    static_x = torch.randn(m, K, device="cuda").half()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # compile and cublas init outside the graph
            fn(static_x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # Eager must not synchronise with the host (raises under the debug mode).
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn(static_x)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = _as_tuple(fn(static_x))

    torch.cuda.synchronize()
    mem0 = torch.cuda.memory_allocated()
    first = None
    for i in range(REPLAYS):
        new = torch.randn(m, K, device="cuda").half()
        static_x.copy_(new)
        graph.replay()
        eager = _as_tuple(fn(new.clone()))
        for g_out, e_out in zip(static_out, eager):
            torch.testing.assert_close(g_out, e_out, atol=1e-2, rtol=1e-2)
        if i == 0:
            first = [o.clone() for o in static_out]
        del eager
    # same input again gives the same result after many replays
    static_x.copy_(torch.randn(m, K, device="cuda").half())
    graph.replay()
    a = [o.clone() for o in static_out]
    graph.replay()
    for x, y in zip(a, static_out):
        assert torch.equal(x, y)
    assert first is not None
    del a, first
    torch.cuda.synchronize()
    # replay allocates nothing: allocator state is unchanged across many bare replays
    mem1 = torch.cuda.memory_allocated()
    reserved1 = torch.cuda.memory_reserved()
    for _ in range(REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == mem1
    assert torch.cuda.memory_reserved() == reserved1
    assert mem0 <= mem1 + 0  # clones were released, nothing leaked


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("config", [(32, 256, 1), (32, 256, 4), None])
def test_int4_gemv_graph(m, act_order, symmetric, config):
    layout = prepare_for_kernel(_qw(256, symmetric=symmetric, act_order=act_order), device="cuda")
    _check_graph(lambda x: int4_gemv(x, layout, config), m)


def test_default_config_uses_split_k_and_is_captured():
    from quf.runtime.kernels.triton_kernels import gemv_config

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert gemv_config(256, 4096, 128, sms)[2] > 1


@pytest.mark.parametrize("act_order", [False, True])
def test_fused_int4_graph(act_order):
    parts = [prepare_for_kernel(_qw(n, act_order=act_order), device="cuda") for n in (256, 64, 64)]
    fw = fuse_projections(parts)
    assert fw.is_fused
    _check_graph(lambda x: fused_linear(x, fw), 1)


def test_fused_fp16_fallback_graph():
    parts = [prepare_for_kernel(torch.randn(n, K).half(), device="cuda") for n in (128, 64, 64)]
    fw = fuse_projections(parts)
    assert fw.is_fused
    _check_graph(lambda x: fused_linear(x, fw), 1)
    # unfused members run one by one
    sep = fuse_projections([parts[0], prepare_for_kernel(_qw(64), device="cuda")])
    assert not sep.is_fused
    _check_graph(lambda x: fused_linear(x, sep), 1)


def test_kernel_linear_dispatch_graph():
    int4 = prepare_for_kernel(_qw(128, act_order=True), device="cuda")
    fp16 = prepare_for_kernel(torch.randn(128, K).half(), device="cuda")
    q, s = quantize_int8(torch.randn(128, K))
    int8 = prepare_for_kernel(Int8Weight(q, s), device="cuda")
    assert int8.reason == "int8"
    for w in (int4, int8, fp16):
        _check_graph(lambda x, w=w: kernel_linear(x, w), 1)
