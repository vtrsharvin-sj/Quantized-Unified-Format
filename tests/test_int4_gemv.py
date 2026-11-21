"""GPU tests for the W4A16 Triton GEMV against the pure-torch layout reference."""

from __future__ import annotations

import pytest
import torch

from quf.runtime.kernels.triton_kernels import Int4Layout, dequantize_layout, int4_gemv

pytestmark = [pytest.mark.gpu, pytest.mark.triton]

# (name, N, K) for Qwen2.5-0.5B (hidden 896, inter 4864) and 1.5B (hidden 1536, inter 8960).
SHAPES = [
    ("0.5b.q", 896, 896), ("0.5b.kv", 128, 896), ("0.5b.qkv", 1152, 896),
    ("0.5b.gate_up", 9728, 896), ("0.5b.down", 896, 4864),
    ("1.5b.q", 1536, 1536), ("1.5b.kv", 256, 1536), ("1.5b.qkv", 2048, 1536),
    ("1.5b.gate", 8960, 1536), ("1.5b.gate_up", 17920, 1536), ("1.5b.down", 1536, 8960),
]


def make_layout(n: int, k: int, gs: int, asym: bool, act_order: bool = False, seed: int = 0) -> Int4Layout:
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda", generator=g)
    scales = (torch.rand(n, k // gs, device="cuda", generator=g) * 0.02 + 0.001).to(torch.float16)
    zeros = torch.randint(0, 16, (n, k // gs), device="cuda", generator=g).to(torch.float16) if asym else None
    perm = torch.randperm(k, device="cuda", generator=g) if act_order else None
    return Int4Layout(packed, scales, zeros, perm, gs, k, n)


def check(layout: Int4Layout, m: int, seed: int = 1) -> None:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, layout.in_features, device="cuda", generator=g).to(torch.float16)
    ref = x.float() @ dequantize_layout(layout).float().T
    out = int4_gemv(x, layout)
    assert out.dtype == torch.float16 and out.shape == (m, layout.out_features)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=2e-3)


@pytest.mark.parametrize("name,n,k", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_real_shapes_m1(name: str, n: int, k: int, asym: bool) -> None:
    check(make_layout(n, k, 128, asym), 1)


@pytest.mark.parametrize("gs", [32, 64, 128])
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("m", [1, 3, 16])
def test_group_sizes_and_batch(gs: int, asym: bool, m: int) -> None:
    check(make_layout(896, 4864, gs, asym), m)


def test_act_order_perm() -> None:
    check(make_layout(896, 896, 128, True, act_order=True), 2)


def test_m_out_of_range() -> None:
    layout = make_layout(128, 896, 128, False)
    with pytest.raises(ValueError):
        int4_gemv(torch.zeros(17, 896, dtype=torch.float16, device="cuda"), layout)
