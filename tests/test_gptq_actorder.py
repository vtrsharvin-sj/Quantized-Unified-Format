"""Act-order, group size, and clipping for Native_GPTQ (Requirements 20.1, 20.2, 21.3, 21.4)."""
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quf.quant import gptq as G
from quf.quant.dequant import reference_dequantize

CPU = torch.device("cpu")


def _hessian(n, seed, samples=256):
    g = torch.Generator().manual_seed(seed)
    scale = torch.rand(n, generator=g) * 4 + 0.1  # uneven diagonal
    x = torch.randn(samples, n, generator=g) * scale
    return 2.0 * x.t() @ x / samples


# Feature: quf-quantized-format, Property 27: act-order processes columns in descending Hessian diagonal
# **Validates: Requirements 20.1, 20.2**
@settings(max_examples=50, deadline=None)
@given(
    gs=st.sampled_from([32, 64, 128]),
    groups=st.integers(1, 4),
    seed=st.integers(0, 10_000),
)
def test_property_act_order_permutation(gs, groups, seed):
    n = gs * groups
    diag = torch.rand(n, generator=torch.Generator().manual_seed(seed))
    perm, g_idx = G.act_order_permutation(diag, gs)
    assert sorted(perm.tolist()) == list(range(n))
    d = diag[perm]
    assert bool((d[:-1] >= d[1:]).all())
    assert g_idx.dtype == torch.int32 and g_idx.shape == (n,)
    counts = torch.bincount(g_idx.long(), minlength=groups)
    assert bool((counts == gs).all())


@pytest.mark.parametrize("gs", [32, 64, 128])
def test_act_order_g_idx_and_original_order(gs):
    n, out = 256, 16
    w = torch.randn(out, n, generator=torch.Generator().manual_seed(1))
    h = _hessian(n, 2)
    qw = G.gptq_quantize(w, h, group_size=gs, act_order=True, device=CPU)
    assert qw.group_size == gs
    assert qw.g_idx is not None and qw.g_idx.shape == (n,)
    assert qw.scales.shape == (n // gs, out)
    perm, g_idx = G.act_order_permutation(torch.diag(h), gs)
    assert torch.equal(qw.g_idx, g_idx)
    # Stored in original order: dequantization with g_idx approximates the original weight.
    deq = reference_dequantize(qw, qw.g_idx).float()
    assert (deq - w).abs().mean() < 0.5 * w.abs().mean()  # a column-order mixup would give ~1x


def test_without_act_order_has_no_g_idx():
    w = torch.randn(8, 128)
    qw = G.gptq_quantize(w, _hessian(128, 3), device=CPU)
    assert qw.g_idx is None


def test_act_order_not_worse_in_output_error():
    n, out = 256, 32
    w = torch.randn(out, n, generator=torch.Generator().manual_seed(4))
    h = _hessian(n, 5)

    def err(qw):
        d = (reference_dequantize(qw, qw.g_idx).float() - w)
        return torch.trace(d @ h @ d.t()).item()

    base = err(G.gptq_quantize(w, h, group_size=64, device=CPU))
    act = err(G.gptq_quantize(w, h, group_size=64, act_order=True, device=CPU))
    assert act <= base * 1.2


def test_clip_search_not_worse_weight_error_per_group():
    n, out = 128, 16
    w = torch.randn(out, n, generator=torch.Generator().manual_seed(6))
    w[:, 5] *= 12  # outliers make clipping useful
    h = torch.eye(n)  # identity Hessian: no error propagation effect on grouping
    plain = G.gptq_quantize(w, h, device=CPU)
    clip = G.gptq_quantize(w, h, clip_search=True, device=CPU)
    e0 = (reference_dequantize(plain).float() - w).pow(2).sum()
    e1 = (reference_dequantize(clip).float() - w).pow(2).sum()
    assert e1 <= e0 * 1.05
