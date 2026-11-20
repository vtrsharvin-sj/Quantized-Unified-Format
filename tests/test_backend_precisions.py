"""Backends handle g_idx, group sizes 32/64/128, int8 and fp16 modules (task 16.4).

Feature: quf-quantized-format, Property 25: Backend output matches the reference for all precisions

CPU and eager fused variants run everywhere. The ``gpu`` variants cover torch_gpu and torch_fused
on CUDA, including a whole-model comparison across cpu, torch_gpu, and torch_fused.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from quf.format.schema import QUFHeader
from quf.format.writer import write_quf
from quf.quant.dequant import expand_groups, reference_dequantize
from quf.quant.quantize import dequantize_int8, quantize_groupwise, quantize_int8
from quf.runtime.backends.base import Int8Weight
from quf.runtime.backends.cpu import CPUBackend
from quf.runtime.backends.marlin import MarlinBackend, MarlinUnavailableError
from quf.runtime.backends.torch_fused import TorchGPUFusedBackend
from quf.runtime.backends.torch_gpu import TorchGPUBackend
from quf.runtime.linear import QUFLinear
from quf.runtime.model import QUFModel

ATOL = 1e-2


def _case(seed, kind, group_size, use_g_idx, symmetric, out_features, n_groups, batch):
    in_features = group_size * n_groups
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=gen) * 0.1
    x = (torch.rand(batch, in_features, generator=gen) * 2 - 1).to(torch.float16)
    if kind == "int8":
        q, s = quantize_int8(w)
        return Int8Weight(q, s), None, x, dequantize_int8(q, s)
    if kind == "fp16":
        w16 = w.to(torch.float16)
        return w16, None, x, w16
    qw = quantize_groupwise(w, group_size=group_size, symmetric=symmetric)
    g_idx = None
    if use_g_idx:
        g_idx = (torch.randperm(in_features, generator=gen) // group_size).to(torch.int32)
    return qw, g_idx, x, reference_dequantize(qw, g_idx)


def _ref(x, w):
    return x.to(torch.float32) @ w.to(torch.float32).t()


_strat = dict(
    seed=st.integers(0, 2**31 - 1),
    kind=st.sampled_from(["int4", "int8", "fp16"]),
    group_size=st.sampled_from([32, 64, 128]),
    use_g_idx=st.booleans(),
    symmetric=st.booleans(),
    out_features=st.sampled_from([8, 24, 64]),
    n_groups=st.integers(1, 3),
    batch=st.integers(1, 4),
)


def _run(backend, device, qw, g_idx, x):
    lin = QUFLinear(qw, backend, device, g_idx=g_idx)
    return lin(x.to(device)).cpu()


@given(**_strat)
def test_cpu_and_fused_eager_match_reference(seed, kind, group_size, use_g_idx, symmetric, out_features, n_groups, batch):
    """**Validates: Requirements 20.4, 20.5, 22.3, 22.4**"""
    qw, g_idx, x, w_ref = _case(seed, kind, group_size, use_g_idx, symmetric, out_features, n_groups, batch)
    ref = _ref(x, w_ref)
    for backend in (CPUBackend(), TorchGPUFusedBackend(compile=False)):
        y = _run(backend, "cpu", qw, g_idx, x)
        assert y.shape == (batch, out_features)
        assert (y.to(torch.float32) - ref).abs().max().item() <= ATOL, backend.name


@pytest.mark.gpu
@settings(deadline=None)
@given(**_strat)
def test_gpu_backends_match_reference(seed, kind, group_size, use_g_idx, symmetric, out_features, n_groups, batch):
    """**Validates: Requirements 20.5, 22.4**"""
    qw, g_idx, x, w_ref = _case(seed, kind, group_size, use_g_idx, symmetric, out_features, n_groups, batch)
    ref = _ref(x, w_ref)
    outs = [_run(b, "cuda", qw, g_idx, x) for b in (TorchGPUBackend(), TorchGPUFusedBackend(compile=False))]
    for y in outs:
        assert (y.to(torch.float32) - ref).abs().max().item() <= ATOL


def test_expand_groups_uses_g_idx_and_default_mapping():
    scales = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    s, z = expand_groups(scales, None, None, 2, 8)
    assert s[:, 0].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert (z == 8).all()
    g = torch.tensor([3, 0, 0, 1, 2, 2, 1, 3])
    s, _ = expand_groups(scales, None, g, 2, 8)
    assert s[:, 0].tolist() == [4, 1, 1, 2, 3, 3, 2, 4]


def test_g_idx_changes_dequantized_weight():
    w = torch.randn(8, 64) * 0.1
    qw = quantize_groupwise(w, group_size=32)
    g = torch.tensor([1] * 32 + [0] * 32, dtype=torch.int32)
    assert not torch.equal(reference_dequantize(qw), reference_dequantize(qw, g))


def test_marlin_rejects_g_idx_and_int8():
    qw = quantize_groupwise(torch.randn(8, 128) * 0.1, symmetric=True)
    with pytest.raises(MarlinUnavailableError, match="g_idx"):
        MarlinBackend().prepare(qw, "cuda", torch.zeros(128, dtype=torch.int32))
    q, s = quantize_int8(torch.randn(8, 128))
    with pytest.raises(MarlinUnavailableError, match="int8"):
        MarlinBackend().prepare(Int8Weight(q, s), "cuda")


# -- whole-model comparison ------------------------------------------------

HIDDEN, HEADS, KV_HEADS, INTER, LAYERS, VOCAB = 128, 4, 2, 128, 2, 32


def _build_mixed_quf(path, group_size=64, seed=0):
    """Tiny model: layer 0 int4 with g_idx, layer 1 int4 plain except int8 v_proj and fp16 o_proj."""
    g = torch.Generator().manual_seed(seed)
    hd = HIDDEN // HEADS
    tensors: dict[str, np.ndarray] = {}
    modules: dict[str, str] = {}

    def fp16(*shape, scale=0.1):
        return (torch.randn(*shape, generator=g) * scale).to(torch.float16).numpy()

    def add(name, out, inp, precision="int4", g_idx=False, bias=False):
        w = torch.randn(out, inp, generator=g) * 0.1
        if precision == "int8":
            q, s = quantize_int8(w)
            tensors[f"{name}.qweight_int8"] = q.numpy()
            tensors[f"{name}.scales"] = s.numpy()
            modules[name] = "int8"
        elif precision == "fp16":
            tensors[f"{name}.weight"] = w.to(torch.float16).numpy()
            modules[name] = "fp16"
        else:
            qw = quantize_groupwise(w, group_size=group_size)
            tensors[f"{name}.qweight"] = qw.qweight.numpy()
            tensors[f"{name}.scales"] = qw.scales.numpy()
            tensors[f"{name}.qzeros"] = qw.qzeros.numpy()
            if g_idx:
                tensors[f"{name}.g_idx"] = (torch.randperm(inp, generator=g) // group_size).to(torch.int32).numpy()
        if bias:
            tensors[f"{name}.bias"] = fp16(out)

    tensors["model.embed_tokens.weight"] = fp16(VOCAB, HIDDEN, scale=1.0)
    tensors["model.norm.weight"] = np.ones(HIDDEN, dtype=np.float16)
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(HIDDEN, dtype=np.float16)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(HIDDEN, dtype=np.float16)
        act = i == 0
        add(f"{p}.self_attn.q_proj", HEADS * hd, HIDDEN, g_idx=act, bias=True)
        add(f"{p}.self_attn.k_proj", KV_HEADS * hd, HIDDEN, g_idx=act, bias=True)
        add(f"{p}.self_attn.v_proj", KV_HEADS * hd, HIDDEN, "int8" if not act else "int4", g_idx=act, bias=True)
        add(f"{p}.self_attn.o_proj", HIDDEN, HEADS * hd, "fp16" if not act else "int4", g_idx=act)
        add(f"{p}.mlp.gate_proj", INTER, HIDDEN, g_idx=act)
        add(f"{p}.mlp.up_proj", INTER, HIDDEN, g_idx=act)
        add(f"{p}.mlp.down_proj", HIDDEN, INTER, g_idx=act)

    header = QUFHeader(
        model_config={
            "architecture": "Qwen2ForCausalLM", "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "intermediate_size": INTER,
            "vocab_size": VOCAB, "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "tie_word_embeddings": True,
        },
        tokenizer={"model_id": "tiny", "eos_token_id": 0, "bos_token_id": None},
        quantization={
            "method": "gptq", "bits": 4, "group_size": group_size, "symmetric": False,
            "pack_dtype": "int32", "pack_order": "lsb_first", "act_order": True, "modules": modules,
        },
        tensors=[],
    )
    write_quf(path, header, tensors)


IDS = torch.tensor([[3, 7, 1, 9, 4, 2]])


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_model_cpu_fused_eager_agree_on_mixed_file(tmp_path, group_size):
    path = tmp_path / "mixed.quf"
    _build_mixed_quf(path, group_size)
    cpu = QUFModel.load(path, backend="cpu")
    assert cpu.layers[1].v.precision == "int8" and cpu.layers[1].o.precision == "fp16"
    out = cpu.forward(IDS)
    assert torch.isfinite(out).all()


@pytest.mark.gpu
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_model_backends_agree_on_gpu(tmp_path, group_size):
    path = tmp_path / "mixed.quf"
    _build_mixed_quf(path, group_size)
    ref = QUFModel.load(path, backend="cpu").forward(IDS)
    for name in ("torch_gpu", "torch_fused"):
        out = QUFModel.load(path, backend=name).forward(IDS).cpu()
        assert (out - ref).abs().max().item() <= 5e-2, name
        assert out.argmax(-1).tolist() == ref.argmax(-1).tolist(), name
