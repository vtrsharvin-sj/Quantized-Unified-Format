"""Unit tests for the Qwen2 decoder and KV_Cache (CPU)."""

from __future__ import annotations

import numpy as np
import torch

from quf.format.schema import QUFHeader
from quf.format.writer import write_quf
from quf.quant.quantize import quantize_groupwise
from quf.runtime.model import KVCache, QUFModel

HIDDEN, HEADS, KV_HEADS, INTER, LAYERS, VOCAB = 128, 4, 2, 128, 2, 32


def build_tiny_quf(path, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    head_dim = HIDDEN // HEADS
    tensors: dict[str, np.ndarray] = {}

    def fp16(*shape, scale=0.1):
        return (torch.randn(*shape, generator=g) * scale).to(torch.float16).numpy()

    def add_linear(name, out, inp, bias=False):
        qw = quantize_groupwise(torch.randn(out, inp, generator=g) * 0.1)
        tensors[f"{name}.qweight"] = qw.qweight.numpy()
        tensors[f"{name}.scales"] = qw.scales.numpy()
        tensors[f"{name}.qzeros"] = qw.qzeros.numpy()
        if bias:
            tensors[f"{name}.bias"] = fp16(out)

    tensors["model.embed_tokens.weight"] = fp16(VOCAB, HIDDEN, scale=1.0)
    tensors["model.norm.weight"] = np.ones(HIDDEN, dtype=np.float16)
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(HIDDEN, dtype=np.float16)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(HIDDEN, dtype=np.float16)
        add_linear(f"{p}.self_attn.q_proj", HEADS * head_dim, HIDDEN, bias=True)
        add_linear(f"{p}.self_attn.k_proj", KV_HEADS * head_dim, HIDDEN, bias=True)
        add_linear(f"{p}.self_attn.v_proj", KV_HEADS * head_dim, HIDDEN, bias=True)
        add_linear(f"{p}.self_attn.o_proj", HIDDEN, HEADS * head_dim)
        add_linear(f"{p}.mlp.gate_proj", INTER, HIDDEN)
        add_linear(f"{p}.mlp.up_proj", INTER, HIDDEN)
        add_linear(f"{p}.mlp.down_proj", HIDDEN, INTER)

    header = QUFHeader(
        model_config={
            "architecture": "Qwen2ForCausalLM",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "intermediate_size": INTER,
            "vocab_size": VOCAB,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
        },
        tokenizer={"model_id": "tiny", "eos_token_id": 0, "bos_token_id": None},
        quantization={
            "method": "gptq", "bits": 4, "group_size": 128, "symmetric": False,
            "pack_dtype": "int32", "pack_order": "lsb_first",
        },
        tensors=[],
    )
    write_quf(path, header, tensors)


def test_forward_shape_and_cache_equivalence(tmp_path):
    path = tmp_path / "tiny.quf"
    build_tiny_quf(path)
    model = QUFModel.load(path, "cpu")
    ids = torch.tensor([[3, 7, 1, 9, 4, 2]])

    full = model.forward(ids)
    assert full.shape == (1, 6, VOCAB)

    cache = KVCache(model.num_layers)
    first = model.forward(ids[:, :4], cache)
    assert cache.length == 4
    steps = [model.forward(ids[:, i : i + 1], cache) for i in (4, 5)]
    assert cache.length == 6
    inc = torch.cat([first] + steps, dim=1)
    assert torch.allclose(inc, full, atol=1e-2)
