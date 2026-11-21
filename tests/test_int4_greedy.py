"""Property 37: triton_int4 greedy tokens equal torch_gpu (GPU + Triton).

Tiny models come from ``build_tiny_quf`` in test_model.py, with the shape-override pattern of
test_cuda_graph.py. The helper's ``quantize_groupwise`` and ``write_quf`` are wrapped so each
model also gets a chosen group size, act-order ``g_idx``, int8 / fp16 modules, a small
embedding scale and a smaller weight scale (well-conditioned logits, so near-ties are rare).
"""

from __future__ import annotations

import gc
import zlib

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import test_model as tm
from quf.quant.dequant import reference_dequantize
from quf.quant.quantize import QuantizedWeight, quantize_groupwise, quantize_int8
from quf.runtime.cuda_graph import make_decoder
from quf.runtime.model import QUFModel

pytestmark = [pytest.mark.gpu, pytest.mark.triton]

# (hidden, heads, kv_heads, inter, layers): tiny, MHA, and the GQA head ratios of
# Qwen2.5-0.5B (14:2) and Qwen2.5-1.5B (12:2) with small depth.
SHAPES = {
    "tiny": (128, 4, 2, 128, 2),
    "mha": (128, 2, 2, 128, 1),
    "qwen0.5b": (896, 14, 2, 256, 2),
    "qwen1.5b": (1536, 12, 2, 256, 2),
}
MIXES = ("int4", "int8", "fp16", "mixed")

WEIGHT_SCALE = 0.05  # helper uses 0.1; smaller keeps activations and logits well-conditioned
EMBED_SCALE = 0.002  # helper uses 1.0; tied logits become tiny, so fp16 kernel noise stays below NEAR_TIE_GAP
VOCAB = 16  # fewer tokens means wider top-2 gaps relative to kernel noise
LOGIT_ATOL = 2e-2
NEAR_TIE_GAP = 1e-3
MAX_NEAR_TIE_FRACTION = 0.05

_TOP_MODEL_CACHE: dict[tuple, tuple[QUFModel, QUFModel]] = {}
_STATS = {"examples": 0, "near_ties": 0}


def _build(path, shape: str, group_size: int, act_order: bool, mix: str, seed: int) -> None:
    hidden, heads, kv, inter, layers = SHAPES[shape]
    rng = np.random.default_rng(seed)
    real_quant, real_write = tm.quantize_groupwise, tm.write_quf

    def quant(w, *a, **k):
        return real_quant(w * (WEIGHT_SCALE / 0.1), group_size=group_size)

    def write(p, header, tensors):
        tensors = dict(tensors)
        tensors["model.embed_tokens.weight"] = (
            tensors["model.embed_tokens.weight"].astype(np.float32) * EMBED_SCALE
        ).astype(np.float16)
        modules: dict[str, str] = {}
        for qname in [n for n in tensors if n.endswith(".qweight")]:
            prefix = qname[: -len(".qweight")]
            prec = {"int4": "int4", "int8": "int8", "fp16": "fp16"}.get(mix) or str(
                rng.choice(["int4", "int8", "fp16"])
            )
            modules[prefix] = prec
            qweight = torch.from_numpy(tensors[qname])
            scales = torch.from_numpy(tensors[prefix + ".scales"])
            qzeros = torch.from_numpy(tensors[prefix + ".qzeros"])
            in_features = qweight.shape[0] * 8
            if prec == "int4":
                if act_order:
                    g = torch.from_numpy(rng.permutation(in_features)) // group_size
                    tensors[prefix + ".g_idx"] = g.to(torch.int32).numpy()
                continue
            dense = reference_dequantize(
                QuantizedWeight(qweight, scales, qzeros, group_size, in_features, qweight.shape[1])
            )
            for suffix in (".qweight", ".scales", ".qzeros"):
                del tensors[prefix + suffix]
            if prec == "int8":
                q, s = quantize_int8(dense.float())
                tensors[prefix + ".qweight_int8"] = q.numpy()
                tensors[prefix + ".scales"] = s.numpy()
            else:
                tensors[prefix + ".weight"] = dense.to(torch.float16).numpy()
        header.quantization["group_size"] = group_size
        header.quantization["act_order"] = bool(act_order)
        header.quantization["modules"] = modules
        return real_write(p, header, tensors)

    saved = (tm.HIDDEN, tm.HEADS, tm.KV_HEADS, tm.INTER, tm.LAYERS, tm.VOCAB)
    tm.HIDDEN, tm.HEADS, tm.KV_HEADS, tm.INTER, tm.LAYERS, tm.VOCAB = hidden, heads, kv, inter, layers, VOCAB
    tm.quantize_groupwise, tm.write_quf = quant, write
    try:
        tm.build_tiny_quf(path, seed=seed)
    finally:
        tm.HIDDEN, tm.HEADS, tm.KV_HEADS, tm.INTER, tm.LAYERS, tm.VOCAB = saved
        tm.quantize_groupwise, tm.write_quf = real_quant, real_write


def _models(root, key: tuple) -> tuple[QUFModel, QUFModel]:
    if key not in _TOP_MODEL_CACHE:
        while len(_TOP_MODEL_CACHE) >= 4:  # bound GPU memory
            _TOP_MODEL_CACHE.pop(next(iter(_TOP_MODEL_CACHE)))
            gc.collect()
            torch.cuda.empty_cache()
        shape, group_size, act_order, mix = key
        path = root / f"{shape}_{group_size}_{int(act_order)}_{mix}.quf"
        _build(path, shape, group_size, act_order, mix, seed=zlib.crc32(repr(key).encode()) % 1000)
        ref = QUFModel.load(path, "cuda", backend="torch_gpu")
        fast = QUFModel.load(path, "cuda", backend="triton_int4")
        _TOP_MODEL_CACHE[key] = (ref, fast)
    return _TOP_MODEL_CACHE[key]


def _greedy(model: QUFModel, prompt: list[int], new: int, cuda_graph: bool):
    """Greedy decode ``new`` tokens; returns (tokens, per-step fp32 logits on the CPU)."""
    cache = model.new_static_cache(len(prompt) + new)
    logits = model.forward(torch.tensor([prompt]), cache)[0, -1].float().cpu()
    tokens, all_logits = [int(logits.argmax())], [logits]
    dec = None
    if cuda_graph and new > 1:
        dec = make_decoder(model, cache, True)
        assert dec is not None, "CUDA graph capture failed"
    stepper = dec if dec is not None else model
    for _ in range(new - 1):
        logits = stepper.decode_token(tokens[-1], cache)[0, -1].float().cpu().clone()
        tokens.append(int(logits.argmax()))
        all_logits.append(logits)
    return tokens, all_logits


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    _STATS.update(examples=0, near_ties=0)
    yield tmp_path_factory.mktemp("int4_greedy")
    _TOP_MODEL_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


@settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    shape=st.sampled_from(sorted(SHAPES)),
    group_size=st.sampled_from([32, 64, 128]),
    act_order=st.booleans(),
    mix=st.sampled_from(MIXES),
    prompt=st.lists(st.integers(1, VOCAB - 1), min_size=1, max_size=24),
    new=st.integers(2, 24),
)
def test_int4_greedy_tokens_equal_torch_gpu(model_root, shape, group_size, act_order, mix, prompt, new):
    """Feature: quf-quantized-format, Property 37: Int4 backend greedy tokens equal Torch_GPU

    **Validates: Requirements 35.1**
    """
    ref, fast = _models(model_root, (shape, group_size, act_order, mix))
    _STATS["examples"] += 1
    near_tie = False
    for cuda_graph in (False, True):
        ref_tokens, ref_logits = _greedy(ref, prompt, new, cuda_graph)
        fast_tokens, fast_logits = _greedy(fast, prompt, new, cuda_graph)
        for i, (a, b) in enumerate(zip(ref_tokens, fast_tokens)):
            err = float((ref_logits[i] - fast_logits[i]).abs().max())
            assert err <= LOGIT_ATOL, f"step {i} logits differ by {err} (graph={cuda_graph})"
            if a != b:
                top2 = torch.topk(ref_logits[i], 2).values
                gap = float(top2[0] - top2[1])
                assert gap < NEAR_TIE_GAP, (
                    f"step {i}: torch_gpu token {a} != triton_int4 token {b}, "
                    f"top-2 gap {gap} (graph={cuda_graph})"
                )
                near_tie = True
                break  # prefixes now differ; later steps are not comparable
    _STATS["near_ties"] += int(near_tie)


def test_near_ties_are_rare(model_root):
    """Runs after the property test (definition order): near-tie runs must stay rare."""
    n = _STATS["examples"]
    if n == 0:
        pytest.skip("property test did not run")
    print(f"near-ties: {_STATS['near_ties']} of {n}")
    assert _STATS["near_ties"] <= MAX_NEAR_TIE_FRACTION * n, (
        f"{_STATS['near_ties']} of {n} examples were near-ties"
    )
