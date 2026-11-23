"""Quantized_Embedding on every Backend (task 34.3).

Feature: quf-quantized-format
Property 41 (backend equivalence): row lookups equal the reference row dequantization.
Property 39: the quantized lm_head GEMV equals ``x @ dequantize_embedding(...).T`` in fp32.
**Validates: Requirements 38.5, 38.6, 38.9, 38.4, 38.10**
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import test_model as tm
from quf.format.reader import QUFReader
from quf.format.writer import write_quf
from quf.quant.embedding import dequantize_embedding, dequantize_embedding_rows, quantize_embedding_rtn
from quf.runtime.cuda_graph import make_decoder
from quf.runtime.embedding import RuntimeEmbedding
from quf.runtime.model import QUFModel

ATOL, RTOL = 1e-2, 2e-3  # Property 39
EMBED_SCALE = 0.05  # keeps tied logits well-conditioned (few near-ties)
gpu_triton = [pytest.mark.gpu, pytest.mark.triton]


def build_v3(path, bits=4, gs=64, untied=False, seed=0, vocab=None) -> None:
    """Tiny model whose embedding (and, if ``untied``, lm_head) is a Quantized_Embedding."""
    real_write = tm.write_quf
    saved = tm.VOCAB
    if vocab:
        tm.VOCAB = vocab

    def write(p, header, tensors):
        tensors = dict(tensors)
        w = torch.from_numpy(tensors.pop("model.embed_tokens.weight")).float() * EMBED_SCALE
        suffix = ".embed_q4" if bits == 4 else ".embed_q8"
        spec = {"bits": bits, "group_size": gs, "method": "rtn"}
        qe = quantize_embedding_rtn(w, bits, gs)
        tensors["model.embed_tokens" + suffix] = qe.packed
        tensors["model.embed_tokens.embed_scales"] = qe.scales
        header.quantization["embedding"] = {"name": "model.embed_tokens", **spec}
        if untied:
            g = torch.Generator().manual_seed(seed + 1)
            ql = quantize_embedding_rtn(torch.randn(w.shape, generator=g) * EMBED_SCALE, bits, gs)
            tensors["lm_head" + suffix] = ql.packed
            tensors["lm_head.embed_scales"] = ql.scales
            header.quantization["lm_head"] = {"name": "lm_head", **spec}
            header.model_config["tie_word_embeddings"] = False
        return real_write(p, header, tensors)

    tm.write_quf = write
    try:
        tm.build_tiny_quf(path, seed=seed)
    finally:
        tm.write_quf, tm.VOCAB = real_write, saved


def _tables(path, key="model.embed_tokens"):
    with QUFReader(path) as r:
        bits = 4 if key + ".embed_q4" in r.tensor_names() else 8
        sfx = ".embed_q4" if bits == 4 else ".embed_q8"
        spec = r.header.embedding if key == "model.embed_tokens" else r.header.lm_head_embedding
        return np.array(r.tensor(key + sfx)), np.array(r.tensor(key + ".embed_scales")), bits, spec["group_size"]


def _ref_logits(x, packed, scales, bits, gs):
    w = dequantize_embedding_rows(packed, scales, bits, gs).float()
    return x.float() @ w.t()


# -- CPU: runs in CI --------------------------------------------------------------------------

@pytest.mark.parametrize("bits,gs", [(4, 32), (4, 64), (8, 32), (8, 64)])
def test_cpu_lookup_equals_reference(tmp_path, bits, gs):
    p = tmp_path / "m.quf"
    build_v3(p, bits, gs)
    packed, scales, b, g = _tables(p)
    model = QUFModel.load(p, "cpu", backend="cpu")
    assert model.embed is None and model.qembed is not None and model.qlm_head is model.qembed
    ids = torch.tensor([[0, tm.VOCAB - 1, 5, 5]])
    got = model.qembed.lookup(ids)
    ref = dequantize_embedding_rows(packed, scales, b, g, ids.reshape(-1).numpy()).reshape(1, 4, -1)
    assert torch.equal(got, ref)
    assert torch.equal(model._embed(ids), ref.to(model.dtype))


@pytest.mark.parametrize("bits,gs", [(4, 32), (8, 64)])
@pytest.mark.parametrize("m", [1, 4, 17])
def test_chunked_head_equals_reference(m, bits, gs):
    g = torch.Generator().manual_seed(m)
    w = torch.randn(300, 128, generator=g) * 0.1
    qe = quantize_embedding_rtn(w, bits, gs)
    emb = RuntimeEmbedding(qe.packed, qe.scales, bits, gs, torch.device("cpu"))
    x = torch.randn(m, 128, generator=g)
    out = emb.logits(x)
    ref = x @ dequantize_embedding(qe).float().t()
    assert out.dtype == torch.float32 and out.shape == (m, 300)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_chunking_is_bounded(monkeypatch):
    import quf.runtime.embedding as E

    monkeypatch.setattr(E, "CHUNK_ELEMENTS", 128 * 7)  # 7 rows per chunk
    qe = quantize_embedding_rtn(torch.randn(50, 128), 4, 64)
    emb = RuntimeEmbedding(qe.packed, qe.scales, 4, 64, torch.device("cpu"))
    x = torch.randn(2, 128)
    torch.testing.assert_close(emb.logits(x), x @ dequantize_embedding(qe).float().t(), atol=ATOL, rtol=RTOL)


def test_v3_model_matches_dense_embedding_model(tmp_path):
    """A v3 model equals a v1-style model whose fp16 table is the dequantized one."""
    p = tmp_path / "v3.quf"
    build_v3(p, 4, 64)
    packed, scales, b, g = _tables(p)
    table = dequantize_embedding_rows(packed, scales, b, g).numpy()
    real = tm.write_quf

    def write(path, header, tensors):
        tensors = dict(tensors)
        tensors["model.embed_tokens.weight"] = table
        return real(path, header, tensors)

    tm.write_quf = write
    try:
        tm.build_tiny_quf(tmp_path / "dense.quf")
    finally:
        tm.write_quf = real
    qm, dm = QUFModel.load(p, "cpu", backend="cpu"), QUFModel.load(tmp_path / "dense.quf", "cpu", backend="cpu")
    assert dm.qembed is None and dm.embed is not None  # v1 behaviour unchanged
    ids = torch.tensor([[3, 7, 1, 9, 4, 2]])
    torch.testing.assert_close(qm.forward(ids), dm.forward(ids), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bits", [4, 8])
def test_untied_quantized_lm_head(tmp_path, bits):
    p = tmp_path / "u.quf"
    build_v3(p, bits, 32, untied=True)
    model = QUFModel.load(p, "cpu", backend="cpu")
    assert model.qlm_head is not None and model.qlm_head is not model.qembed
    packed, scales, b, g = _tables(p, "lm_head")
    h = torch.randn(3, model.hidden_size)
    torch.testing.assert_close(model.qlm_head.logits(h), _ref_logits(h, packed, scales, b, g), atol=ATOL, rtol=RTOL)
    assert model.forward(torch.tensor([[1, 2, 3]])).shape == (1, 3, tm.VOCAB)


# -- GPU: Triton ------------------------------------------------------------------------------

def _gemv_case(rows, cols, bits, gs, m, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * 0.1  # embedding-like magnitudes
    qe = quantize_embedding_rtn(w, bits, gs)
    emb = RuntimeEmbedding(qe.packed, qe.scales, bits, gs, torch.device("cuda"), use_kernel=True)
    x = torch.randn(m, cols, generator=g).to(torch.float16).cuda()
    ref = x.float() @ dequantize_embedding(qe).float().cuda().t()
    return emb, x, ref


@pytest.mark.gpu
@pytest.mark.triton
@settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    rows=st.integers(1, 300),
    k_groups=st.integers(1, 12),
    bits=st.sampled_from([4, 8]),
    gs=st.sampled_from([32, 64]),
    m=st.sampled_from([1, 4, 17]),
    seed=st.integers(0, 1000),
)
def test_lm_head_gemv_equals_reference(rows, k_groups, bits, gs, m, seed):
    """Feature: quf-quantized-format, Property 39: Quantized lm_head GEMV equals the dequantized reference
    **Validates: Requirements 38.6**
    """
    emb, x, ref = _gemv_case(rows, k_groups * gs, bits, gs, m, seed)
    out = emb.logits(x)  # M <= 16 runs the Triton GEMV, M = 17 the chunked path
    assert out.dtype == torch.float32 and out.shape == (m, rows)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.parametrize("bits,gs", [(4, 32), (4, 64), (8, 32), (8, 64)])
def test_gemv_tall_vocab_and_argmax(bits, gs):
    emb, x, ref = _gemv_case(20000, 896 if gs == 64 else 960, bits, gs, 1, 7)
    out = emb.logits(x)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
    top2 = torch.topk(ref[0], 2).values
    if float(top2[0] - top2[1]) > ATOL + RTOL * float(top2[0].abs()):
        assert int(out.argmax()) == int(ref.argmax())


@pytest.mark.gpu
@pytest.mark.triton
def test_gemv_is_cuda_graph_capturable():
    emb, x, ref = _gemv_case(1000, 128, 4, 64, 1, 3)
    emb.logits(x)  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = emb.logits(x)
    x.copy_(x * 0.5)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, (x.float() @ dequantize_embedding_rows(emb.packed, emb.scales, 4, 64).float().t()), atol=ATOL, rtol=RTOL)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.parametrize("backend", ["cpu", "torch_gpu", "torch_fused", "triton_int4"])
@pytest.mark.parametrize("bits,gs", [(4, 32), (8, 64)])
def test_row_lookup_equals_reference_on_every_backend(tmp_path, backend, bits, gs):
    """Feature: quf-quantized-format, Property 41: row gather equals the reference row dequant on every Backend
    **Validates: Requirements 38.5, 38.9**
    """
    p = tmp_path / "m.quf"
    build_v3(p, bits, gs)
    packed, scales, b, g = _tables(p)
    model = QUFModel.load(p, "cpu" if backend == "cpu" else "cuda", backend=backend)
    ids = torch.tensor([[0, tm.VOCAB - 1, 17, 17]])
    ref = dequantize_embedding_rows(packed, scales, b, g, ids.reshape(-1).numpy()).reshape(1, 4, -1)
    got = model.qembed.lookup(ids.to(model.device))
    assert torch.equal(got.cpu(), ref)
    # no full fp16 table is kept on any Backend
    assert model.embed is None and model.qembed.packed.dtype == torch.int8
    assert model.qlm_head is model.qembed


@pytest.mark.gpu
@pytest.mark.triton
def test_triton_backend_uses_gemv_kernel(tmp_path):
    p = tmp_path / "m.quf"
    build_v3(p, 4, 64)
    assert QUFModel.load(p, "cuda", backend="triton_int4").qembed.use_kernel
    assert not QUFModel.load(p, "cuda", backend="torch_gpu").qembed.use_kernel


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.parametrize("backend", ["torch_gpu", "torch_fused"])
@pytest.mark.parametrize("bits", [4, 8])
def test_gpu_backends_logits_match_cpu(tmp_path, backend, bits):
    p = tmp_path / "m.quf"
    build_v3(p, bits, 64, untied=True)
    cpu = QUFModel.load(p, "cpu", backend="cpu")
    gpu = QUFModel.load(p, "cuda", backend=backend)
    ids = torch.tensor([[3, 7, 1, 9, 4, 2]])
    torch.testing.assert_close(gpu.forward(ids).cpu(), cpu.forward(ids), atol=5e-2, rtol=5e-2)


NEAR_TIE_GAP = 5e-3  # fp16 kernel noise reaches about 1.5e-3 on the top-2 gap (hypothesis found 1.45e-3)
_GREEDY: dict[tuple, tuple[QUFModel, QUFModel]] = {}


def _greedy(model, prompt, new, graph):
    cache = model.new_static_cache(len(prompt) + new)
    logits = model.forward(torch.tensor([prompt]), cache)[0, -1].float().cpu()
    toks, allg = [int(logits.argmax())], [logits]
    stepper = model
    if graph and new > 1:
        stepper = make_decoder(model, cache, True)
        assert stepper is not None, "CUDA graph capture failed"
    for _ in range(new - 1):
        logits = stepper.decode_token(toks[-1], cache)[0, -1].float().cpu().clone()
        toks.append(int(logits.argmax()))
        allg.append(logits)
    return toks, allg


@pytest.mark.gpu
@pytest.mark.triton
@settings(max_examples=12, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    bits=st.sampled_from([4, 8]),
    untied=st.booleans(),
    prompt=st.lists(st.integers(1, 15), min_size=1, max_size=12),
    new=st.integers(2, 12),
)
def test_greedy_tokens_cpu_vs_triton_int4(tmp_path_factory, bits, untied, prompt, new):
    """Greedy tokens of a tiny quantized-embedding model match between cpu and triton_int4
    (graph off and on), except where the cpu top-2 logits are a near-tie.
    **Validates: Requirements 38.5, 38.6, 38.9**
    """
    key = (bits, untied)
    if key not in _GREEDY:
        p = tmp_path_factory.mktemp("greedy") / "m.quf"
        build_v3(p, bits, 64, untied=untied, vocab=16)
        _GREEDY[key] = (QUFModel.load(p, "cpu", backend="cpu"), QUFModel.load(p, "cuda", backend="triton_int4"))
    cpu, fast = _GREEDY[key]
    for graph in (False, True):
        ref_t, ref_l = _greedy(cpu, prompt, new, False)
        got_t, got_l = _greedy(fast, prompt, new, graph)
        for i, (a, b) in enumerate(zip(ref_t, got_t)):
            assert float((ref_l[i] - got_l[i]).abs().max()) <= 5e-2, f"step {i} logits differ (graph={graph})"
            if a != b:
                top2 = torch.topk(ref_l[i], 2).values
                assert float(top2[0] - top2[1]) < NEAR_TIE_GAP, f"step {i}: {a} != {b}, not a near-tie"
                break
