"""Task 33.3: gate, CUDA-graph capture, and fallback of the Fused_Decode_Kernels.

The fallback is all-or-nothing for the decode step: when any fused kernel raises on its first
call (compile) or at capture, the whole model returns to the 0.3.0 path, logs one INFO line, and
keeps producing the same greedy tokens as a model that never used the fused path. The quantized
lm_head GEMV is the one exception: it falls back by itself to the chunked dequantize path.

**Validates: Requirements 37.6, 37.7, 37.8**
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from quf.runtime.generate import GREEDY_CHECK_EVERY, generate
from quf.runtime.kernels.triton_kernels import triton_available
from quf.runtime.model import QUFModel

HAS_TRITON = torch.cuda.is_available() and triton_available()[0]
needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs CUDA and Triton")
gpu_triton = [pytest.mark.gpu, pytest.mark.triton, needs_triton]

FUSED_MODULES = ("fused", "decode_gemv", "attention", "embed_gemv")
PROMPT = [3, 5, 7]


class FakeTok:
    def encode(self, text):
        return [ord(c) % 32 for c in text]

    def decode(self, ids):
        return "".join(chr(97 + int(i) % 26) for i in ids)


def _build(path, untied=True):
    from test_embed_runtime import build_v3

    build_v3(path, 4, 32, untied=untied, seed=3)


@pytest.fixture(scope="module")
def quf_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("fallback") / "m.quf"
    _build(p)
    return p


def _load(path) -> QUFModel:
    model = QUFModel.load(path, "cuda", backend="triton_int4")
    model.header.tokenizer["eos_token_id"] = None
    return model


def _reference_tokens(path, n=20, graph=True, prompt=PROMPT):
    """Greedy tokens of the 0.3.0 decode path (fused off, chunked quantized head)."""
    ref = _load(path)
    ref.fused_decode = False
    ref._disable_embed_kernels()
    return generate(ref, FakeTok(), prompt, max_new_tokens=n, cuda_graph=graph)


def _info_lines(caplog, *needles):
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.INFO and any(n in r.getMessage() for n in needles)
    ]


# -- gate (no GPU needed) -------------------------------------------------------------------

def test_gate_honours_no_fused_and_no_triton(monkeypatch):
    monkeypatch.setenv("QUF_NO_FUSED", "1")
    monkeypatch.delenv("QUF_NO_TRITON", raising=False)
    ok, why = QUFModel._fused_gate()
    assert not ok and "QUF_NO_FUSED" in why
    monkeypatch.delenv("QUF_NO_FUSED")
    monkeypatch.setenv("QUF_NO_TRITON", "1")
    ok, why = QUFModel._fused_gate()
    assert not ok and "QUF_NO_TRITON" in why


def _run(script, **env_extra):
    env = dict(os.environ)
    env.pop("QUF_NO_TRITON", None)
    env.pop("QUF_NO_FUSED", None)
    env.update(env_extra)
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "OK" in proc.stdout, proc.stdout


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
def test_no_fused_env_never_imports_fused_modules(quf_path):
    _run(
        f"""
        import sys
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        from quf.runtime.model import QUFModel
        from quf.runtime.generate import generate
        m = QUFModel.load({str(quf_path)!r}, "cuda", backend="triton_int4")
        assert not m.fused_decode and not m.qembed.use_kernel and not m.qlm_head.use_kernel
        out = generate(m, None, [3, 5, 7], max_new_tokens=6)
        assert len(out) >= 1
        for name in {FUSED_MODULES!r}:
            assert "quf.runtime.kernels." + name not in sys.modules, name
        print("OK")
        """,
        QUF_NO_FUSED="1",
    )


# -- fallback: a kernel raises on its first call ---------------------------------------------

def _boom(*a, **k):
    raise RuntimeError("forced compile failure")


def _boom_on_capture(real):
    def wrapper(*a, **k):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("forced capture failure")
        return real(*a, **k)

    return wrapper


def _patch(monkeypatch, model, which, wrap):
    """Replace one kernel entry point of ``model`` with ``wrap(real)``."""
    from quf.runtime.kernels import attention, decode_gemv, embed_gemv

    if which in ("add_rmsnorm", "silu_mul_scatter", "rope_kv_write", "flash_decode"):
        attr = {"add_rmsnorm": "_add_rmsnorm", "silu_mul_scatter": "_silu_mul_scatter",
                "rope_kv_write": "_rope_kv_write", "flash_decode": "_flash_decode"}[which]
        monkeypatch.setattr(model, attr, wrap(getattr(model, attr)))
    elif which == "gemv_jmajor":
        monkeypatch.setattr(decode_gemv, "gemv_jmajor", wrap(decode_gemv.gemv_jmajor))
    elif which == "scatter_to":
        model._fast_attention = False  # the torch attention path is the one that uses scatter_to
        monkeypatch.setattr(decode_gemv, "scatter_to", wrap(decode_gemv.scatter_to))
    elif which == "embed_gemv":
        monkeypatch.setattr(embed_gemv, "embed_gemv", wrap(embed_gemv.embed_gemv))
    else:  # pragma: no cover
        raise AssertionError(which)


KERNELS = ["add_rmsnorm", "gemv_jmajor", "silu_mul_scatter", "rope_kv_write", "flash_decode",
           "scatter_to", "embed_gemv"]


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
@pytest.mark.parametrize("which", KERNELS)
def test_compile_failure_falls_back_whole_model(quf_path, monkeypatch, caplog, which, graph):
    expected = _reference_tokens(quf_path, graph=graph)
    model = _load(quf_path)
    assert model.fused_decode
    _patch(monkeypatch, model, which, lambda real: _boom)
    with caplog.at_level(logging.INFO, logger="quf.runtime"):
        out = generate(model, FakeTok(), PROMPT, max_new_tokens=20, cuda_graph=graph)
        again = generate(model, FakeTok(), PROMPT, max_new_tokens=20, cuda_graph=graph)
    lines = _info_lines(caplog, "0.3.0 path", "chunked dequantize")
    assert len(lines) == 1, lines
    assert "forced compile failure" in lines[0]
    if which == "embed_gemv":
        assert model.fused_decode and not model.qlm_head.use_kernel
    else:
        assert not model.fused_decode
    assert out == again
    if which == "embed_gemv":
        # fused layers with the chunked head: the same arithmetic as running the model with the
        # lm_head kernel off from the start
        ref = _load(quf_path)
        ref._disable_embed_kernels()
        assert out == generate(ref, FakeTok(), PROMPT, max_new_tokens=20, cuda_graph=graph)
    else:
        assert out == expected


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("which", [k for k in KERNELS if k != "embed_gemv"])
def test_capture_failure_falls_back_without_partial_graph(quf_path, monkeypatch, caplog, which):
    """The kernel compiles and runs eagerly but raises while the graph is captured."""
    expected = _reference_tokens(quf_path)
    model = _load(quf_path)
    _patch(monkeypatch, model, which, _boom_on_capture)
    with caplog.at_level(logging.INFO, logger="quf.runtime"):
        out = generate(model, FakeTok(), PROMPT, max_new_tokens=20, cuda_graph=True)
    lines = _info_lines(caplog, "0.3.0 path", "CUDA graph capture failed")
    assert len(lines) == 1, lines
    assert "capture failed" in lines[0]
    assert not model.fused_decode
    assert out == expected  # captured from the 0.3.0 path, not eager


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("module", ["fused", "decode_gemv", "attention"])
def test_import_failure_keeps_0_3_0_path(quf_path, monkeypatch, caplog, module):
    for name in ("fused", "decode_gemv", "attention"):
        monkeypatch.delitem(sys.modules, f"quf.runtime.kernels.{name}", raising=False)
    monkeypatch.setitem(sys.modules, f"quf.runtime.kernels.{module}", None)
    with caplog.at_level(logging.INFO, logger="quf.runtime"):
        model = _load(quf_path)
        out = generate(model, FakeTok(), PROMPT, max_new_tokens=20)
    assert not model.fused_decode
    assert len(_info_lines(caplog, "0.3.0 path")) == 1
    monkeypatch.undo()
    assert out == _reference_tokens(quf_path)


# -- edge cases on the real fused path ------------------------------------------------------

@pytest.fixture(scope="module")
def fused_model(quf_path):
    model = _load(quf_path)
    assert model.fused_decode and model._fast_attention
    return model


def _gen(model, prompt, n, **kw):
    return generate(model, FakeTok(), prompt, max_new_tokens=n, **kw)


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
def test_prompt_of_one_and_single_new_token(fused_model):
    full = _gen(fused_model, [9], 12)
    assert len(full) == 12
    assert _gen(fused_model, [9], 1) == full[:1]
    assert _gen(fused_model, [9], 2) == full[:2]
    assert _gen(fused_model, [9], 12, cuda_graph=False) == full


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
def test_first_and_last_cache_positions(fused_model):
    """decode_step at position 0 and at position max_len - 1, against the 0.3.0 path."""
    m = fused_model
    for max_len, fill in ((8, 0), (8, 7)):
        logits = []
        for fused in (True, False):
            m.fused_decode = fused
            try:
                cache = m.new_static_cache(max_len)
                if fill:
                    m.forward(torch.tensor([[1, 2, 3, 4, 5, 6, 7][:fill]]), cache)
                logits.append(m.decode_token(11, cache)[0, -1].float().clone())
                assert cache.length == fill + 1
            finally:
                m.fused_decode = True
        assert torch.allclose(logits[0], logits[1], atol=2e-2), float((logits[0] - logits[1]).abs().max())
    # graph decode that ends exactly on the last slot (prompt 3 + 5 new = cache of 8)
    assert _gen(m, PROMPT, 5) == _gen(m, PROMPT, 5, cuda_graph=False)


def _first_index_token(ref, k):
    """A token whose first occurrence in ``ref`` is index k, or None."""
    return ref[k] if ref[k] not in ref[:k] else None


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("k", [GREEDY_CHECK_EVERY, GREEDY_CHECK_EVERY + 1, 2 * GREEDY_CHECK_EVERY])
@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_eos_at_check_boundary_and_one_past(fused_model, k, graph):
    ref = tok = None
    for first in range(1, 32):
        ref = _gen(fused_model, [first, 4], 30, cuda_graph=graph)
        if _first_index_token(ref, k) is not None:
            tok = ref[k]
            prompt = [first, 4]
            break
    if tok is None:
        pytest.skip("no prompt whose k-th token is new")
    fused_model.header.tokenizer["eos_token_id"] = tok
    try:
        for stream_on in (False, True):
            streamed: list[str] = []
            out = _gen(fused_model, prompt, 30, cuda_graph=graph,
                       stream=streamed.append if stream_on else None)
            assert out == ref[: k + 1]  # EOS included, nothing after it
            if stream_on:
                assert "".join(streamed) == FakeTok().decode(ref[:k])  # EOS is not streamed
    finally:
        fused_model.header.tokenizer["eos_token_id"] = None


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
def test_streaming_matches_non_streaming(fused_model):
    for graph in (True, False):
        plain = _gen(fused_model, PROMPT, 20, cuda_graph=graph)
        streamed: list[str] = []
        assert _gen(fused_model, PROMPT, 20, cuda_graph=graph, stream=streamed.append) == plain
        assert "".join(streamed) == FakeTok().decode(plain)


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
def test_sampling_mixed_with_greedy_graph(fused_model):
    greedy = _gen(fused_model, PROMPT, 20)
    s1 = _gen(fused_model, PROMPT, 20, temperature=0.9, seed=5, top_k=8)
    assert _gen(fused_model, PROMPT, 20, temperature=0.9, seed=5, top_k=8) == s1
    assert _gen(fused_model, PROMPT, 20, temperature=0.9, seed=5, top_k=8, cuda_graph=False) == s1
    assert _gen(fused_model, PROMPT, 20) == greedy  # a sampled run leaves no state behind
    assert all(0 <= t < 32 for t in s1)


# -- capture and replay of every kernel -----------------------------------------------------

REPLAYS = 6


def _check_graph(fn, inputs, mutate):
    """Warm up (compile), capture ``fn()``, replay with new inputs: outputs equal eager, and bare
    replays allocate nothing."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = fn()
    static = static if isinstance(static, tuple) else (static,)
    torch.cuda.synchronize()
    for i in range(REPLAYS):
        mutate(i)
        graph.replay()
        eager = fn()
        eager = eager if isinstance(eager, tuple) else (eager,)
        for g, e in zip(static, eager):
            torch.testing.assert_close(g, e, atol=0, rtol=0)
    graph.replay()
    a = [o.clone() for o in static]
    graph.replay()
    for x, y in zip(a, static):
        assert torch.equal(x, y)  # stable across replays with the same inputs
    del a, eager
    torch.cuda.synchronize()
    mem, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for _ in range(REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == mem and torch.cuda.memory_reserved() == reserved


def _layout(n=64, k=128, act_order=True, gs=32):
    from quf.quant.quantize import quantize_groupwise
    from quf.runtime.kernels import prepare_for_kernel

    qw = quantize_groupwise(torch.randn(n, k), group_size=gs, symmetric=False)
    if act_order:
        import dataclasses

        g_idx = (torch.arange(k) // gs)[torch.randperm(k)].to(torch.int32)
        qw = dataclasses.replace(qw, g_idx=g_idx)
    return prepare_for_kernel(qw, device="cuda")


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("with_res", [True, False])
@pytest.mark.parametrize("with_index", [True, False])
def test_add_rmsnorm_graph(with_res, with_index):
    from quf.runtime.kernels.decode_gemv import activation_index
    from quf.runtime.kernels.fused import add_rmsnorm

    x = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    res = torch.randn_like(x)
    w = torch.rand(128, device="cuda", dtype=torch.float16) + 0.5
    idx = activation_index(_layout()) if with_index else None
    _check_graph(lambda: add_rmsnorm(x, res if with_res else None, w, 1e-6, out_index=idx), [x],
                 lambda i: (x.copy_(torch.randn_like(x)), res.copy_(torch.randn_like(res))))


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("act_order", [True, False])
def test_gemv_and_scatter_kernels_graph(act_order):
    from quf.runtime.kernels.decode_gemv import (
        FastLinear, scatter_to, silu_mul_scatter, to_jmajor,
    )

    layout = _layout(act_order=act_order)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)
    fl = FastLinear(layout, bias)
    xj = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    gu = torch.randn(1, 256, device="cuda", dtype=torch.float16)
    idx = fl.index
    mutate = lambda i: (xj.copy_(torch.randn_like(xj)), gu.copy_(torch.randn_like(gu)))  # noqa: E731
    _check_graph(lambda: fl(xj), [xj], mutate)
    _check_graph(lambda: silu_mul_scatter(gu, idx), [gu], mutate)
    _check_graph(lambda: scatter_to(xj, idx), [xj], mutate)
    torch.testing.assert_close(scatter_to(xj, idx), to_jmajor(xj, layout), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("t", [64, 1100], ids=["one-split", "three-splits"])
def test_rope_kv_write_and_flash_decode_graph(t):
    """Positions 0, a chunk boundary, and the last slot, read from the device position tensor."""
    from quf.runtime.kernels.attention import flash_decode, rope_kv_write

    h, kv, d = 4, 2, 32
    qkv = torch.randn(1, (h + 2 * kv) * d, device="cuda", dtype=torch.float16)
    cos = torch.randn(t, d, device="cuda", dtype=torch.float16)
    sin = torch.randn(t, d, device="cuda", dtype=torch.float16)
    kc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16)
    vc = torch.randn(1, kv, t, d, device="cuda", dtype=torch.float16)
    pos = torch.zeros(1, dtype=torch.long, device="cuda")
    index = torch.randperm(h * d, device="cuda").to(torch.int32)
    positions = [0, 1, 511, 512, t - 2, t - 1] if t > 512 else [0, 1, 31, 32, t - 2, t - 1]

    def fn():
        q = rope_kv_write(qkv, cos, sin, pos, kc, vc, h, kv)
        return (q, flash_decode(q, kc, vc, pos, h, kv, out_index=index), flash_decode(q, kc, vc, pos, h, kv))

    def mutate(i):
        pos.fill_(positions[i % len(positions)])
        qkv.copy_(torch.randn_like(qkv))

    _check_graph(fn, [qkv], mutate)
    # the slot written is the one the position tensor names, and no other slot changes
    for p in positions:
        before = kc.clone()
        pos.fill_(p)
        rope_kv_write(qkv, cos, sin, pos, kc, vc, h, kv)
        changed = (kc != before).any(dim=-1).any(dim=1)[0].nonzero().flatten().tolist()
        assert changed in ([], [p]), (p, changed)
        assert torch.equal(vc[0, :, p], qkv.view(h + 2 * kv, d)[h + kv :])


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("bits", [4, 8])
def test_embed_gemv_graph(bits):
    from quf.quant.embedding import quantize_embedding_rtn
    from quf.runtime.kernels.embed_gemv import embed_gemv

    qe = quantize_embedding_rtn(torch.randn(300, 128) * 0.05, bits, 32)
    packed, scales = torch.from_numpy(qe.packed.copy()).cuda(), torch.from_numpy(qe.scales.copy()).cuda()
    x = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    _check_graph(lambda: embed_gemv(x, packed, scales, bits, 32), [x], lambda i: x.copy_(torch.randn_like(x)))


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
def test_full_decode_graph_replay_allocates_nothing(fused_model):
    from quf.runtime.cuda_graph import make_decoder

    cache = fused_model.new_static_cache(16)
    fused_model.forward(torch.tensor([PROMPT]), cache)
    dec = make_decoder(fused_model, cache, True)
    assert dec is not None and dec.supports_greedy_device
    dec.begin_greedy(5)
    dec.step_greedy()
    torch.cuda.synchronize()
    mem, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for _ in range(6):
        dec.step_greedy()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == mem and torch.cuda.memory_reserved() == reserved
    assert len(dec.drain_greedy()) == 7


# -- fallback: an int8 kernel raises (task 39.2) -----------------------------------------------

@pytest.fixture(scope="module")
def int8_quf_paths(tmp_path_factory):
    from test_int4_greedy import _build as build_mix

    root = tmp_path_factory.mktemp("fallback_int8")
    paths = {}
    for mix in ("int8", "mixed"):
        paths[mix] = root / f"{mix}.quf"
        build_mix(paths[mix], "tiny", 32, False, mix, seed=7)
    return paths


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
@pytest.mark.parametrize("mix", ["int8", "mixed"])
def test_int8_kernel_failure_falls_back_whole_model(int8_quf_paths, monkeypatch, caplog, mix, graph):
    """Every layer of an int8 or mixed file is on the fast path; when gemv_int8 raises (compile
    failure), the whole model returns to the 0.3.0 path and keeps the reference tokens."""
    import importlib

    # sys.modules, not the package attribute: test_import_failure_keeps_0_3_0_path leaves the two apart
    decode_gemv = importlib.import_module("quf.runtime.kernels.decode_gemv")

    path = int8_quf_paths[mix]
    expected = _reference_tokens(path, graph=graph)
    model = _load(path)
    assert model.fused_decode and all(f is not None for f in model._fast_layers)
    monkeypatch.setattr(decode_gemv, "gemv_int8", _boom)
    with caplog.at_level(logging.INFO, logger="quf.runtime"):
        out = generate(model, FakeTok(), PROMPT, max_new_tokens=20, cuda_graph=graph)
    lines = _info_lines(caplog, "0.3.0 path", "CUDA graph capture failed")
    assert len(lines) == 1 and "forced compile failure" in lines[0], lines
    assert not model.fused_decode
    assert out == expected


@pytest.mark.gpu
@pytest.mark.triton
@needs_triton
def test_unsupported_group_logs_reason_once_at_debug(tmp_path, caplog):
    """A layer the fast path cannot take is skipped, with one DEBUG line per reason."""
    import importlib

    decode_gemv = importlib.import_module("quf.runtime.kernels.decode_gemv")
    model = _load(_make_int8_path(tmp_path))
    real = decode_gemv.fast_group_of
    def refuse(module):
        return None, "forced test reason"

    decode_gemv.fast_group_of = refuse
    try:
        with caplog.at_level(logging.DEBUG, logger="quf.runtime.model"):
            model._build_fast_layers()
    finally:
        decode_gemv.fast_group_of = real
    assert all(f is None for f in model._fast_layers)
    lines = [r.getMessage() for r in caplog.records if "forced test reason" in r.getMessage()]
    assert len(lines) == 1, lines
    out = generate(model, FakeTok(), PROMPT, max_new_tokens=6)
    assert len(out) == 6


def _make_int8_path(tmp_path):
    from test_int4_greedy import _build as build_mix

    p = tmp_path / "m.quf"
    build_mix(p, "tiny", 32, False, "mixed", seed=7)
    return p
