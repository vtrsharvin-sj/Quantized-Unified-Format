"""Unit tests for the decode path: Static_KV_Cache, CUDA graph fallback and opt-outs.

Requirement 31. CPU tests run everywhere; tests marked ``gpu`` need an NVIDIA GPU.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from test_model import build_tiny_quf

from quf.runtime import cuda_graph as cg
from quf.runtime.generate import generate
from quf.runtime.model import CacheLengthError, QUFModel

PROMPT = [3, 7, 1, 9]


def _no_eos(model: QUFModel) -> QUFModel:
    """Stop the tiny random model from ending early on its eos token."""
    model.header.tokenizer["eos_token_id"] = None
    return model


@pytest.fixture(scope="module")
def quf_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("decode") / "tiny.quf"
    build_tiny_quf(path)
    return path


@pytest.fixture()
def cpu_model(quf_path):
    return _no_eos(QUFModel.load(quf_path, "cpu"))


@pytest.fixture()
def gpu_model(quf_path):
    return _no_eos(QUFModel.load(quf_path, "cuda", backend="torch_gpu"))


@pytest.fixture()
def capture_calls(monkeypatch):
    """Count CUDAGraphDecoder.capture attempts while still running the real capture."""
    calls = []
    real = cg.CUDAGraphDecoder.capture

    def spy(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(cg.CUDAGraphDecoder, "capture", spy)
    return calls


# --- Static_KV_Cache -------------------------------------------------------------------


def test_static_cache_overflow_message_names_needed_and_cache_length(cpu_model):
    cache = cpu_model.new_static_cache(4)
    cpu_model.forward(torch.tensor([[1, 2, 3, 4]]), cache)
    with pytest.raises(CacheLengthError) as ei:
        cpu_model.decode_token(5, cache)
    assert str(ei.value) == "sequence length (5) exceeds the static KV cache length (4)"
    assert (ei.value.needed, ei.value.cache_len) == (5, 4)
    assert cache.length == 4  # a failed step leaves the cache unchanged


def test_static_cache_overflow_on_prefill(cpu_model):
    cache = cpu_model.new_static_cache(3)
    with pytest.raises(CacheLengthError, match=r"\(4\).*\(3\)"):
        cpu_model.forward(torch.tensor([[1, 2, 3, 4]]), cache)


def test_static_cache_false_opt_out_matches_static(cpu_model):
    for prompt in (PROMPT, [5]):
        legacy = generate(cpu_model, None, prompt, max_new_tokens=10, static_cache=False)
        static = generate(cpu_model, None, prompt, max_new_tokens=10)
        assert len(legacy) == 10
        assert legacy == static


def test_static_cache_false_never_builds_a_static_cache(cpu_model, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("static cache must not be allocated")

    monkeypatch.setattr(QUFModel, "new_static_cache", boom)
    assert len(generate(cpu_model, None, PROMPT, max_new_tokens=5, static_cache=False)) == 5


# --- CPU never captures ----------------------------------------------------------------


def test_cpu_backend_never_attempts_capture(cpu_model, monkeypatch):
    def boom(self):
        raise AssertionError("capture attempted on CPU")

    monkeypatch.setattr(cg.CUDAGraphDecoder, "capture", boom)
    cache = cpu_model.new_static_cache(8)
    assert cg.make_decoder(cpu_model, cache) is None
    assert cg.make_decoder(cpu_model, cache, enabled=True) is None
    out = generate(cpu_model, None, PROMPT, max_new_tokens=6, cuda_graph=True)
    assert len(out) == 6


def test_cpu_forced_graph_logs_nothing(cpu_model, caplog):
    cache = cpu_model.new_static_cache(8)
    with caplog.at_level(logging.DEBUG, logger="quf.runtime"):
        assert cg.make_decoder(cpu_model, cache, enabled=True) is None
    assert caplog.records == []


# --- Opt-outs: env var and CLI flag ----------------------------------------------------


@pytest.mark.parametrize(
    "value,disabled",
    [("1", True), ("true", True), ("yes", True), ("0", False), ("", False), ("false", False)],
)
def test_env_var_parsing(monkeypatch, value, disabled):
    monkeypatch.setenv(cg.ENV_DISABLE, value)
    assert cg.cuda_graph_disabled_by_env() is disabled


def test_env_var_unset_means_enabled(monkeypatch):
    monkeypatch.delenv(cg.ENV_DISABLE, raising=False)
    assert cg.cuda_graph_disabled_by_env() is False


def test_cli_flag_parses_to_cuda_graph_false():
    from quf.cli import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["run", "m.quf", "--prompt", "hi"]).cuda_graph is True
    assert parser.parse_args(["run", "m.quf", "--prompt", "hi", "--no-cuda-graph"]).cuda_graph is False


@pytest.mark.parametrize("flag,expected", [([], None), (["--no-cuda-graph"], False)])
def test_cli_run_passes_cuda_graph_to_generate(monkeypatch, tmp_path, capsys, flag, expected):
    """`quf run` forwards None by default and False with --no-cuda-graph."""
    import quf.runtime.generate as gen_mod
    import quf.tokenization.wrapper as wrap_mod
    import tokenizers
    from quf import cli

    seen = {}
    stub_model = SimpleNamespace(header=SimpleNamespace(tokenizer={"files": {"tokenizer.json": "{}"}}))

    class _Tok:
        @staticmethod
        def from_str(_s):
            return object()

    class _Wrapper:
        def __init__(self, hf_tokenizer=None, **_):
            pass

        def decode(self, ids):
            return "ok"

    def fake_generate(model, tok, prompt, **kw):
        seen.update(kw)
        return [1]

    monkeypatch.setattr(QUFModel, "load", classmethod(lambda cls, *a, **k: stub_model))
    monkeypatch.setattr(tokenizers, "Tokenizer", _Tok)
    monkeypatch.setattr(wrap_mod, "TokenizerWrapper", _Wrapper)
    monkeypatch.setattr(gen_mod, "generate", fake_generate)
    f = tmp_path / "m.quf"
    f.write_bytes(b"")
    rc = cli.main(["run", str(f), "--prompt", "hi", "--device", "cpu", "--no-stream", *flag])
    assert rc == 0
    assert seen["cuda_graph"] is expected


# --- GPU: graph capture, fallback, env var, determinism --------------------------------


@pytest.mark.gpu
def test_gpu_graph_path_matches_eager(gpu_model, capture_calls):
    graph = generate(gpu_model, None, PROMPT, max_new_tokens=12, cuda_graph=True)
    assert capture_calls == [1]
    eager = generate(gpu_model, None, PROMPT, max_new_tokens=12, cuda_graph=False)
    assert graph == eager and len(graph) == 12


@pytest.mark.gpu
def test_gpu_capture_failure_logs_one_info_line_and_result_is_correct(gpu_model, monkeypatch, caplog):
    def fail(self):
        raise RuntimeError("forced capture failure")

    expected = generate(gpu_model, None, PROMPT, max_new_tokens=12, cuda_graph=False)
    monkeypatch.setattr(cg.CUDAGraphDecoder, "capture", fail)
    with caplog.at_level(logging.INFO, logger="quf.runtime"):
        out = generate(gpu_model, None, PROMPT, max_new_tokens=12, cuda_graph=True)

    records = [r for r in caplog.records if r.name == "quf.runtime"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "forced capture failure" in records[0].getMessage()
    assert out == expected


@pytest.mark.gpu
def test_gpu_env_var_disables_graph(gpu_model, monkeypatch, capture_calls):
    monkeypatch.setenv(cg.ENV_DISABLE, "1")
    out = generate(gpu_model, None, PROMPT, max_new_tokens=8)
    assert capture_calls == []
    assert cg.make_decoder(gpu_model, gpu_model.new_static_cache(16)) is None
    monkeypatch.delenv(cg.ENV_DISABLE)
    assert generate(gpu_model, None, PROMPT, max_new_tokens=8) == out
    assert capture_calls == [1]


@pytest.mark.gpu
def test_gpu_explicit_false_disables_graph(gpu_model, monkeypatch, capture_calls):
    monkeypatch.delenv(cg.ENV_DISABLE, raising=False)
    generate(gpu_model, None, PROMPT, max_new_tokens=8, cuda_graph=False)
    assert capture_calls == []


@pytest.mark.gpu
def test_gpu_sampling_is_deterministic_for_fixed_seed_under_graph(gpu_model, monkeypatch, capture_calls):
    monkeypatch.delenv(cg.ENV_DISABLE, raising=False)
    kw = dict(max_new_tokens=16, temperature=0.9, top_k=10, seed=1234)
    a = generate(gpu_model, None, PROMPT, **kw)
    b = generate(gpu_model, None, PROMPT, **kw)
    assert capture_calls == [1, 1]  # the graph path was used both times
    assert a == b and len(a) == 16
