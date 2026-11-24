"""Tests for the Python API, ``quf.load`` and ``quf.Model`` (CPU, runs in CI)."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import quf
from quf import api, cli
from quf.format.errors import QUFFormatError
from quf.format.schema import QUFHeader
from quf.runtime.generate import ContextLengthError, generate
from quf.runtime.model import QUFModel
from quf.tokenization.wrapper import TokenizerWrapper, tokenizer_from_header

from test_model import VOCAB, build_tiny_quf
from test_runtime_unit import WORDS, make_quf

ROOT = Path(__file__).resolve().parents[1]
# Words that the tiny WordLevel tokenizer maps to one token each.
PROMPT_WORDS = [w for w in WORDS if w.isalnum()] + [f"w{i}" for i in range(len(WORDS), VOCAB)]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """Tiny files with an embedded tokenizer and the same weights. "plain" uses EOS id 31, as
    the runtime tests do. "eos" uses the token that greedy decoding emits most often from
    one-word prompts, so many of its runs stop at the end-of-sequence token."""
    d = tmp_path_factory.mktemp("api")
    plain = make_quf(d / "plain.quf", eos_token_id=VOCAB - 1)
    model = QUFModel.load(plain, "cpu")
    tok = tokenizer_from_header(model.header)
    counts: Counter[int] = Counter()
    for word in PROMPT_WORDS:
        counts.update(generate(model, tok, word, max_new_tokens=8))
    eos = counts.most_common(1)[0][0]
    return {"plain": plain, "eos": make_quf(d / "eos.quf", eos_token_id=eos), "eos_id": eos}


@pytest.fixture(scope="module")
def model(files):
    with quf.load(files["plain"], device="cpu") as m:
        yield m


def _quf_run(path, prompt, **options) -> str:
    """What `quf run --device cpu` writes to stdout, without its final newline."""
    argv = ["run", str(path), "--device", "cpu", "--prompt", prompt]
    argv += [f"--{k.replace('_', '-')}={v}" for k, v in options.items()]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(argv)
    assert code == 0
    printed = out.getvalue()
    assert printed.endswith("\n"), printed
    return printed[:-1]


# Feature: quf-quantized-format, Property 44: The Python API returns the same text as quf run
@settings(max_examples=50, deadline=None)
@given(
    which=st.sampled_from(["plain", "eos"]),
    words=st.lists(st.sampled_from(PROMPT_WORDS), min_size=1, max_size=8),
    sep=st.sampled_from([" ", "  ", "\n", "\t"]),
    n=st.integers(min_value=1, max_value=16),
)
def test_property_44_api_returns_the_text_of_quf_run(files, which, words, sep, n):
    """**Validates: Requirements 43.3, 43.5, 43.6, 43.7**"""
    path = files[which]
    prompt = sep.join(words)
    pieces: list[str] = []
    with quf.load(path, device="cpu") as m:
        text = m.generate(prompt, max_new_tokens=n, temperature=0, stream=pieces.append)
    assert "".join(pieces) == text

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(["run", str(path), "--device", "cpu", "--prompt", prompt,
                         "--max-new-tokens", str(n), "--temperature", "0"])
    assert code == 0
    printed = out.getvalue()
    assert printed.endswith("\n")
    assert text == printed[:-1]


def test_eos_file_stops_early_and_matches_quf_run(files):
    model = QUFModel.load(files["eos"], "cpu")
    tok = tokenizer_from_header(model.header)
    stopped = [w for w in PROMPT_WORDS
               if generate(model, tok, w, max_new_tokens=16)[-1:] == [files["eos_id"]]]
    assert stopped, "no prompt reaches the EOS token, so Property 44 would not cover it"
    with quf.load(files["eos"], device="cpu") as m:
        text = m.generate(stopped[0], max_new_tokens=16)
    assert text == _quf_run(files["eos"], stopped[0], max_new_tokens=16, temperature=0)


def test_handle_attributes(model, files):
    assert isinstance(model, quf.Model) and isinstance(model, api.Model)
    assert model.backend == "cpu"
    assert isinstance(model.header, QUFHeader)
    assert model.header.model_config["architecture"] == "Qwen2ForCausalLM"
    assert model.header.tokenizer["eos_token_id"] == VOCAB - 1
    assert isinstance(model.tokenizer, TokenizerWrapper)
    assert model.tokenizer.encode("hello world") == [1, 2]


def test_stream_pieces_join_to_the_returned_text(model):
    pieces: list[str] = []
    text = model.generate("hello world", max_new_tokens=6, stream=pieces.append)
    assert pieces and "".join(pieces) == text
    assert model.generate("hello world", max_new_tokens=6) == text


def test_sampling_options_match_quf_run(model, files):
    opts = {"temperature": 1.0, "top_k": 5, "top_p": 0.9, "seed": 7}
    text = model.generate("foo bar", max_new_tokens=6, **opts)
    assert model.generate("foo bar", max_new_tokens=6, **opts) == text
    assert text == _quf_run(files["plain"], "foo bar", max_new_tokens=6, **opts)


# Each value just outside a `quf run` limit, and each value on it.
INVALID = [
    ({"prompt": ""}, "prompt"),
    ({"prompt": " \n\t"}, "prompt"),
    ({"max_new_tokens": 0}, "max_new_tokens"),
    ({"temperature": -0.001}, "temperature"),
    ({"top_k": 0}, "top_k"),
    ({"top_p": 0.0}, "top_p"),
    ({"top_p": 1.001}, "top_p"),
]
BOUNDARIES = [
    {"max_new_tokens": 1},
    {"temperature": 0.0},
    {"temperature": 1.0, "top_k": 1},
    {"temperature": 1.0, "top_p": 1.0},
    {"temperature": 1.0, "top_p": 0.001},
]


@pytest.mark.parametrize("kwargs,name", INVALID)
def test_generate_rejects_values_that_quf_run_rejects(model, files, capsys, kwargs, name):
    args = {"prompt": "hello", "max_new_tokens": 2, **kwargs}
    with pytest.raises(ValueError, match=name):
        model.generate(**args)
    prompt = args.pop("prompt")
    argv = ["run", str(files["plain"]), "--device", "cpu", f"--prompt={prompt}"]
    argv += [f"--{k.replace('_', '-')}={v}" for k, v in args.items()]
    assert cli.main(argv) == 1
    assert "quf: error:" in capsys.readouterr().err


@pytest.mark.parametrize("kwargs", BOUNDARIES)
def test_generate_accepts_the_boundary_values_of_quf_run(model, files, kwargs):
    args = {"max_new_tokens": 2, **kwargs}
    assert model.generate("hello", **args) == _quf_run(files["plain"], "hello", **args)


def test_generate_type_errors(model):
    with pytest.raises(TypeError, match="prompt"):
        model.generate([1, 2])
    with pytest.raises(TypeError, match="stream"):
        model.generate("hello", stream="not callable")


def test_context_length_error_propagates(model):
    with pytest.raises(ContextLengthError):
        model.generate("hello", max_new_tokens=100_000)


def test_close_twice_then_generate_raises(files):
    m = quf.load(files["plain"], device="cpu")
    m.close()
    m.close()
    assert m.tokenizer is None
    with pytest.raises(ValueError, match="the model is closed"):
        m.generate("hello")
    assert m.backend == "cpu" and isinstance(m.header, QUFHeader)


def test_context_manager_releases_the_model_and_the_file(tmp_path):
    path = make_quf(tmp_path / "ctx.quf", eos_token_id=VOCAB - 1)
    with quf.load(path, device="cpu") as m:
        assert isinstance(m.generate("hello", max_new_tokens=2), str)
    assert m.tokenizer is None and m._model is None
    with pytest.raises(ValueError, match="the model is closed"):
        m.generate("hello")
    os.remove(path)  # Windows refuses to delete a file that is still open or memory-mapped

    with pytest.raises(RuntimeError, match="boom"):
        with quf.load(make_quf(tmp_path / "ctx2.quf"), device="cpu") as other:
            raise RuntimeError("boom")
    assert other.tokenizer is None and other._model is None


def _run_choices() -> dict[str, tuple]:
    parser = cli._build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {a.dest: tuple(a.choices) for a in sub.choices["run"]._actions if a.choices}


def test_load_checks_device_and_backend(files, monkeypatch):
    choices = _run_choices()
    assert api.DEVICES == choices["device"] and api.BACKENDS == choices["backend"]
    with pytest.raises(ValueError, match="device must be one of auto, cpu, cuda; got 'tpu'"):
        quf.load(files["plain"], device="tpu")
    with pytest.raises(ValueError,
                       match="backend must be one of auto, cached, fused, marlin, int4; got 'fast'"):
        quf.load(files["plain"], backend="fast")

    import quf.device as device
    from quf.runtime.selector import NoGPUError

    monkeypatch.setattr(device, "detect_devices", lambda: device.DeviceInfo(False, None, None, None))
    with pytest.raises(NoGPUError, match="no NVIDIA GPU"):
        quf.load(files["plain"], device="cuda")


def test_load_missing_directory_and_corrupt_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        quf.load(tmp_path / "nope.quf", device="cpu")
    with pytest.raises(IsADirectoryError):
        quf.load(tmp_path, device="cpu")
    bad = tmp_path / "bad.quf"
    bad.write_bytes(b"garbage" * 10)
    with pytest.raises(QUFFormatError):
        quf.load(bad, device="cpu")


def test_file_without_tokenizer_json(tmp_path, capsys):
    path = tmp_path / "notok.quf"
    build_tiny_quf(path)
    with pytest.raises(ValueError, match="the .quf file does not contain tokenizer.json"):
        quf.load(path, device="cpu")
    # quf run builds its tokenizer with the same helper and reports the same problem.
    assert cli.main(["run", str(path), "--device", "cpu", "--prompt", "hello"]) == 1
    assert "quf: error: the .quf file does not contain tokenizer.json" in capsys.readouterr().err


@pytest.mark.gpu
@pytest.mark.triton
def test_load_and_generate_on_cuda_emit_no_warnings(files):
    """The API path shows no Triton warning, such as "TRITON_DEFAULT_FP_FUSION is unset".

    Runs in a fresh interpreter: Triton emits that warning once per process, and this test
    process already set TRITON_DEFAULT_FP_FUSION when it imported quf.runtime.kernels, so
    the child gets an environment without it.
    """
    code = (
        "import json, os, sys, warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import quf\n"
        "    with quf.load(sys.argv[1], device='cuda') as model:\n"
        "        backend = model.backend\n"
        "        text = model.generate('hello world', max_new_tokens=8)\n"
        "print(json.dumps({'backend': backend, 'text': text,\n"
        "                  'fp_fusion': os.environ.get('TRITON_DEFAULT_FP_FUSION'),\n"
        "                  'warnings': [f'{w.category.__name__}: {w.message}' for w in caught]}))\n"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("TRITON_DEFAULT_FP_FUSION", "QUF_NO_TRITON", "QUF_NO_FUSED",
                        "QUF_NO_CUDA_GRAPH", "PYTHONWARNINGS")}
    r = subprocess.run([sys.executable, "-c", code, str(files["plain"])], capture_output=True,
                       text=True, cwd=ROOT, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    import json

    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["backend"] == "triton_int4", out
    assert isinstance(out["text"], str)
    assert out["fp_fusion"] == "1"  # Triton's own default, set by quf.runtime.kernels
    assert out["warnings"] == [], out["warnings"]
    assert "Warning" not in r.stderr, r.stderr


def test_import_quf_is_lazy():
    code = (
        "import sys\n"
        "def heavy():\n"
        "    return sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'tokenizers')\n"
        "                  or m.startswith('quf.runtime'))\n"
        "import quf\n"
        "assert not heavy() and 'quf.api' not in sys.modules, heavy()\n"
        "load, Model = quf.load, quf.Model\n"  # resolved on first access
        "api = sys.modules['quf.api']\n"
        "assert load is api.load and Model is api.Model\n"
        "assert {'api', 'load', 'Model'} <= set(quf.__all__) & set(dir(quf))\n"
        "assert not heavy(), heavy()\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OK"
