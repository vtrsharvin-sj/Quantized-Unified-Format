"""Regression tests: every CLI failure is one ``quf: error:`` line, a nonzero exit, no traceback."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from quf import __version__
from quf.cli import main


def _run(capsys, argv):
    code = main(argv)
    err = capsys.readouterr().err
    return code, err


def _assert_clean_error(code, err, expect=None, exit_code=1):
    assert code == exit_code
    assert "Traceback" not in err
    errors = [ln for ln in err.splitlines() if ln.startswith("quf: error:")]
    assert len(errors) == 1, err
    if expect:
        assert expect.lower() in errors[0].lower(), errors[0]


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert __version__ in out


def test_version_subprocess():
    r = subprocess.run([sys.executable, "-m", "quf.cli", "--version"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert __version__ in r.stdout


@pytest.mark.parametrize("cmd", ["info", "run", "bench"])
def test_missing_file(capsys, tmp_path, cmd):
    missing = str(tmp_path / "nope.quf")
    argv = {
        "info": ["info", missing],
        "run": ["run", missing, "--prompt", "hi"],
        "bench": ["bench", missing, "--out", str(tmp_path / "o.json")],
    }[cmd]
    code, err = _run(capsys, argv)
    _assert_clean_error(code, err, "not found")


def test_directory_instead_of_file(capsys, tmp_path):
    code, err = _run(capsys, ["info", str(tmp_path)])
    _assert_clean_error(code, err, "directory")


@pytest.mark.parametrize("name,data", [
    ("bad.quf", b"this is not a quf file at all"),
    ("empty.quf", b""),
    ("model.txt", b"QUF1garbage"),
    ("trunc.quf", b"QUF1\x00\x00"),
])
def test_corrupt_or_wrong_extension_info(capsys, tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    code, err = _run(capsys, ["info", str(p)])
    _assert_clean_error(code, err)


def test_corrupt_file_run(capsys, tmp_path):
    p = tmp_path / "bad.quf"
    p.write_bytes(b"garbage" * 10)
    code, err = _run(capsys, ["run", str(p), "--prompt", "hello", "--device", "cpu"])
    _assert_clean_error(code, err)


def test_output_exists(capsys, tmp_path):
    out = tmp_path / "out.quf"
    out.write_bytes(b"x")
    code, err = _run(capsys, ["convert", "some/model", str(out)])
    _assert_clean_error(code, err, "already exists")
    assert out.read_bytes() == b"x"


def test_output_directory_missing(capsys, tmp_path):
    code, err = _run(capsys, ["convert", "some/model", str(tmp_path / "nodir" / "o.quf")])
    _assert_clean_error(code, err, "does not exist")


def test_missing_convert_extras(capsys, tmp_path, monkeypatch):
    import quf.convert.converter as conv

    def boom(*a, **k):
        raise ImportError("No module named 'transformers'", name="transformers")

    monkeypatch.setattr(conv, "convert_checkpoint", boom)
    monkeypatch.setattr(conv, "read_source_config", boom)
    code, err = _run(capsys, ["convert", "some/model", str(tmp_path / "o.quf")])
    _assert_clean_error(code, err, "quf[convert]")


def test_missing_convert_extras_precheck(capsys, tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: None if n == "datasets" else real(n, *a, **k))
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    code, err = _run(capsys, ["convert", "some/model", str(tmp_path / "o.quf")])
    _assert_clean_error(code, err, "quf[convert]")


def test_nonexistent_model_id(capsys, tmp_path, monkeypatch):
    import quf.convert.converter as conv

    class RepositoryNotFoundError(Exception):
        pass

    def boom(*a, **k):
        raise RepositoryNotFoundError("404 Client Error: Repository not found")

    monkeypatch.setattr(conv, "convert_checkpoint", boom)
    monkeypatch.setattr(conv, "read_source_config", boom)
    code, err = _run(capsys, ["convert", "no/such-model", str(tmp_path / "o.quf")])
    _assert_clean_error(code, err, "not found on Hugging Face")


def test_no_network(capsys, tmp_path, monkeypatch):
    import quf.convert.converter as conv

    class ConnectError(Exception):
        pass

    def boom(*a, **k):
        raise ConnectError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(conv, "convert_checkpoint", boom)
    monkeypatch.setattr(conv, "read_source_config", boom)
    code, err = _run(capsys, ["convert", "org/model", str(tmp_path / "o.quf")])
    _assert_clean_error(code, err, "network")


def test_device_cuda_without_gpu(capsys, tmp_path, monkeypatch):
    import quf.device as device

    monkeypatch.setattr(device, "detect_devices", lambda: device.DeviceInfo(False, None, None, None))
    p = tmp_path / "m.quf"
    p.write_bytes(b"x")
    code, err = _run(capsys, ["run", str(p), "--prompt", "hi", "--device", "cuda"])
    _assert_clean_error(code, err, "GPU")


@pytest.mark.parametrize("prompt", ["", "   \n"])
def test_empty_prompt(capsys, tmp_path, prompt):
    p = tmp_path / "m.quf"
    p.write_bytes(b"x")
    code, err = _run(capsys, ["run", str(p), "--prompt", prompt])
    _assert_clean_error(code, err, "prompt is empty")


def test_prompt_too_long(capsys, tmp_path, monkeypatch):
    import quf.runtime.generate as gen
    import quf.runtime.model as rmodel
    import quf.tokenization.wrapper as wrap
    import tokenizers

    header = SimpleNamespace(tokenizer={"files": {"tokenizer.json": "{}"}})
    monkeypatch.setattr(rmodel.QUFModel, "load", classmethod(lambda cls, *a, **k: SimpleNamespace(header=header)))
    monkeypatch.setattr(tokenizers.Tokenizer, "from_str", staticmethod(lambda s: object()))
    monkeypatch.setattr(wrap, "TokenizerWrapper", lambda **k: SimpleNamespace())

    def too_long(*a, **k):
        raise gen.ContextLengthError(5000, 64, 4096)

    monkeypatch.setattr(gen, "generate", too_long)
    p = tmp_path / "m.quf"
    p.write_bytes(b"x")
    code, err = _run(capsys, ["run", str(p), "--prompt", "word " * 10])
    _assert_clean_error(code, err, "context length")


def test_keyboard_interrupt(capsys, tmp_path, monkeypatch):
    import quf.cli as cli

    def interrupt(args):
        raise KeyboardInterrupt

    monkeypatch.setitem(cli._HANDLERS, "info", interrupt)
    code, err = _run(capsys, ["info", str(tmp_path / "x.quf")])
    assert code == 130
    assert "Traceback" not in err
    assert "interrupted" in err
    assert len(err.strip().splitlines()) == 1


def test_unexpected_exception_hides_traceback_unless_verbose(capsys, tmp_path, monkeypatch):
    import quf.cli as cli

    def bad(args):
        raise ZeroDivisionError("boom")

    monkeypatch.setitem(cli._HANDLERS, "info", bad)
    code, err = _run(capsys, ["info", "x"])
    _assert_clean_error(code, err, "boom")
    code, err = _run(capsys, ["info", "x", "--verbose"])
    assert code == 1
    assert "Traceback" in err
    assert err.count("quf: error:") == 1


def test_gpu_hint_when_cpu_only_torch(capsys, monkeypatch):
    import quf.device as device

    monkeypatch.setattr(device, "nvidia_gpu_present", lambda: True)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None)
    hint = device.cpu_only_torch_hint()
    assert hint and "download.pytorch.org/whl/cu" in hint and "\n" not in hint


def test_no_gpu_hint_without_nvidia(monkeypatch):
    import quf.device as device

    monkeypatch.setattr(device, "nvidia_gpu_present", lambda: False)
    assert device.cpu_only_torch_hint() is None


def test_hint_printed_by_auto_device(capsys, tmp_path, monkeypatch):
    import quf.device as device

    monkeypatch.setattr(device, "cpu_only_torch_hint", lambda: "HINT-LINE")
    p = tmp_path / "m.quf"
    p.write_bytes(b"x")
    code, err = _run(capsys, ["bench", str(p), "--out", str(tmp_path / "o.json")])
    assert "quf: HINT-LINE" in err
    assert "Traceback" not in err


def test_hint_not_printed_for_explicit_device(capsys, tmp_path, monkeypatch):
    import quf.device as device

    monkeypatch.setattr(device, "cpu_only_torch_hint", lambda: "HINT-LINE")
    p = tmp_path / "m.quf"
    p.write_bytes(b"x")
    _, err = _run(capsys, ["bench", str(p), "--out", str(tmp_path / "o.json"), "--device", "cpu"])
    assert "HINT-LINE" not in err
