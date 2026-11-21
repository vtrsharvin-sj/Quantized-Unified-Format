"""Unit tests for Backend_Selector logging and the CLI ``--device`` option (Req 10.1, 10.7, 10.8)."""

from __future__ import annotations

import logging

import pytest

from quf import cli
from quf.device import DeviceInfo
from quf.runtime import selector
from quf.runtime.selector import Int4UnavailableError, NoGPUError, select_backend

NO_GPU = DeviceInfo(False, None, None, None)
GPU_80 = DeviceInfo(True, "Fake", (8, 0), 1 << 33)
GPU_75 = DeviceInfo(True, "Fake", (7, 5), 1 << 33)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("QUF_NO_TRITON", raising=False)


def _ok():
    return True, "ok"


def _bad():
    return False, "triton import failed: nope"


def _selector_records(caplog):
    return [r for r in caplog.records if r.name == "quf.runtime.selector"]


@pytest.mark.parametrize(
    "option,info,kwargs,backend",
    [
        ("cpu", GPU_80, {}, "cpu"),
        ("auto", NO_GPU, {}, "cpu"),
        ("auto", GPU_75, {}, "torch_gpu"),
        ("auto", GPU_80, {}, "torch_gpu"),
        ("cuda", GPU_80, {}, "torch_gpu"),
        ("auto", GPU_80, {"prefer_fused": True}, "torch_fused"),
        ("auto", GPU_80, {"cached_bytes": 10, "free_bytes": 5}, "torch_fused"),
        ("auto", GPU_80, {"cached_bytes": 5, "free_bytes": 10}, "torch_gpu"),
        ("auto", GPU_80, {"int4_probe": _ok}, "triton_int4"),
        ("cuda", GPU_80, {"int4_probe": _ok}, "triton_int4"),
        ("auto", GPU_80, {"int4_probe": _ok, "prefer_fused": True}, "torch_fused"),
        ("auto", GPU_80, {"int4_probe": _ok, "backend_option": "fused"}, "torch_fused"),
        ("auto", GPU_80, {"int4_probe": _ok, "backend_option": "cached"}, "torch_gpu"),
        ("auto", GPU_80, {"int4_probe": _ok, "backend_option": "marlin"}, "marlin"),
        ("auto", GPU_80, {"int4_probe": _ok, "backend_option": "int4"}, "triton_int4"),
        ("auto", GPU_80, {"int4_probe": _bad}, "torch_gpu"),
        ("auto", NO_GPU, {"int4_probe": _ok}, "cpu"),
        ("cpu", GPU_80, {"int4_probe": _ok}, "cpu"),
        ("auto", GPU_80, {"backend_option": "marlin"}, "marlin"),
        ("auto", GPU_80, {"backend_option": "fused"}, "torch_fused"),
        ("auto", GPU_80, {"backend_option": "cached", "prefer_fused": True}, "torch_gpu"),
    ],
)
def test_selection_logs_backend_and_reason(caplog, option, info, kwargs, backend):
    with caplog.at_level(logging.INFO, logger="quf.runtime.selector"):
        choice = select_backend(option, info, **kwargs)
    assert choice.backend == backend
    infos = [r for r in _selector_records(caplog) if r.levelno == logging.INFO]
    assert len(infos) == 1
    msg = infos[0].getMessage()
    assert msg == f"Selected backend {backend}: {choice.reason}"
    assert choice.reason


@pytest.mark.parametrize("test", [lambda: False, lambda: 1 / 0])
def test_failed_fused_self_test_logs_info_and_falls_back(caplog, test):
    with caplog.at_level(logging.INFO, logger="quf.runtime.selector"):
        choice = select_backend("auto", GPU_80, prefer_fused=True, fused_self_test=test)
    assert choice.backend == "torch_gpu"
    assert not any(r.levelno >= logging.WARNING for r in _selector_records(caplog))
    assert any("falling back" in r.getMessage() for r in _selector_records(caplog))


def test_marlin_requires_gpu_and_gpu_device():
    with pytest.raises(NoGPUError):
        select_backend("auto", NO_GPU, backend_option="marlin")
    with pytest.raises(ValueError):
        select_backend("cpu", GPU_80, backend_option="marlin")


def test_marlin_backend_without_gptqmodel_names_package():
    from quf.runtime.backends.marlin import MarlinUnavailableError, _load_gptqmodel, require_marlin

    try:
        _load_gptqmodel()
    except MarlinUnavailableError:
        with pytest.raises(MarlinUnavailableError, match="gptqmodel"):
            require_marlin()
    else:
        require_marlin()


@pytest.mark.parametrize("cmd", [["run", "m.quf", "--prompt", "hi"], ["bench", "m.quf", "--out", "o.json"]])
def test_backend_option_default_and_values(cmd):
    parser = cli._build_parser()
    assert parser.parse_args(cmd).backend == "auto"
    assert parser.parse_args([*cmd, "--backend", "marlin"]).backend == "marlin"
    assert parser.parse_args([*cmd, "--backend", "int4"]).backend == "int4"
    assert parser.parse_args([*cmd, "--prefer-fused"]).prefer_fused is True

def test_cuda_without_gpu_raises():
    with pytest.raises(NoGPUError):
        select_backend("cuda", NO_GPU)


def test_unknown_device_option_rejected():
    with pytest.raises(ValueError):
        select_backend("tpu", NO_GPU)  # type: ignore[arg-type]


@pytest.mark.parametrize("cmd", [["run", "m.quf", "--prompt", "hi"], ["bench", "m.quf", "--out", "o.json"]])
@pytest.mark.parametrize("value", ["auto", "cpu", "cuda"])
def test_device_values_accepted(cmd, value):
    args = cli._build_parser().parse_args([*cmd, "--device", value])
    assert args.device == value


@pytest.mark.parametrize("cmd", [["run", "m.quf", "--prompt", "hi"], ["bench", "m.quf", "--out", "o.json"]])
def test_device_default_auto_and_invalid_rejected(cmd, capsys):
    assert cli._build_parser().parse_args(cmd).device == "auto"
    assert cli.main([*cmd, "--device", "tpu"]) == 2
    assert "invalid choice" in capsys.readouterr().err


def test_run_cuda_without_gpu_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr("quf.device.detect_devices", lambda: NO_GPU)
    code = cli.main(["run", "missing.quf", "--prompt", "hi", "--device", "cuda"])
    err = capsys.readouterr().err
    assert code != 0
    assert "no NVIDIA GPU" in err


def test_run_auto_logs_selection_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr("quf.device.detect_devices", lambda: NO_GPU)
    code = cli.main(["run", "missing.quf", "--prompt", "hi", "--device", "auto"])
    err = capsys.readouterr().err
    assert code != 0  # the file does not exist, but selection happened first
    assert "Selected backend cpu: no NVIDIA GPU was found" in err



def test_auto_int4_failure_logs_one_info_line_with_reason(caplog):
    with caplog.at_level(logging.INFO, logger="quf.runtime.selector"):
        choice = select_backend("auto", GPU_80, int4_probe=_bad)
    assert choice.backend == "torch_gpu"
    infos = [r for r in _selector_records(caplog) if r.levelno >= logging.INFO]
    assert len(infos) == 1 and infos[0].levelno == logging.INFO
    assert "triton import failed: nope" in infos[0].getMessage()


def test_auto_int4_probe_raising_falls_back():
    def boom():
        raise RuntimeError("kaput")

    choice = select_backend("auto", GPU_80, int4_probe=boom)
    assert choice.backend == "torch_gpu" and "kaput" in choice.reason


def test_default_switch_off_keeps_int4_opt_in(monkeypatch):
    monkeypatch.setattr(selector, "INT4_DEFAULT_ON", False)
    calls = []
    choice = select_backend("auto", GPU_80, int4_probe=lambda: calls.append(1) or (True, "ok"))
    assert choice.backend == "torch_gpu" and not calls
    assert select_backend("auto", GPU_80, backend_option="int4", int4_probe=_ok).backend == "triton_int4"


def test_no_triton_env_disables_int4_without_calling_probe(monkeypatch):
    monkeypatch.setenv("QUF_NO_TRITON", "1")
    calls = []

    def probe():
        calls.append(1)
        return True, "ok"

    choice = select_backend("auto", GPU_80, int4_probe=probe)
    assert choice.backend == "torch_gpu" and "QUF_NO_TRITON" in choice.reason
    with pytest.raises(Int4UnavailableError, match="QUF_NO_TRITON"):
        select_backend("auto", GPU_80, backend_option="int4", int4_probe=probe)
    assert not calls


def test_explicit_int4_failure_raises_with_reason():
    with pytest.raises(Int4UnavailableError, match="triton import failed: nope"):
        select_backend("auto", GPU_80, backend_option="int4", int4_probe=_bad)
    with pytest.raises(NoGPUError):
        select_backend("auto", NO_GPU, backend_option="int4", int4_probe=_ok)
    with pytest.raises(ValueError):
        select_backend("cpu", GPU_80, backend_option="int4", int4_probe=_ok)


def test_cpu_selection_never_calls_probe():
    calls = []
    select_backend("auto", NO_GPU, int4_probe=lambda: calls.append(1) or (True, "ok"))
    select_backend("cpu", GPU_80, int4_probe=lambda: calls.append(1) or (True, "ok"))
    assert not calls


def test_int4_probe_with_no_triton_env_does_not_import_triton(monkeypatch):
    import sys

    monkeypatch.setenv("QUF_NO_TRITON", "1")
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    ok, why = selector.int4_probe()
    assert not ok and "QUF_NO_TRITON" in why and "triton" not in sys.modules


def test_run_int4_without_gpu_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr("quf.device.detect_devices", lambda: NO_GPU)
    code = cli.main(["run", "missing.quf", "--prompt", "hi", "--backend", "int4"])
    assert code != 0
    assert "quf: error:" in capsys.readouterr().err

def test_graph_disabled_prefers_torch_gpu_without_probing():
    calls = []
    choice = select_backend("auto", GPU_80, int4_probe=lambda: calls.append(1) or (True, "ok"), cuda_graph=False)
    assert choice.backend == "torch_gpu" and not calls
    assert "CUDA graph is disabled" in choice.reason
    # graph enabled (default) still picks int4
    assert select_backend("auto", GPU_80, int4_probe=_ok, cuda_graph=True).backend == "triton_int4"
    # an explicit request is honored whatever the graph setting
    assert select_backend("auto", GPU_80, backend_option="int4", int4_probe=_ok, cuda_graph=False).backend == "triton_int4"


def test_default_decision_is_int4_on():
    assert selector.INT4_DEFAULT_ON is True


def test_model_load_reads_no_cuda_graph_env(monkeypatch):
    from quf.runtime import model as model_mod

    seen = {}
    monkeypatch.setattr(model_mod, "estimate_cached_bytes", lambda p: 1)
    monkeypatch.setattr(model_mod, "_available_gpu_bytes", lambda: 1 << 30)
    monkeypatch.setattr("quf.device.detect_devices", lambda: GPU_80)

    def fake_select(*a, **kw):
        seen["cuda_graph"] = kw["cuda_graph"]
        return selector.BackendChoice("torch_gpu", "x")

    monkeypatch.setattr(selector, "select_backend", fake_select)
    monkeypatch.setattr(model_mod.QUFModel, "_load_with", classmethod(lambda cls, p, b, f=True: b))
    monkeypatch.delenv("QUF_NO_CUDA_GRAPH", raising=False)
    model_mod.QUFModel.load("x.quf", device_option="cuda")
    assert seen["cuda_graph"] is True
    monkeypatch.setenv("QUF_NO_CUDA_GRAPH", "1")
    model_mod.QUFModel.load("x.quf", device_option="cuda")
    assert seen["cuda_graph"] is False
    model_mod.QUFModel.load("x.quf", device_option="cuda", cuda_graph=True)
    assert seen["cuda_graph"] is True


def test_capture_failure_with_int4_logs_one_info_line_with_hint(monkeypatch, caplog):
    from quf.runtime import cuda_graph as cg

    class FakeModel:
        backend_name = "triton_int4"

    class FakeDec:
        def __init__(self, model, cache):
            pass

        def capture(self):
            raise RuntimeError("no capture")

    monkeypatch.setattr(cg.CUDAGraphDecoder, "available", staticmethod(lambda m: None))
    monkeypatch.setattr(cg, "CUDAGraphDecoder", type("D", (FakeDec,), {"available": staticmethod(lambda m: None)}))
    monkeypatch.setattr(cg.torch.cuda, "synchronize", lambda: None)
    with caplog.at_level(logging.INFO, logger="quf.runtime"):
        assert cg.make_decoder(FakeModel(), None, True) is None
    msgs = [r.getMessage() for r in caplog.records if "capture failed" in r.getMessage()]
    assert len(msgs) == 1 and "--backend cached" in msgs[0]
    FakeModel.backend_name = "torch_gpu"
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="quf.runtime"):
        cg.make_decoder(FakeModel(), None, True)
    assert "--backend cached" not in caplog.text
