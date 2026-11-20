"""Unit tests for Backend_Selector logging and the CLI ``--device`` option (Req 10.1, 10.7, 10.8)."""

from __future__ import annotations

import logging

import pytest

from quf import cli
from quf.device import DeviceInfo
from quf.runtime.selector import NoGPUError, select_backend

NO_GPU = DeviceInfo(False, None, None, None)
GPU_80 = DeviceInfo(True, "Fake", (8, 0), 1 << 33)
GPU_75 = DeviceInfo(True, "Fake", (7, 5), 1 << 33)


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

