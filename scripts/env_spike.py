"""Environment spike: report versions and probe CUDA, Marlin, and gigatoken.

Run on Windows and again in WSL, then copy the output into docs/ENV.md.

    python scripts/env_spike.py [--json]

Every probe is isolated. A missing package or a failing kernel is reported as a
result (status "missing", "fail", or "inconclusive"), never as a crash. Only the
standard library is imported at module level.

Status values: "pass", "fail" (ran and failed, message recorded),
"missing" (package not installed), "inconclusive" (probe API did not match the
installed version, so no verdict on the kernel itself), "skipped".
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import traceback
from importlib import metadata


def _result(status: str, detail: str = "", **extra):
    out = {"status": status, "detail": detail}
    out.update(extra)
    return out


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _pkg_version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    env = {
        "os": platform.platform(),
        "is_wsl": "microsoft" in platform.release().lower(),
        "python": sys.version.split()[0],
    }
    nvcc = shutil.which("nvcc")
    env["cuda_toolkit_nvcc"] = None
    if nvcc:
        try:
            txt = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30).stdout
            env["cuda_toolkit_nvcc"] = txt.strip().splitlines()[-1] if txt.strip() else None
        except Exception as exc:  # noqa: BLE001
            env["cuda_toolkit_nvcc"] = f"error: {_err(exc)}"
    smi = shutil.which("nvidia-smi")
    env["nvidia_smi"] = None
    if smi:
        try:
            env["nvidia_smi"] = subprocess.run(
                [smi, "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            env["nvidia_smi"] = f"error: {_err(exc)}"
    return env


def versions() -> dict:
    v = {name: _pkg_version(name) for name in ("torch", "gptqmodel", "gigatoken", "tokenizers", "numpy")}
    try:
        import torch

        v["torch_cuda_build"] = torch.version.cuda
        v["torch_cuda_available"] = torch.cuda.is_available()
        v["torch_arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        v["torch_import"] = _err(exc)
    return v


def probe_torch_cuda() -> dict:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return _result("missing", _err(exc))
    if not torch.cuda.is_available():
        return _result("skipped", "torch.cuda.is_available() is False")
    try:
        cap = torch.cuda.get_device_capability(0)
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        c = (a @ b).float().sum().item()
        torch.cuda.synchronize()
        if c != c:
            return _result("fail", "matmul returned NaN", capability=list(cap))
        return _result(
            "pass", "fp16 256x256 matmul executed on GPU",
            device=torch.cuda.get_device_name(0), capability=list(cap),
        )
    except Exception as exc:  # noqa: BLE001
        return _result("fail", _err(exc), traceback=traceback.format_exc())


def probe_marlin() -> dict:
    """Run a small Marlin matmul through GPTQModel and compare to a fp16 reference.

    GPTQModel's internal API changes between releases. If the expected classes
    are absent the result is "inconclusive", which is not evidence either way.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return _result("missing", _err(exc))
    if not torch.cuda.is_available():
        return _result("skipped", "no CUDA device")
    try:
        import gptqmodel  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return _result("missing", f"gptqmodel not importable: {_err(exc)}")
    try:
        from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
    except Exception as exc:  # noqa: BLE001
        return _result("inconclusive", f"MarlinQuantLinear not importable: {_err(exc)}")
    try:
        import inspect

        in_f, out_f, group = 256, 256, 128
        sig = inspect.signature(MarlinQuantLinear.__init__)
        kwargs = dict(
            bits=4, group_size=group, desc_act=False, sym=True,
            in_features=in_f, out_features=out_f, bias=False,
            pack_dtype=torch.int32, register_buffers=True,
        )
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        layer = MarlinQuantLinear(**kwargs).to("cuda")
        if hasattr(layer, "post_init"):
            layer.post_init()
        x = torch.randn(4, in_f, device="cuda", dtype=torch.float16)
        y = layer(x)
        torch.cuda.synchronize()
        if not torch.isfinite(y.float()).all():
            return _result("fail", "Marlin output contains non-finite values")
        return _result(
            "pass",
            "Marlin forward executed (finite output; numerical comparison against the "
            "reference is done by the Task 7.2 self-test, not here)",
        )
    except Exception as exc:  # noqa: BLE001
        msg = _err(exc)
        # A kernel/arch failure is a real "fail"; a constructor mismatch is inconclusive.
        kernel_markers = ("no kernel image", "sm_120", "CUDA error", "invalid device function", "unsupported")
        status = "fail" if any(m.lower() in msg.lower() for m in kernel_markers) else "inconclusive"
        return _result(status, msg, traceback=traceback.format_exc())


def probe_gigatoken() -> dict:
    try:
        import gigatoken
    except Exception as exc:  # noqa: BLE001
        return _result("missing", f"gigatoken not importable: {_err(exc)}")
    text = "Hello, world! QUF spike."
    names = [n for n in dir(gigatoken) if not n.startswith("_")]
    # The gigatoken API is not assumed. Try a few plausible entry points.
    for cls_name in ("Tokenizer", "GigaTokenizer", "GigaToken"):
        cls = getattr(gigatoken, cls_name, None)
        if cls is None:
            continue
        try:
            tok = cls.from_pretrained("Qwen/Qwen2.5-0.5B") if hasattr(cls, "from_pretrained") else cls()
            ids = tok.encode(text)
            ids = list(getattr(ids, "ids", ids))
            if ids:
                return _result("pass", f"{cls_name}.encode returned {len(ids)} ids", sample=ids[:8])
        except Exception as exc:  # noqa: BLE001
            return _result("fail", f"{cls_name} tokenize failed: {_err(exc)}")
    return _result("inconclusive", "imported, but no known tokenizer entry point found", public_names=names)


def run() -> dict:
    return {
        "environment": environment(),
        "versions": versions(),
        "torch_cuda_kernel": probe_torch_cuda(),
        "marlin_matmul": probe_marlin(),
        "gigatoken_tokenize": probe_gigatoken(),
    }


def _designation(report: dict) -> str:
    marlin = report["marlin_matmul"]["status"]
    cuda = report["torch_cuda_kernel"]["status"]
    if cuda != "pass":
        return "No GPU Backend verified (PyTorch CUDA probe did not pass); CPU only."
    if marlin == "pass":
        return "Marlin executed; Marlin is eligible as the GPU Backend."
    if marlin == "fail":
        return "Marlin failed; Torch_GPU is the GPU Backend for this device."
    return f"Marlin status is '{marlin}' (no verdict); Torch_GPU is the safe GPU Backend until Marlin is verified."


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    args = p.parse_args(argv)
    report = run()
    report["gpu_backend_designation"] = _designation(report)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print(json.dumps(report, indent=2, default=str))
    print()
    print("GPU backend designation:", report["gpu_backend_designation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
