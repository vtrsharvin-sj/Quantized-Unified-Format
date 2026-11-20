"""Device detection. ``torch`` is imported lazily so the package imports without extras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DeviceInfo:
    cuda_available: bool
    name: Optional[str]
    compute_capability: Optional[Tuple[int, int]]
    total_memory_bytes: Optional[int]


_NO_GPU = DeviceInfo(False, None, None, None)


CUDA_TORCH_INSTALL = "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"


def nvidia_gpu_present() -> bool:
    """True when ``nvidia-smi`` lists a GPU. Never raises."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:
        return False


def cpu_only_torch_hint() -> Optional[str]:
    """One-line hint when an NVIDIA GPU exists but PyTorch cannot use it, else ``None``."""
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return None
        cpu_build = getattr(torch.version, "cuda", None) is None
    except Exception:
        return None
    if not nvidia_gpu_present():
        return None
    if cpu_build:
        return ("an NVIDIA GPU was found but PyTorch is CPU-only, running on CPU; "
                f"for GPU speed run: {CUDA_TORCH_INSTALL}")
    return ("an NVIDIA GPU was found but PyTorch cannot use it, running on CPU; "
            "check the NVIDIA driver (nvidia-smi)")


def detect_devices() -> DeviceInfo:
    """Return details of CUDA device 0, or ``DeviceInfo(False, None, None, None)``.

    Never raises: a missing torch or any CUDA query failure counts as no GPU.
    """
    try:
        import torch
    except Exception:
        return _NO_GPU
    try:
        if not torch.cuda.is_available():
            return _NO_GPU
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        return DeviceInfo(
            True,
            str(torch.cuda.get_device_name(0)),
            (int(major), int(minor)),
            int(props.total_memory),
        )
    except Exception:
        return _NO_GPU
