"""CI safety: the package imports and degrades cleanly when ``import triton`` fails (Req 36.3, 36.4)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.modules["triton"] = None  # makes `import triton` raise ImportError
    sys.modules["triton.language"] = None
    import quf, quf.runtime.model
    import quf.runtime.kernels.triton_kernels as tk
    import quf.runtime.kernels.dispatch
    import quf.runtime.backends.triton_int4 as ti
    ok, why = tk.triton_available()
    assert ok is False and why, (ok, why)
    ok, why = ti.triton_int4_self_test()
    assert ok is False and why, (ok, why)
    from quf.runtime import selector
    ok, why = selector.int4_probe()
    assert ok is False, why
    # Fused_Decode_Kernels (task 33.3): the gate fails, nothing imports a fused kernel module.
    from quf.runtime.model import QUFModel
    import quf.runtime.cuda_graph, quf.runtime.embedding, quf.runtime.generate  # noqa: F401
    ok, why = QUFModel._fused_gate()
    assert ok is False and why, (ok, why)
    for name in ("fused", "decode_gemv", "attention", "embed_gemv"):
        assert "quf.runtime.kernels." + name not in sys.modules, name
    assert "triton" not in sys.modules or sys.modules["triton"] is None
    print("OK")
    """
)

_FUSED_MODULES_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.modules["triton"] = None
    sys.modules["triton.language"] = None
    import importlib
    for name in ("fused", "decode_gemv", "attention", "embed_gemv"):
        try:
            importlib.import_module("quf.runtime.kernels." + name)
        except ImportError:
            continue
        raise AssertionError(name + " imported without Triton")
    print("OK")
    """
)


def test_fused_kernel_modules_need_triton_at_import():
    """Each fused kernel module imports Triton at the top, so only a passed gate may import it."""
    proc = subprocess.run([sys.executable, "-c", _FUSED_MODULES_SCRIPT], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout


@pytest.mark.parametrize("env_flag", [False, True], ids=["import-blocked", "QUF_NO_TRITON"])
def test_package_works_without_triton(env_flag, monkeypatch):
    import os

    env = dict(os.environ)
    env.pop("QUF_NO_TRITON", None)
    if env_flag:
        env["QUF_NO_TRITON"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, env=env, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout
