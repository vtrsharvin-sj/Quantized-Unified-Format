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
    print("OK")
    """
)


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
