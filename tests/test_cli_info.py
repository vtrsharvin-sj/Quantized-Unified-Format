"""Property test for `quf info` output (Property 16)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile

import numpy as np
from hypothesis import given, strategies as st

from quf.cli import main
from quf.format.constants import GROUP_SIZE
from quf.format.schema import QUFHeader
from quf.format.writer import write_quf

_scalars = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12),
    st.booleans(),
)
_config = st.dictionaries(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
    _scalars,
    min_size=1,
    max_size=6,
)


@given(
    model_config=_config,
    extra_quant=_config,
    n_tensors=st.integers(min_value=0, max_value=6),
    data=st.data(),
)
def test_info_output_completeness(model_config, extra_quant, n_tensors, data):
    """**Validates: Requirements 12.2**"""
    quantization = dict(extra_quant)
    quantization.update({"bits": 4, "group_size": GROUP_SIZE, "symmetric": False})

    tensors = {}
    for i in range(n_tensors):
        n = data.draw(st.integers(min_value=1, max_value=16))
        tensors[f"t{i}.bias"] = np.zeros(n, dtype=np.float32)

    header = QUFHeader(model_config, {"files": {}}, quantization, [])
    fd, path = tempfile.mkstemp(suffix=".quf")
    os.close(fd)
    try:
        write_quf(path, header, tensors)
        size = os.path.getsize(path)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["info", path])
        text = out.getvalue()
    finally:
        os.remove(path)

    assert code == 0, err.getvalue()
    # Model config and quantization scheme: every key and value is present.
    assert json.dumps(model_config, indent=2, sort_keys=True) in text
    assert json.dumps(quantization, indent=2, sort_keys=True) in text
    assert f"Tensor count: {n_tensors}" in text
    assert f"File size: {size} bytes" in text


def test_info_summarizes_long_module_list():
    from quf.cli import _summarize_modules

    modules = {f"layer{i}": "int4" if i % 2 else "int8" for i in range(30)}
    quant = {"bits": 4, "modules": modules}
    short = _summarize_modules(quant, verbose=False)
    assert short["modules"]["count"] == 30
    assert short["modules"]["by_precision"] == {"int4": 15, "int8": 15}
    assert _summarize_modules(quant, verbose=True)["modules"] == modules
    small = {"bits": 4, "modules": {"a": "int4"}}
    assert _summarize_modules(small, verbose=False) == small
