"""Writer for QUF files (versions 1, 2 and 3).

Layout (little-endian): magic ``QUF1``, uint32 version, uint64 header length,
UTF-8 JSON header, zero padding, then tensors at absolute offsets that are
multiples of 64 (zero padding between tensors).
"""

from __future__ import annotations

import os
import struct
from typing import Mapping

import numpy as np

from .constants import ALIGN, DTYPES, INT8_SUFFIX, MAGIC, PREFIX_SIZE, align_up
from .errors import HeaderSchemaError
from .schema import QUFHeader, TensorEntry
from .validate import (
    check_quantized_embeddings,
    check_tensor_group_layout,
    required_version,
)

_FP16_ONLY_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


def _dtype_code(name: str, arr: np.ndarray) -> str:
    for code, (np_str, _) in DTYPES.items():
        if arr.dtype == np.dtype(np_str):
            return code
    raise ValueError(
        f"tensor '{name}': unsupported numpy dtype {arr.dtype} (supported: fp16, fp32, int32)"
    )


def _validate(header: QUFHeader, tensors: Mapping[str, np.ndarray]) -> None:
    if "group_size" not in header.quantization:
        raise ValueError("quantization.group_size is required")
    try:
        check_tensor_group_layout(
            header.quantization,
            {n: tuple(a.shape) for n, a in tensors.items()},
            lambda n: tensors[n],
        )
    except HeaderSchemaError as e:
        raise ValueError(str(e)) from e
    codes = {n: _dtype_code(n, a) for n, a in tensors.items()}
    try:
        check_quantized_embeddings(
            header.quantization, {n: tuple(a.shape) for n, a in tensors.items()}, codes
        )
    except HeaderSchemaError as e:
        raise ValueError(str(e)) from e
    symmetric = header.quantization.get("symmetric", False)
    for module, prec in header.modules.items():
        if prec == "int8":
            need, want = module + INT8_SUFFIX, "int8"
        elif prec == "fp16":
            need, want = module + ".weight", "fp16"
        else:
            need, want = module + ".qweight", "int32"
        arr = tensors.get(need)
        if arr is None or _dtype_code(need, arr) != want:
            raise ValueError(
                f"module '{module}' is {prec}: tensor '{need}' must be present as {want}"
            )
    for name, arr in tensors.items():
        code = _dtype_code(name, arr)
        if name.endswith(INT8_SUFFIX):
            if code != "int8":
                raise ValueError(f"tensor '{name}' (int8 weight) must be int8, got {code}")
            scales = tensors.get(name[: -len(INT8_SUFFIX)] + ".scales")
            if scales is None or _dtype_code(name, scales) != "fp16":
                raise ValueError(
                    f"tensor '{name[: -len(INT8_SUFFIX)]}.scales' must be present and fp16"
                )
        if name.endswith(_FP16_ONLY_SUFFIXES) and code != "fp16":
            raise ValueError(f"tensor '{name}' must be stored as fp16, got {code}")
        if name.endswith(".qweight"):
            if code != "int32":
                raise ValueError(f"tensor '{name}' (Packed_Int4) must be int32, got {code}")
            base = name[: -len(".qweight")]
            scales = tensors.get(base + ".scales")
            if scales is None or _dtype_code(base + ".scales", scales) != "fp16":
                raise ValueError(f"tensor '{base}.scales' must be present and fp16")
            if not symmetric and (base + ".qzeros") not in tensors:
                raise ValueError(
                    f"tensor '{base}.qzeros' is required when the scheme is asymmetric"
                )


def _layout(header: QUFHeader, entries: list[TensorEntry]) -> tuple[bytes, list[TensorEntry]]:
    """Assign offsets. Offsets affect header length, so iterate to a fixed point."""
    offsets = [0] * len(entries)
    for _ in range(16):
        for e, off in zip(entries, offsets):
            e.offset = off
        header_bytes = QUFHeader(
            header.model_config, header.tokenizer, header.quantization, entries
        ).to_json_bytes()
        pos = align_up(PREFIX_SIZE + len(header_bytes))
        new_offsets = []
        for e in entries:
            new_offsets.append(pos)
            pos = align_up(pos + e.nbytes)
        if new_offsets == offsets:
            return header_bytes, entries
        offsets = new_offsets
    raise RuntimeError("header layout did not converge")  # pragma: no cover


def write_quf(path, header: QUFHeader, tensors: Mapping[str, np.ndarray]) -> QUFHeader:
    """Write a QUF file as version 1, 2 or 3, whichever the features used require
    (see ``required_version``). The Tensor_Table is built from ``tensors`` (in mapping order);
    any entries already in ``header.tensors`` are ignored. Returns the header as written.
    """
    _validate(header, tensors)

    arrays: list[np.ndarray] = []
    entries: list[TensorEntry] = []
    for name, arr in tensors.items():
        code = _dtype_code(name, arr)
        shape = arr.shape
        # ascontiguousarray promotes 0-d arrays to 1-d; restore the original shape.
        arr = np.ascontiguousarray(arr, dtype=np.dtype(DTYPES[code][0])).reshape(shape)
        arrays.append(arr)
        entries.append(TensorEntry(name=name, dtype=code, shape=list(arr.shape), offset=0))

    version = required_version(header.quantization, [(e.name, e.dtype) for e in entries])
    header_bytes, entries = _layout(header, entries)

    with open(os.fspath(path), "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        pos = PREFIX_SIZE + len(header_bytes)
        for entry, arr in zip(entries, arrays):
            assert entry.offset % ALIGN == 0 and entry.offset >= pos
            f.write(b"\x00" * (entry.offset - pos))
            f.write(arr.tobytes(order="C"))
            pos = entry.offset + arr.nbytes
        f.write(b"\x00" * (align_up(pos) - pos))

    return QUFHeader(header.model_config, header.tokenizer, header.quantization, entries)
