"""Validation and version selection shared by the Writer and the Reader."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from .constants import (
    EMBED_BITS,
    EMBED_GROUP_SIZES,
    EMBED_KEYS,
    EMBED_METHODS,
    EMBED_Q4_SUFFIX,
    EMBED_Q8_SUFFIX,
    EMBED_SCALES_SUFFIX,
    EMBED_VERSION,
    GROUP_SIZE,
    G_IDX_SUFFIX,
    INT8_SUFFIX,
    PRECISIONS,
    SUPPORTED_GROUP_SIZES,
)
from .errors import HeaderSchemaError


def check_group_size(value: Any) -> int:
    """Return ``value`` if it is a supported group size, else raise HeaderSchemaError."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in SUPPORTED_GROUP_SIZES
    ):
        raise HeaderSchemaError(
            f"unsupported group size {value!r}: supported values are "
            f"{', '.join(str(g) for g in SUPPORTED_GROUP_SIZES)}"
        )
    return value


def check_modules(modules: Any) -> dict[str, str]:
    """Validate the ``quantization.modules`` precision map."""
    if not isinstance(modules, dict):
        raise HeaderSchemaError("quantization.modules must be an object")
    for name, prec in modules.items():
        if not isinstance(prec, str) or prec not in PRECISIONS:
            raise HeaderSchemaError(
                f"quantization.modules['{name}']: invalid precision {prec!r}: "
                f"supported values are {', '.join(PRECISIONS)}"
            )
    return modules


def check_quantization(quantization: dict) -> int:
    """Validate group size and module precisions. Returns the effective group size.

    A missing ``group_size`` is read as 128 (version 1 files may omit it).
    """
    group_size = check_group_size(quantization.get("group_size", GROUP_SIZE))
    if "modules" in quantization:
        check_modules(quantization["modules"])
    act_order = quantization.get("act_order", False)
    if not isinstance(act_order, bool):
        raise HeaderSchemaError("quantization.act_order must be a boolean")
    return group_size


def check_g_idx(name: str, g_idx: np.ndarray, in_features: int, group_size: int) -> None:
    """Check one g_idx tensor: int32, length ``in_features``, values in [0, groups)."""
    if g_idx.dtype != np.dtype("<i4"):
        raise HeaderSchemaError(f"tensor '{name}': g_idx must be int32, got {g_idx.dtype}")
    if g_idx.ndim != 1 or g_idx.shape[0] != in_features:
        raise HeaderSchemaError(
            f"tensor '{name}': g_idx must have length {in_features} "
            f"(the input dimension), got shape {tuple(g_idx.shape)}"
        )
    n_groups = in_features // group_size
    if g_idx.size and (int(g_idx.min()) < 0 or int(g_idx.max()) >= n_groups):
        bad = int(g_idx.min()) if int(g_idx.min()) < 0 else int(g_idx.max())
        raise HeaderSchemaError(
            f"tensor '{name}': g_idx value {bad} is outside the range 0 to {n_groups - 1}"
        )


def check_tensor_group_layout(
    quantization: dict,
    shapes: dict[str, tuple[int, ...]],
    get_tensor: Callable[[str], np.ndarray],
) -> None:
    """Validate the group size and every g_idx tensor against its module's input dimension.

    ``shapes`` maps tensor name to shape; ``get_tensor`` returns the array for g_idx checks.
    """
    group_size = check_quantization(quantization)
    for name in shapes:
        if not name.endswith(G_IDX_SUFFIX):
            continue
        base = name[: -len(G_IDX_SUFFIX)]
        qshape = shapes.get(base + ".qweight")
        if qshape is None or len(qshape) != 2:
            raise HeaderSchemaError(
                f"tensor '{name}': g_idx has no matching '{base}.qweight' tensor"
            )
        in_features = int(qshape[0]) * 8
        if in_features % group_size != 0:
            raise HeaderSchemaError(
                f"tensor '{name}': input dimension {in_features} is not a multiple "
                f"of the group size {group_size}"
            )
        check_g_idx(name, get_tensor(name), in_features, group_size)


def has_quantized_embedding(quantization: dict, tensor_names: Iterable[str] = ()) -> bool:
    """True when the header declares, or the tensor table holds, a Quantized_Embedding."""
    if any(k in quantization for k in EMBED_KEYS):
        return True
    return any(
        n.endswith((EMBED_Q4_SUFFIX, EMBED_Q8_SUFFIX, EMBED_SCALES_SUFFIX)) for n in tensor_names
    )


def check_quantized_embeddings(
    quantization: dict,
    shapes: dict[str, tuple[int, ...]],
    dtypes: dict[str, str],
) -> None:
    """Validate every Quantized_Embedding declared in ``quantization`` (keys in EMBED_KEYS).

    Each spec is ``{"name", "bits", "group_size", "method"}``. The tensors ``<name>.embed_q4``
    (bits 4, int8 bytes ``[rows, cols/2]``) or ``<name>.embed_q8`` (bits 8, ``[rows, cols]``)
    and ``<name>.embed_scales`` (fp16 ``[rows, cols/group_size]``) must exist.
    """
    declared: set[str] = set()
    for key in EMBED_KEYS:
        if key not in quantization:
            continue
        spec = quantization[key]
        if not isinstance(spec, dict):
            raise HeaderSchemaError(f"quantization.{key} must be an object")
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise HeaderSchemaError(f"quantization.{key}.name must be a non-empty string")
        bits, gs, method = spec.get("bits"), spec.get("group_size"), spec.get("method", "rtn")
        if isinstance(bits, bool) or bits not in EMBED_BITS:
            raise HeaderSchemaError(
                f"quantization.{key}.bits {bits!r} is not supported: use 4 or 8"
            )
        if isinstance(gs, bool) or not isinstance(gs, int) or gs not in EMBED_GROUP_SIZES:
            raise HeaderSchemaError(
                f"quantization.{key}.group_size {gs!r} is not supported: use 32 or 64"
            )
        if method not in EMBED_METHODS:
            raise HeaderSchemaError(
                f"quantization.{key}.method {method!r} is not supported: "
                f"use {', '.join(EMBED_METHODS)}"
            )
        q_name = name + (EMBED_Q4_SUFFIX if bits == 4 else EMBED_Q8_SUFFIX)
        s_name = name + EMBED_SCALES_SUFFIX
        for n in (q_name, s_name):
            if n not in shapes:
                raise HeaderSchemaError(f"quantization.{key}: tensor '{n}' is missing")
        if dtypes[q_name] != "int8":
            raise HeaderSchemaError(f"tensor '{q_name}' must be int8, got {dtypes[q_name]}")
        if dtypes[s_name] != "fp16":
            raise HeaderSchemaError(f"tensor '{s_name}' must be fp16, got {dtypes[s_name]}")
        qs, ss = shapes[q_name], shapes[s_name]
        if len(qs) != 2 or qs[0] == 0 or qs[1] == 0:
            raise HeaderSchemaError(
                f"tensor '{q_name}': shape must be [rows, cols] with rows and cols > 0, "
                f"got {list(qs)}"
            )
        cols = qs[1] * 2 if bits == 4 else qs[1]
        if cols % gs != 0:
            raise HeaderSchemaError(
                f"tensor '{q_name}': hidden size {cols} is not a multiple of group size {gs}"
            )
        if tuple(ss) != (qs[0], cols // gs):
            raise HeaderSchemaError(
                f"tensor '{s_name}': shape must be {[qs[0], cols // gs]}, got {list(ss)}"
            )
        declared.update((q_name, s_name))
    for n in shapes:
        if n.endswith((EMBED_Q4_SUFFIX, EMBED_Q8_SUFFIX, EMBED_SCALES_SUFFIX)) and n not in declared:
            raise HeaderSchemaError(
                f"tensor '{n}' is a quantized embedding tensor with no matching "
                f"quantization.embedding or quantization.lm_head declaration"
            )


def required_version(quantization: dict, tensor_names_and_dtypes: Iterable[tuple[str, str]]) -> int:
    """Return 1 when only version 1 features are used, 2 for version 2 features, and 3
    when a Quantized_Embedding is present.

    Version 1 means: group size 128, no g_idx, no int8 tensor, and no int8 or fp16
    module entry. Everything else needs version 2.
    """
    tensor_names_and_dtypes = list(tensor_names_and_dtypes)
    if has_quantized_embedding(quantization, (n for n, _ in tensor_names_and_dtypes)):
        return EMBED_VERSION
    if quantization.get("group_size", GROUP_SIZE) != GROUP_SIZE:
        return 2
    modules = quantization.get("modules") or {}
    if any(p in ("int8", "fp16") for p in modules.values()):
        return 2
    for name, dtype in tensor_names_and_dtypes:
        if dtype == "int8" or name.endswith(G_IDX_SUFFIX) or name.endswith(INT8_SUFFIX):
            return 2
    return 1
