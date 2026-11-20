"""Validation and version selection shared by the Writer and the Reader."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from .constants import (
    GROUP_SIZE,
    G_IDX_SUFFIX,
    INT8_SUFFIX,
    MAX_VERSION,
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


def required_version(quantization: dict, tensor_names_and_dtypes: Iterable[tuple[str, str]]) -> int:
    """Return 1 when only version 1 features are used, else 2.

    Version 1 means: group size 128, no g_idx, no int8 tensor, and no int8 or fp16
    module entry. Everything else needs version 2.
    """
    if quantization.get("group_size", GROUP_SIZE) != GROUP_SIZE:
        return MAX_VERSION
    modules = quantization.get("modules") or {}
    if any(p in ("int8", "fp16") for p in modules.values()):
        return MAX_VERSION
    for name, dtype in tensor_names_and_dtypes:
        if dtype == "int8" or name.endswith(G_IDX_SUFFIX) or name.endswith(INT8_SUFFIX):
            return MAX_VERSION
    return 1
