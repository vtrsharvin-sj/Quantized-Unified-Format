"""Canonical Packed_Int4 packing: 8 four-bit values per int32 along the last axis.

Value ``i`` of each group of 8 occupies bits ``4 * (i % 8)`` (least significant first).
"""

from __future__ import annotations

import numpy as np

_PER_WORD = 8
_SHIFTS = (np.arange(_PER_WORD, dtype=np.uint32) * 4).astype(np.uint32)


def pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack 4-bit values (0..15) along the last axis into int32.

    ``[..., N]`` with ``N % 8 == 0`` becomes ``int32 [..., N // 8]``.
    """
    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError("values must have at least one dimension")
    n = arr.shape[-1]
    if n % _PER_WORD != 0:
        raise ValueError(f"last dimension {n} is not a multiple of {_PER_WORD}")
    if arr.size and (arr.min() < 0 or arr.max() > 15):
        raise ValueError("values must be in the range 0..15")
    grouped = arr.astype(np.uint32).reshape(*arr.shape[:-1], n // _PER_WORD, _PER_WORD)
    words = np.bitwise_or.reduce(grouped << _SHIFTS, axis=-1).astype(np.uint32)
    return words.view(np.int32)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_int4`: ``int32 [..., M]`` becomes ``uint8 [..., M * 8]``."""
    arr = np.asarray(packed)
    if arr.ndim == 0:
        raise ValueError("packed must have at least one dimension")
    if arr.dtype != np.int32:
        raise ValueError(f"packed must have dtype int32, got {arr.dtype}")
    words = np.ascontiguousarray(arr).view(np.uint32)
    nibbles = (words[..., None] >> _SHIFTS) & np.uint32(0xF)
    return nibbles.reshape(*arr.shape[:-1], arr.shape[-1] * _PER_WORD).astype(np.uint8)
