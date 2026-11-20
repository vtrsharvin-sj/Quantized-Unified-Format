"""Header schema for QUF (versions 1 and 2) and its JSON (de)serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .constants import DTYPES, GROUP_SIZE, REQUIRED_TENSOR_FIELDS, REQUIRED_TOP_LEVEL_FIELDS
from .errors import (  # noqa: F401  (re-exported)
    HeaderSchemaError,
    InvalidMagicError,
    QUFFormatError,
    TensorBoundsError,
    TruncatedHeaderError,
    UnsupportedVersionError,
)


@dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: list[int]
    offset: int

    @property
    def nbytes(self) -> int:
        """Byte size computed from dtype and shape."""
        if self.dtype not in DTYPES:
            raise HeaderSchemaError(f"tensor '{self.name}': unknown dtype '{self.dtype}'")
        n = 1
        for d in self.shape:
            n *= int(d)
        return n * DTYPES[self.dtype][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": [int(d) for d in self.shape],
            "offset": int(self.offset),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "TensorEntry":
        if not isinstance(d, dict):
            raise HeaderSchemaError("tensor table entry is not an object")
        for key in REQUIRED_TENSOR_FIELDS:
            if key not in d:
                label = d.get("name", "<unnamed>")
                raise HeaderSchemaError(
                    f"tensor table entry '{label}' is missing required field '{key}'"
                )
        name, dtype, shape, offset = d["name"], d["dtype"], d["shape"], d["offset"]
        if not isinstance(name, str):
            raise HeaderSchemaError("tensor name must be a string")
        if dtype not in DTYPES:
            raise HeaderSchemaError(f"tensor '{name}': unknown dtype '{dtype}'")
        if not isinstance(shape, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in shape
        ):
            raise HeaderSchemaError(f"tensor '{name}': shape must be a list of non-negative integers")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise HeaderSchemaError(f"tensor '{name}': offset must be a non-negative integer")
        return cls(name=name, dtype=dtype, shape=list(shape), offset=offset)


@dataclass
class QUFHeader:
    model_config: dict
    tokenizer: dict
    quantization: dict
    tensors: list[TensorEntry] = field(default_factory=list)

    @property
    def group_size(self) -> int:
        """Group size; a header that omits it is read as 128 (version 1 files)."""
        return self.quantization.get("group_size", GROUP_SIZE)

    @property
    def act_order(self) -> bool:
        return bool(self.quantization.get("act_order", False))

    @property
    def modules(self) -> dict[str, str]:
        """Per-module precision map (``int4``, ``int8``, ``fp16``); empty when absent."""
        return self.quantization.get("modules") or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "tokenizer": self.tokenizer,
            "quantization": self.quantization,
            "tensors": [t.to_dict() for t in self.tensors],
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_dict(cls, d: Any) -> "QUFHeader":
        if not isinstance(d, dict):
            raise HeaderSchemaError("header JSON is not an object")
        for key in REQUIRED_TOP_LEVEL_FIELDS:
            if key not in d:
                raise HeaderSchemaError(f"header is missing required field '{key}'")
        for key in ("model_config", "tokenizer", "quantization"):
            if not isinstance(d[key], dict):
                raise HeaderSchemaError(f"header field '{key}' must be an object")
        if not isinstance(d["tensors"], list):
            raise HeaderSchemaError("header field 'tensors' must be a list")
        return cls(
            model_config=d["model_config"],
            tokenizer=d["tokenizer"],
            quantization=d["quantization"],
            tensors=[TensorEntry.from_dict(t) for t in d["tensors"]],
        )

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "QUFHeader":
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise HeaderSchemaError(f"header is not valid JSON: {e}") from e
        return cls.from_dict(parsed)
