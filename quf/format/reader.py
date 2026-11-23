"""QUF reader (versions 1, 2 and 3): validates a file and exposes zero-copy memory-mapped tensors."""

from __future__ import annotations

import mmap
import os
import struct
from typing import BinaryIO

import numpy as np

from .constants import ALIGN, DTYPES, MAGIC, MAX_VERSION, PREFIX_SIZE
from .errors import (  # noqa: F401  (re-exported for convenience)
    HeaderSchemaError,
    InvalidMagicError,
    QUFFormatError,
    TensorBoundsError,
    TruncatedHeaderError,
    UnsupportedVersionError,
)
from .schema import QUFHeader, TensorEntry
from .validate import check_quantized_embeddings, check_tensor_group_layout, has_quantized_embedding


class QUFReader:
    """Open a QUF file read-only, validate it, and expose its header and tensors.

    Validation order: magic, version, header length, JSON parse, required
    fields, then per-tensor alignment and bounds.
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self._file: BinaryIO | None = open(self.path, "rb")
        self._mm: mmap.mmap | None = None
        try:
            self.file_size = os.fstat(self._file.fileno()).st_size
            self.header = self._validate()
        except BaseException:
            self.close()
            raise
        self._entries = {t.name: t for t in self.header.tensors}

    # -- validation -------------------------------------------------------

    def _validate(self) -> QUFHeader:
        f = self._file
        assert f is not None
        size = self.file_size

        # 1. magic
        magic = f.read(4)
        if magic != MAGIC:
            raise InvalidMagicError(magic)

        # 2. version
        if size < 8:
            raise TruncatedHeaderError("truncated header: file ends before the version field")
        (version,) = struct.unpack("<I", f.read(4))
        if version > MAX_VERSION:
            raise UnsupportedVersionError(version, MAX_VERSION)

        # 3. header length
        if size < PREFIX_SIZE:
            raise TruncatedHeaderError(
                "truncated header: file ends before the header length field"
            )
        (header_len,) = struct.unpack("<Q", f.read(8))
        if PREFIX_SIZE + header_len > size:
            raise TruncatedHeaderError(
                f"truncated header: header length {header_len} extends past the end "
                f"of the file ({size} bytes)"
            )

        # 4-5. JSON parse and required fields
        header = QUFHeader.from_json_bytes(f.read(header_len))

        # 6. per-tensor alignment and bounds
        seen: set[str] = set()
        for t in header.tensors:
            if t.name in seen:
                raise HeaderSchemaError(f"duplicate tensor name '{t.name}'")
            seen.add(t.name)
            if t.offset % ALIGN != 0:
                raise TensorBoundsError(
                    t.name, f"offset {t.offset} is not a multiple of {ALIGN}"
                )
            if t.offset + t.nbytes > size:
                raise TensorBoundsError(
                    t.name,
                    f"offset {t.offset} plus size {t.nbytes} extends past the end of "
                    f"the file ({size} bytes)",
                )

        if size > 0:
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # 7. group size, module precisions, and g_idx (version 2 features).
        # A missing group size is read as 128.
        self.version = version
        self._entries = {t.name: t for t in header.tensors}
        check_tensor_group_layout(
            header.quantization,
            {t.name: tuple(t.shape) for t in header.tensors},
            self.tensor,
        )
        # 8. version 3 field: Quantized_Embedding
        check_quantized_embeddings(
            header.quantization,
            {t.name: tuple(t.shape) for t in header.tensors},
            {t.name: t.dtype for t in header.tensors},
        )
        if version < 3 and has_quantized_embedding(
            header.quantization, (t.name for t in header.tensors)
        ):
            raise HeaderSchemaError(
                f"a quantized embedding requires format version 3, file version is {version}"
            )
        return header

    # -- access -----------------------------------------------------------

    def tensor_names(self) -> list[str]:
        return [t.name for t in self.header.tensors]

    def tensor(self, name: str) -> np.ndarray:
        """Return a zero-copy, read-only view of tensor ``name``."""
        if self._mm is None:
            raise ValueError("reader is closed")
        try:
            entry: TensorEntry = self._entries[name]
        except KeyError:
            raise KeyError(f"no tensor named '{name}' in {self.path}") from None
        np_dtype = np.dtype(DTYPES[entry.dtype][0])
        count = 1
        for d in entry.shape:
            count *= d
        arr = np.frombuffer(self._mm, dtype=np_dtype, count=count, offset=entry.offset)
        return arr.reshape(entry.shape)

    def close(self) -> None:
        """Release the mapping and file. Views returned earlier become invalid."""
        mm, self._mm = self._mm, None
        f, self._file = self._file, None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                # Live numpy views still reference the map; the OS releases it
                # when they are garbage collected.
                pass
        if f is not None:
            f.close()

    def __enter__(self) -> "QUFReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
