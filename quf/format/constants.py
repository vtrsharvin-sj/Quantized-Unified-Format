"""Constants for the QUF file format (versions 1 and 2). All integers are little-endian."""

MAGIC = b"QUF1"
VERSION = 1  # version written for files that use only version 1 features
MAX_VERSION = 2  # highest version this package reads and writes
ALIGN = 64
GROUP_SIZE = 128  # default group size, and the only one version 1 files use
SUPPORTED_GROUP_SIZES = (32, 64, 128)

# Per-module precision strings allowed in ``quantization.modules``
PRECISIONS = ("int4", "int8", "fp16")

# Tensor name suffixes introduced by version 2
G_IDX_SUFFIX = ".g_idx"
INT8_SUFFIX = ".qweight_int8"

# Fixed prefix: magic (4) + version uint32 (4) + header_len uint64 (8)
PREFIX_SIZE = 16

# dtype code -> (numpy dtype string, bytes per element)
DTYPES = {
    "fp16": ("<f2", 2),
    "fp32": ("<f4", 4),
    "int32": ("<i4", 4),
    "int8": ("i1", 1),
}

REQUIRED_TOP_LEVEL_FIELDS = ("model_config", "tokenizer", "quantization", "tensors")
REQUIRED_TENSOR_FIELDS = ("name", "dtype", "shape", "offset")


def align_up(value: int, align: int = ALIGN) -> int:
    """Round ``value`` up to the next multiple of ``align``."""
    return (value + align - 1) // align * align
