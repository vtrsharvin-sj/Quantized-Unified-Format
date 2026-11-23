"""QUF: Quantized Unified Format.

Importing this package never requires optional extras (torch, gptqmodel, ...).
Submodules are imported lazily on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.4.0"

_LAZY_SUBMODULES = frozenset(
    {"device", "format", "quant", "tokenization", "convert", "runtime", "bench", "cli"}
)

__all__ = ["__version__", *sorted(_LAZY_SUBMODULES)]


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)
