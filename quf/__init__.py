"""QUF: Quantized Unified Format.

Importing this package never requires optional extras (torch, gptqmodel, ...).
Submodules, ``quf.load``, and ``quf.Model`` are imported lazily on first attribute access.

Example::

    import quf

    with quf.load("model.quf") as model:
        print(model.generate("The capital of France is", max_new_tokens=24))
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.4.1"

_LAZY_SUBMODULES = frozenset(
    {"api", "device", "format", "quant", "tokenization", "convert", "runtime", "bench", "cli"}
)
# Public names that live in a submodule, resolved on first access: name -> submodule.
_LAZY_ATTRS = {"load": "api", "Model": "api"}

__all__ = ["__version__", *sorted(_LAZY_ATTRS), *sorted(_LAZY_SUBMODULES)]


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        return getattr(importlib.import_module(f"{__name__}.{_LAZY_ATTRS[name]}"), name)
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES | set(_LAZY_ATTRS))
