"""Runtime errors shared by the loader and the CLI."""

from __future__ import annotations


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024 ** 2):.1f} MiB"


class GPUMemoryError(RuntimeError):
    """Raised when the GPU runs out of memory while loading a model."""

    def __init__(self, required: int, available: int):
        self.required = int(required)
        self.available = int(available)
        super().__init__(
            f"GPU out of memory: {_fmt_bytes(self.required)} required "
            f"({self.required} bytes), {_fmt_bytes(self.available)} available "
            f"({self.available} bytes)"
        )
