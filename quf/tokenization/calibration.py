"""Calibration_Builder: reproducible fixed-length token windows for calibration.

``build_calibration`` takes either a plain iterable of texts (one source) or a
list of :class:`CalibrationSource` objects (several sources, each with a share).

Share rounding rule (multi-source): source ``i`` first gets
``floor(n_samples * share_i)`` samples. The samples left over are given one each
to the sources with the largest fractional parts (ties go to the earlier
source). The counts always add up to ``n_samples``. Each source is shuffled with
``random.Random(seed + index)`` and sliced into non-overlapping windows. The
returned list is the windows of source 0, then source 1, and so on.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from quf.tokenization.wrapper import TokenizerWrapper


class CalibrationShortfallError(Exception):
    """Raised when a source has fewer tokens than it needs."""

    def __init__(self, needed: int, available: int, source: str | None = None):
        self.needed = needed
        self.available = available
        self.source = source
        self.shortfall = needed - available
        where = "" if source is None else f" from source '{source}'"
        super().__init__(
            f"Calibration data{where} is too small: need {needed} tokens, "
            f"have {available} (short by {self.shortfall})."
        )


@dataclass(frozen=True)
class CalibrationSource:
    """A named text source. ``loader()`` returns its documents; ``share`` is its
    fraction of the samples (shares of all sources must sum to 1)."""

    name: str
    loader: Callable[[], Iterable[str]]
    share: float


def split_samples(n_samples: int, shares: Sequence[float]) -> list[int]:
    """Split ``n_samples`` by ``shares`` with the rule in the module docstring."""
    counts = [math.floor(n_samples * s + 1e-9) for s in shares]
    rest = n_samples - sum(counts)
    # Rounding keeps float noise (2.4 - 2 != 0.4) from breaking exact ties.
    fracs = [(round(n_samples * s - c, 9), -i) for i, (s, c) in enumerate(zip(shares, counts))]
    for _, neg_i in sorted(fracs, reverse=True)[:rest]:
        counts[-neg_i] += 1
    return counts


def _windows(
    texts: Iterable[str],
    tok: TokenizerWrapper,
    n: int,
    seq_len: int,
    seed: int,
    name: str | None,
) -> list[list[int]]:
    if n == 0:
        return []
    docs = [tok.encode(t) for t in texts]
    random.Random(seed).shuffle(docs)

    needed = n * seq_len
    available = sum(len(d) for d in docs)
    if available < needed:
        raise CalibrationShortfallError(needed, available, name)

    flat: list[int] = []
    for d in docs:
        flat.extend(d)
        if len(flat) >= needed:
            break
    return [flat[i * seq_len : (i + 1) * seq_len] for i in range(n)]


def build_calibration(
    texts: Iterable[str] | Sequence[CalibrationSource],
    tok: TokenizerWrapper,
    n_samples: int,
    seq_len: int,
    seed: int,
) -> list[list[int]]:
    """Build ``n_samples`` non-overlapping windows of ``seq_len`` token ids.

    ``texts`` is an iterable of documents (single source, shuffled with
    ``random.Random(seed)``) or a non-empty list of ``CalibrationSource``. The
    same inputs always give the same output. Raises ``CalibrationShortfallError``
    (naming the source in the multi-source case) if there are too few tokens.
    """
    if n_samples < 1 or seq_len < 1:
        raise ValueError("n_samples and seq_len must be positive integers")

    if isinstance(texts, (list, tuple)) and texts and all(
        isinstance(s, CalibrationSource) for s in texts
    ):
        sources = list(texts)
        if any(s.share < 0 for s in sources) or abs(sum(s.share for s in sources) - 1.0) > 1e-6:
            raise ValueError("source shares must be non-negative and sum to 1")
        out: list[list[int]] = []
        for i, (src, n) in enumerate(zip(sources, split_samples(n_samples, [s.share for s in sources]))):
            if n:
                out.extend(_windows(src.loader(), tok, n, seq_len, seed + i, src.name))
        return out

    return _windows(texts, tok, n_samples, seq_len, seed, None)
