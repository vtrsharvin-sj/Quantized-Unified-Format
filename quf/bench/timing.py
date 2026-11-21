"""Stable throughput timing: warm-up, N repeats, median with min and max (Requirement 28.1).

``median`` and ``summarize`` are pure functions of the per-repeat values, so the result
does not depend on the order of the samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 7
MIN_REPEATS = 5


def median(samples: Sequence[float]) -> float:
    """Median of ``samples`` (mean of the two middle values for an even count)."""
    if len(samples) == 0:
        raise ValueError("median of an empty sample list")
    s = sorted(samples)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


@dataclass(frozen=True)
class TimingStats:
    median: float
    min: float
    max: float
    repeats: int
    samples: tuple[float, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["samples"] = list(self.samples)
        return d


def summarize(samples: Sequence[float]) -> TimingStats:
    """Median, min, and max of per-repeat values (tokens per second)."""
    return TimingStats(
        median=median(samples),
        min=float(min(samples)),
        max=float(max(samples)),
        repeats=len(samples),
        samples=tuple(float(x) for x in samples),
    )


def time_repeats(
    run_once: Callable[[], float],
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> TimingStats:
    """Call ``run_once`` ``warmup`` times (discarded), then ``repeats`` times and summarize.

    ``run_once`` returns one tokens-per-second value. ``repeats`` must be at least 5.
    """
    if repeats < MIN_REPEATS:
        raise ValueError(f"repeats must be at least {MIN_REPEATS}, got {repeats}")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    for _ in range(warmup):
        run_once()
    return summarize([run_once() for _ in range(repeats)])
