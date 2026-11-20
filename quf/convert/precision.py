"""Mixed precision: ``PrecisionPlan`` (which module gets which precision) and
``SizeReport`` (bytes by precision class, computed from the Tensor_Table).

Module names are the Hugging Face names without ``.weight``, for example
``model.layers.3.mlp.down_proj`` or ``lm_head``. Patterns are shell globs
(``fnmatch``), where ``*`` also matches dots, so ``*.v_proj`` selects every
``v_proj``. When several rules match a module, the last rule wins.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from quf.format.constants import INT8_SUFFIX, PRECISIONS

DEFAULT_PRECISION = "int4"

# Presets chosen from the diagnosis in docs/ACCURACY.md. ``{last}`` is the last layer index.
PRESETS: dict[str, tuple[str, ...]] = {
    "down_proj": ("*.down_proj=int8",),
    "sensitive": ("*.down_proj=int8", "*.layers.{last}.*=int8"),
}


class PrecisionRuleError(ValueError):
    """Raised for a malformed precision rule or an unknown preset."""


def parse_rule(text: str) -> tuple[str, str]:
    """Parse ``"pattern=precision"`` into ``(pattern, precision)``."""
    pattern, sep, prec = text.rpartition("=")
    pattern, prec = pattern.strip(), prec.strip().lower()
    if not sep or not pattern:
        raise PrecisionRuleError(f"invalid precision rule {text!r}: expected 'pattern=precision'")
    if prec not in PRECISIONS:
        raise PrecisionRuleError(
            f"invalid precision {prec!r} in rule {text!r}: supported values are "
            f"{', '.join(PRECISIONS)}"
        )
    return pattern, prec


@dataclass
class PrecisionPlan:
    """Ordered ``(glob, precision)`` rules; modules matching no rule are ``int4``."""

    rules: list[tuple[str, str]] = field(default_factory=list)
    default: str = DEFAULT_PRECISION

    @classmethod
    def from_options(
        cls,
        mixed: Iterable[str] = (),
        *,
        presets: Iterable[str] = (),
        int8_modules: Iterable[str] = (),
        num_layers: int | None = None,
    ) -> "PrecisionPlan":
        """Build a plan. Order of precedence (later wins): presets, ``int8_modules``, ``mixed``.

        Each ``mixed`` item is ``"pattern=precision"``; items may hold several rules
        separated by commas.
        """
        rules: list[tuple[str, str]] = []
        for name in presets:
            if name not in PRESETS:
                raise PrecisionRuleError(
                    f"unknown preset {name!r}: available presets are {', '.join(sorted(PRESETS))}"
                )
            for r in PRESETS[name]:
                if "{last}" in r:
                    if not num_layers:
                        raise PrecisionRuleError(f"preset {name!r} needs the number of layers")
                    r = r.replace("{last}", str(num_layers - 1))
                rules.append(parse_rule(r))
        rules += [(p.strip(), "int8") for p in int8_modules if p.strip()]
        for item in mixed:
            for part in item.split(","):
                if part.strip():
                    rules.append(parse_rule(part))
        return cls(rules)

    def precision(self, module: str) -> str:
        """Precision of ``module``: the last matching rule, else the default."""
        result = self.default
        for pattern, prec in self.rules:
            if fnmatch.fnmatchcase(module, pattern):
                result = prec
        return result

    def explicit(self, module: str) -> str | None:
        """Precision from a matching rule, or ``None`` if no rule matches."""
        found = None
        for pattern, prec in self.rules:
            if fnmatch.fnmatchcase(module, pattern):
                found = prec
        return found

    def __bool__(self) -> bool:
        return bool(self.rules)

    def modules_map(self, linear_names: Iterable[str]) -> dict[str, str]:
        """Precision of each linear module name (for the header ``modules`` field)."""
        return {n: self.precision(n) for n in linear_names}


# ---------------------------------------------------------------------------
# Size accounting
# ---------------------------------------------------------------------------

_INT4_SUFFIXES = (".qweight", ".qzeros", ".g_idx", ".scales")


@dataclass
class SizeReport:
    """Tensor bytes by precision class, plus the fp16-equivalent size.

    ``int4``, ``int8`` and ``fp16`` sum the Tensor_Table byte sizes, so
    ``int4 + int8 + fp16 == tensor_bytes``. ``scales``, ``qzeros`` and ``g_idx`` count
    toward the class of their module. ``file_bytes`` includes header and padding.
    """

    int4: int = 0
    int8: int = 0
    fp16: int = 0
    fp16_equivalent: int = 0
    file_bytes: int | None = None
    groups: dict[str, dict[str, int]] = field(default_factory=dict)  # precision -> {label: count}

    @property
    def tensor_bytes(self) -> int:
        return self.int4 + self.int8 + self.fp16

    @property
    def total(self) -> int:
        return self.file_bytes if self.file_bytes is not None else self.tensor_bytes

    @property
    def ratio(self) -> float:
        return self.total / self.fp16_equivalent if self.fp16_equivalent else 0.0

    @classmethod
    def from_tensors(
        cls,
        tensors: Iterable[tuple[str, str, Sequence[int]]],
        modules: Mapping[str, str] | None = None,
        file_bytes: int | None = None,
    ) -> "SizeReport":
        """Build a report from ``(name, dtype, shape)`` triples (the Tensor_Table)."""
        items = [(n, d, tuple(int(x) for x in s)) for n, d, s in tensors]
        names = {n for n, _, _ in items}
        int4_mods = {n[: -len(".qweight")] for n in names if n.endswith(".qweight")}
        int8_mods = {n[: -len(INT8_SUFFIX)] for n in names if n.endswith(INT8_SUFFIX)}
        fp16_mods = {m for m, p in (modules or {}).items() if p == "fp16"}
        rep = cls(file_bytes=file_bytes)
        for name, dtype, shape in items:
            n_el = 1
            for d in shape:
                n_el *= d
            nbytes = n_el * ({"fp16": 2, "fp32": 4, "int32": 4, "int8": 1}[dtype])
            base = name.rsplit(".", 1)[0]
            if name.endswith(INT8_SUFFIX) or (name.endswith(".scales") and base in int8_mods):
                rep.int8 += nbytes
            elif name.endswith(_INT4_SUFFIXES) and base in int4_mods:
                rep.int4 += nbytes
            else:
                rep.fp16 += nbytes
            # logical fp16 size of the weights this tensor stands for
            if name.endswith(".qweight") and base in int4_mods:
                rep.fp16_equivalent += 2 * n_el * 8
            elif name.endswith(INT8_SUFFIX):
                rep.fp16_equivalent += 2 * n_el
            elif name.endswith((".scales", ".qzeros", ".g_idx")) and base in (int4_mods | int8_mods):
                pass
            else:
                rep.fp16_equivalent += 2 * n_el
        for prec, mods in (("int4", int4_mods), ("int8", int8_mods), ("fp16", fp16_mods)):
            counts: dict[str, int] = {}
            for m in sorted(mods):
                label = m.rsplit(".", 1)[-1]
                counts[label] = counts.get(label, 0) + 1
            if counts:
                rep.groups[prec] = counts
        return rep

    @classmethod
    def from_header(cls, header, file_bytes: int | None = None) -> "SizeReport":
        return cls.from_tensors(
            [(t.name, t.dtype, t.shape) for t in header.tensors],
            header.modules,
            file_bytes,
        )

    def format(self) -> str:
        lines = ["Size by precision:"]
        for prec, nbytes in (("int4", self.int4), ("int8", self.int8), ("fp16", self.fp16)):
            groups = self.groups.get(prec)
            detail = ""
            if groups:
                n = sum(groups.values())
                detail = f" ({n} modules: " + ", ".join(f"{k} x{v}" for k, v in groups.items()) + ")"
            elif prec == "fp16":
                detail = " (embeddings, norms, biases and fp16 modules)"
            lines.append(f"  {prec}: {nbytes} bytes{detail}")
        lines.append(f"  tensor total: {self.tensor_bytes} bytes")
        if self.file_bytes is not None:
            lines.append(f"  file total: {self.file_bytes} bytes (tensors plus header and padding)")
        if self.fp16_equivalent:
            lines.append(
                f"  ratio to fp16 checkpoint: {self.ratio:.4f} ({self.fp16_equivalent} bytes)"
            )
        return "\n".join(lines)

