"""Draw the README chart docs/assets/quf_vs_gguf.png from docs/bench_accuracy.json.

The chart has one panel per model with rows in ``pareto_040`` or ``scaling_3b``.
Each panel plots the file size in MB (10^6 bytes) against the WikiText-2
perplexity under the llama protocol. Every measured format is a labelled point,
QUF 0.4.0 has its own colour and marker, and the fp16 perplexity from
``models_llama`` is a dashed reference line where it was measured. A panel
without an fp16 measurement says "fp16 not measured". No number is written in
this file: every value comes from the JSON.

matplotlib is a development-only tool for this script. It is deliberately not
declared in pyproject.toml (no dependency, no extra). Install it into the
current Python with::

    python -m pip install matplotlib

Usage::

    python scripts/make_readme_chart.py           # write docs/assets/quf_vs_gguf.png
    python scripts/make_readme_chart.py --check   # validate the JSON only, no matplotlib

Exit status: 0 on success, 1 when matplotlib is missing, 2 when the JSON cannot
be read or lacks a key that the chart needs. The PNG is 1100 x 600 pixels at
150 dpi and is byte-for-byte reproducible for a given matplotlib version.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = ROOT / "docs" / "bench_accuracy.json"
DEFAULT_OUT = ROOT / "docs" / "assets" / "quf_vs_gguf.png"
MISSING_MATPLOTLIB = "matplotlib is required: pip install matplotlib"

QUF_ROW = "QUF"
# Bench_JSON row names that the chart renames (design, Data Models); other names stay.
ROW_LABELS = {
    "QUF": "QUF 0.4.0",
    "Q4_K_M-imat": "Q4_K_M imatrix",
    "IQ4_XS": "IQ4_XS imatrix",  # the pareto_040 IQ4_XS files were built with an imatrix
    "IQ4_XS-imat": "IQ4_XS imatrix",
}

FIG_WIDTH_PX = 1100
FIG_HEIGHT_PX = 600
DPI = 150

QUF_COLOR = "#E8590C"
QUF_EDGE = "#8F3300"
QUF_TEXT = "#B23F05"
GGUF_COLOR = "#3D6FB6"
FP16_COLOR = "#6B6B6B"
TEXT_COLOR = "#222222"
QUF_MARKERSIZE = 14.0
GGUF_MARKERSIZE = 6.5
FP16_DASHES = (0, (4, 3))

# Label positions: (dx, dy) in units of the marker gap, then the text alignment.
POSITIONS = {
    "right": (1.0, 0.0, "left", "center"),
    "left": (-1.0, 0.0, "right", "center"),
    "above": (0.0, 1.0, "center", "bottom"),
    "below": (0.0, -1.0, "center", "top"),
    "upper right": (0.7, 0.7, "left", "bottom"),
    "lower right": (0.7, -0.7, "left", "top"),
    "upper left": (-0.7, 0.7, "right", "bottom"),
    "lower left": (-0.7, -0.7, "right", "top"),
}
# Manual preferences, tried before the order above. Q4_K_M imatrix has almost the
# file size of Q4_K_M and a lower perplexity, so the Q4_K_M label goes above its
# point and the imatrix label below its point. Q5_K_M is the largest file, so its
# label sits above or below its point instead of next to the right edge.
PREFERRED_POSITION = {
    "Q4_K_M": ("above",),
    "Q4_K_M imatrix": ("below",),
    "Q5_K_M": ("above", "below"),
}
# Corners (axes fractions) tried in order for the "fp16 not measured" note.
NOTE_CORNERS = (
    (0.97, 0.96, "right", "top"),
    (0.03, 0.04, "left", "bottom"),
    (0.03, 0.96, "left", "top"),
    (0.97, 0.04, "right", "bottom"),
)


@dataclass(frozen=True)
class Point:
    label: str            # "QUF 0.4.0", "Q4_K_M", "Q4_K_M imatrix", "Q4_K_S", ...
    size_mb: float        # file_size_bytes / 1e6
    ppl_wikitext2: float  # llama protocol
    is_quf: bool


@dataclass(frozen=True)
class Panel:
    model: str               # "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"
    points: tuple[Point, ...]
    fp16_ppl: float | None   # None draws "fp16 not measured"


# ---------------------------------------------------------------------------
# Data (pure, no matplotlib)
# ---------------------------------------------------------------------------

def _number(value) -> float | None:
    """A finite positive number, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _is_row(row) -> bool:
    return isinstance(row, dict) and ("file_size_bytes" in row or "ppl_wikitext2" in row)


def _point(row_name: str, row) -> Point | None:
    """The chart point of one Bench_JSON row, or None without a size and a WikiText-2 value."""
    if not isinstance(row, dict):
        return None
    size = _number(row.get("file_size_bytes"))
    ppl = _number(row.get("ppl_wikitext2"))
    if size is None or ppl is None:
        return None
    return Point(ROW_LABELS.get(row_name, row_name), size / 1e6, ppl, row_name == QUF_ROW)


def _model_id(bench: dict, size_key: str) -> str | None:
    """The models_llama id of a pareto_040 key, e.g. "0.5B" -> "Qwen/Qwen2.5-0.5B"."""
    models = bench.get("models_llama")
    if isinstance(models, dict):
        for model_id in models:
            if isinstance(model_id, str) and model_id.endswith("-" + size_key):
                return model_id
    return None


def _sections(bench: dict):
    """Yield (json_path, model_id, panel_name, rows) for each pareto_040 model and scaling_3b."""
    pareto = bench.get("pareto_040")
    if isinstance(pareto, dict):
        for key, rows in pareto.items():
            if isinstance(rows, dict):
                model_id = _model_id(bench, key)
                name = model_id.split("/")[-1] if model_id else key
                yield f"pareto_040.{key}", model_id, name, rows
    scaling = bench.get("scaling_3b")
    if isinstance(scaling, dict):
        env = scaling.get("env")
        model_id = env.get("model") if isinstance(env, dict) else None
        if isinstance(model_id, str) and model_id:
            yield "scaling_3b", model_id, model_id.split("/")[-1], scaling


def _fp16_ppl(bench: dict, model_id: str | None) -> float | None:
    """The fp16 WikiText-2 perplexity (llama protocol) from models_llama, or None."""
    models = bench.get("models_llama")
    entry = models.get(model_id) if isinstance(models, dict) and model_id else None
    fp16 = entry.get("fp16") if isinstance(entry, dict) else None
    if not isinstance(fp16, dict):
        return None
    if fp16.get("status", "done") != "done" or fp16.get("protocol", "llama") != "llama":
        return None
    return _number(fp16.get("ppl_wikitext2"))


def collect_points(bench: dict) -> list[Panel]:
    """One Panel per model that has rows with a size and a WikiText-2 value, in JSON order."""
    if not isinstance(bench, dict):
        return []
    points: dict[str, list[Point]] = {}
    fp16: dict[str, float | None] = {}
    for _, model_id, name, rows in _sections(bench):
        for row_name, row in rows.items():
            point = _point(row_name, row)
            if point is None:
                continue
            panel_points = points.setdefault(name, [])
            fp16.setdefault(name, _fp16_ppl(bench, model_id))
            if all(p.label != point.label for p in panel_points):
                panel_points.append(point)
    return [Panel(name, tuple(pts), fp16[name]) for name, pts in points.items()]


def skipped_rows(bench: dict) -> list[str]:
    """Rows that name a size or a WikiText-2 value but lack a usable pair of them."""
    if not isinstance(bench, dict):
        return []
    return [f"{path}.{row_name}"
            for path, _, _, rows in _sections(bench)
            for row_name, row in rows.items()
            if _is_row(row) and _point(row_name, row) is None]


def check_bench(bench) -> list[str]:
    """Problems that stop the chart from being drawn; an empty list means the JSON is usable."""
    if not isinstance(bench, dict):
        return ["the file does not hold a JSON object"]
    problems = []
    env = bench.get("environment")
    device = env.get("device") if isinstance(env, dict) else None
    if not isinstance(device, str) or not device.strip():
        problems.append("environment.device (the GPU named in the caption) is missing")
    pareto = bench.get("pareto_040")
    if not isinstance(pareto, dict) or not pareto:
        problems.append("pareto_040 is missing or empty")
    else:
        for key, rows in pareto.items():
            rows = rows if isinstance(rows, dict) else {}
            if _point(QUF_ROW, rows.get(QUF_ROW)) is None:
                problems.append(f"pareto_040.{key}.QUF needs file_size_bytes and ppl_wikitext2")
            if not any(_point(n, r) is not None for n, r in rows.items() if n != QUF_ROW):
                problems.append(f"pareto_040.{key} has no GGUF row with file_size_bytes and ppl_wikitext2")
    if not isinstance(bench.get("models_llama"), dict):
        problems.append("models_llama (the fp16 reference) is missing")
    scaling = bench.get("scaling_3b")
    if scaling is not None:
        if not isinstance(scaling, dict):
            problems.append("scaling_3b is not an object")
        elif any(_point(n, r) is not None for n, r in scaling.items()):
            env3 = scaling.get("env")
            if not isinstance(env3, dict) or not isinstance(env3.get("model"), str) or not env3["model"]:
                problems.append("scaling_3b.env.model (the panel title) is missing")
    if not problems and not collect_points(bench):
        problems.append("no model has a row with file_size_bytes and ppl_wikitext2")
    return problems


def describe(panels: list[Panel]) -> list[str]:
    """One summary line per panel: the plotted rows and the fp16 reference."""
    lines = []
    for panel in panels:
        fp16 = "fp16 not measured" if panel.fp16_ppl is None else f"fp16 {panel.fp16_ppl:.3f}"
        rows = ", ".join(f"{p.label} {p.size_mb:.1f} MB {p.ppl_wikitext2:.3f}" for p in panel.points)
        lines.append(f"{panel.model}: {len(panel.points)} rows ({rows}); {fp16}")
    return lines


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


# ---------------------------------------------------------------------------
# Drawing (matplotlib, Agg backend)
# ---------------------------------------------------------------------------

def _padded(lo: float, hi: float, below: float, above: float) -> tuple[float, float]:
    span = hi - lo
    if span <= 0:
        span = abs(hi) * 0.05 or 1.0
    return lo - span * below, hi + span * above


def _pad(box, pixels: float):
    from matplotlib.transforms import Bbox
    return Bbox.from_extents(box.x0 - pixels, box.y0 - pixels, box.x1 + pixels, box.y1 + pixels)


def _inside(box, outer) -> bool:
    return box.x0 >= outer.x0 and box.y0 >= outer.y0 and box.x1 <= outer.x1 and box.y1 <= outer.y1


def _place_note(ax, renderer, obstacles):
    """Write "fp16 not measured" in the first free corner; return its box."""
    style = {"transform": ax.transAxes, "fontsize": 7, "style": "italic", "color": FP16_COLOR}
    for x, y, ha, va in NOTE_CORNERS:
        note = ax.text(x, y, "fp16 not measured", ha=ha, va=va, **style)
        box = _pad(note.get_window_extent(renderer), 2)
        if not any(box.overlaps(o) for o in obstacles):
            return box
        note.remove()
    x, y, ha, va = NOTE_CORNERS[0]
    return _pad(ax.text(x, y, "fp16 not measured", ha=ha, va=va, **style).get_window_extent(renderer), 2)


def _place_labels(ax, renderer, panel: Panel, obstacles: list) -> list[str]:
    """Label every point at the first position that stays inside the axes and overlaps nothing.

    QUF is placed first. Returns the labels that had no free position.
    """
    ax_box = ax.get_window_extent(renderer)
    crowded = []
    for p in sorted(panel.points, key=lambda q: not q.is_quf):  # stable: QUF first, then JSON order
        names = list(PREFERRED_POSITION.get(p.label, ()))
        names += [n for n in POSITIONS if n not in names]
        gap = (QUF_MARKERSIZE if p.is_quf else GGUF_MARKERSIZE) / 2 + 2.5  # points
        style = ({"fontsize": 8, "fontweight": "bold", "color": QUF_TEXT} if p.is_quf
                 else {"fontsize": 7, "color": TEXT_COLOR})

        def annotate(name):
            dx, dy, ha, va = POSITIONS[name]
            return ax.annotate(p.label, xy=(p.size_mb, p.ppl_wikitext2), xytext=(dx * gap, dy * gap),
                               textcoords="offset points", ha=ha, va=va, zorder=5, **style)

        chosen = None
        for name in names:
            label = annotate(name)
            box = label.get_window_extent(renderer)
            if _inside(box, ax_box) and not any(_pad(box, 1.5).overlaps(o) for o in obstacles):
                chosen = box
                break
            label.remove()
        if chosen is None:
            crowded.append(p.label)
            chosen = annotate(names[0]).get_window_extent(renderer)
        obstacles.append(_pad(chosen, 1.5))
    return crowded


def _draw_panel(ax, panel: Panel, renderer, ylabel: bool) -> list[str]:
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import Bbox

    gguf = [p for p in panel.points if not p.is_quf]
    quf = [p for p in panel.points if p.is_quf]
    if gguf:
        ax.plot([p.size_mb for p in gguf], [p.ppl_wikitext2 for p in gguf], linestyle="none",
                marker="o", markersize=GGUF_MARKERSIZE, color=GGUF_COLOR,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    if quf:
        ax.plot([p.size_mb for p in quf], [p.ppl_wikitext2 for p in quf], linestyle="none",
                marker="*", markersize=QUF_MARKERSIZE, color=QUF_COLOR,
                markeredgecolor=QUF_EDGE, markeredgewidth=0.6, zorder=4)
    xs = [p.size_mb for p in panel.points]
    ys = [p.ppl_wikitext2 for p in panel.points]
    if panel.fp16_ppl is not None:
        ys.append(panel.fp16_ppl)
        ax.axhline(panel.fp16_ppl, color=FP16_COLOR, linestyle=FP16_DASHES, linewidth=1.0, zorder=2)
    ax.set_xlim(*_padded(min(xs), max(xs), 0.12, 0.18))  # room for a label centred on the largest file
    ax.set_ylim(*_padded(min(ys), max(ys), 0.10, 0.14))

    ax.set_title(panel.model, fontsize=9, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax.set_xlabel("File size (MB = 10$^6$ bytes)", fontsize=7.5)
    if ylabel:
        ax.set_ylabel("WikiText-2 perplexity\n(llama protocol, lower is better)", fontsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.tick_params(labelsize=7, length=3, width=0.6)
    ax.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Everything a label must not cover, in display pixels.
    ax_box = ax.get_window_extent(renderer)
    obstacles = []
    for p in panel.points:
        cx, cy = ax.transData.transform((p.size_mb, p.ppl_wikitext2))
        r = (QUF_MARKERSIZE if p.is_quf else GGUF_MARKERSIZE) / 2 * DPI / 72
        obstacles.append(Bbox.from_extents(cx - r, cy - r, cx + r, cy + r))
    if panel.fp16_ppl is not None:
        y = ax.transData.transform((ax.get_xlim()[0], panel.fp16_ppl))[1]
        obstacles.append(Bbox.from_extents(ax_box.x0, y - 2, ax_box.x1, y + 2))
        tag = ax.annotate("fp16", xy=(0.985, panel.fp16_ppl), xycoords=ax.get_yaxis_transform(),
                          xytext=(0, 2.5), textcoords="offset points", ha="right", va="bottom",
                          fontsize=7, color=FP16_COLOR)
        obstacles.append(_pad(tag.get_window_extent(renderer), 2))
    else:
        obstacles.append(_place_note(ax, renderer, obstacles))
    return _place_labels(ax, renderer, panel, obstacles)


def draw(panels: list[Panel], device: str, source: str, out_path: Path) -> list[str]:
    """Write the chart PNG; return "model: label" for labels that had no free position."""
    if not panels:
        raise ValueError("no panel to draw")
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    crowded = []
    with matplotlib.rc_context():
        matplotlib.rcdefaults()  # ignore any local matplotlibrc so that the output is reproducible
        matplotlib.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.8,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        })
        # +0.5 px keeps float rounding from dropping a pixel of the requested size.
        fig = Figure(figsize=((FIG_WIDTH_PX + 0.5) / DPI, (FIG_HEIGHT_PX + 0.5) / DPI),
                     dpi=DPI, facecolor="white")
        canvas = FigureCanvasAgg(fig)
        axes = fig.subplots(1, len(panels), squeeze=False)[0]
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.2, top=0.78, wspace=0.34)
        renderer = canvas.get_renderer()
        for index, (ax, panel) in enumerate(zip(axes, panels)):
            crowded += [f"{panel.model}: {label}" for label in _draw_panel(ax, panel, renderer, index == 0)]

        fig.text(0.5, 0.975, "QUF 0.4.0 vs GGUF: file size and WikiText-2 perplexity",
                 ha="center", va="top", fontsize=10.5, fontweight="bold", color=TEXT_COLOR)
        handles = [
            Line2D([], [], linestyle="none", marker="*", markersize=11, color=QUF_COLOR,
                   markeredgecolor=QUF_EDGE, markeredgewidth=0.6, label="QUF 0.4.0"),
            Line2D([], [], linestyle="none", marker="o", markersize=GGUF_MARKERSIZE, color=GGUF_COLOR,
                   markeredgecolor="white", markeredgewidth=0.7, label="GGUF (llama.cpp)"),
            Line2D([], [], linestyle=FP16_DASHES, linewidth=1.0, color=FP16_COLOR, label="fp16 reference"),
        ]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.1, 0.885), ncol=3,
                   frameon=False, fontsize=7.5, handlelength=2.2, handletextpad=0.5,
                   columnspacing=1.5, borderaxespad=0)
        fig.text(0.985, 0.885, "lower-left is better", ha="right", va="center", fontsize=8,
                 style="italic", color=TEXT_COLOR)
        fig.text(0.5, 0.02, f"Source: {source}. GPU: {device}.", ha="center", va="bottom",
                 fontsize=7, color="#555555")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=DPI, facecolor="white", metadata={"Software": None})
    return crowded


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _print_summary(panels: list[Panel], bench: dict) -> None:
    for line in describe(panels):
        print(line)
    skipped = skipped_rows(bench)
    print("rows without a size and a WikiText-2 value: " + (", ".join(skipped) if skipped else "none"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw docs/assets/quf_vs_gguf.png from docs/bench_accuracy.json.")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH,
                        help="benchmark JSON (default: docs/bench_accuracy.json)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output PNG (default: docs/assets/quf_vs_gguf.png)")
    parser.add_argument("--check", action="store_true",
                        help="only check that the JSON has the keys the chart needs (no matplotlib)")
    args = parser.parse_args(argv)

    if not args.check:
        try:
            import matplotlib  # noqa: F401  (development-only, see the module docstring)
        except ImportError:
            print(MISSING_MATPLOTLIB, file=sys.stderr)
            return 1

    try:
        bench = json.loads(args.bench.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.bench}: {exc}", file=sys.stderr)
        return 2
    problems = check_bench(bench)
    if problems:
        for problem in problems:
            print(f"{args.bench}: {problem}", file=sys.stderr)
        return 2

    panels = collect_points(bench)
    if args.check:
        _print_summary(panels, bench)
        print("check passed")
        return 0

    crowded = draw(panels, device=bench["environment"]["device"].strip(),
                   source=_display_path(args.bench), out_path=args.out)
    for item in crowded:
        print(f"warning: label without a free position, check the chart: {item}", file=sys.stderr)
    print(f"wrote {args.out}")
    _print_summary(panels, bench)
    return 0


if __name__ == "__main__":
    sys.exit(main())
