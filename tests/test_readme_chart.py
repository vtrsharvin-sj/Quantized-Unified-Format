"""scripts/make_readme_chart.py: the README chart data and its command line.

Validates: Requirements 43.15, 43.16, 43.17, 43.18, 43.19, 43.20, 43.22.
Only the PNG smoke test needs matplotlib, so CI (without matplotlib) skips it.
"""

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "make_readme_chart.py"
BENCH = ROOT / "docs" / "bench_accuracy.json"

pytestmark = pytest.mark.skipif(
    not (SCRIPT.exists() and BENCH.exists()),
    reason="needs the repository checkout (scripts/ and docs/ are not in the sdist)")


@pytest.fixture(scope="module")
def chart():
    spec = importlib.util.spec_from_file_location("make_readme_chart", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return json.loads(BENCH.read_text(encoding="utf-8"))


def _plottable(rows):
    return {name: row for name, row in rows.items()
            if isinstance(row, dict) and "file_size_bytes" in row and "ppl_wikitext2" in row}


def test_collect_points_on_the_real_json(chart, bench):
    panels = {panel.model: panel for panel in chart.collect_points(bench)}
    expected = {
        "Qwen2.5-0.5B": _plottable(bench["pareto_040"]["0.5B"]),
        "Qwen2.5-1.5B": _plottable(bench["pareto_040"]["1.5B"]),
        "Qwen2.5-3B": _plottable(bench["scaling_3b"]),
    }
    assert list(panels) == list(expected)
    for model, rows in expected.items():
        points = panels[model].points
        assert [p.label for p in points] == [chart.ROW_LABELS.get(name, name) for name in rows]
        for point, (name, row) in zip(points, rows.items()):
            assert point.size_mb == pytest.approx(row["file_size_bytes"] / 1e6)
            assert point.ppl_wikitext2 == pytest.approx(row["ppl_wikitext2"])
            assert point.is_quf == (name == "QUF")
        assert [p.label for p in points if p.is_quf] == ["QUF 0.4.0"]
    llama = bench["models_llama"]
    assert panels["Qwen2.5-0.5B"].fp16_ppl == pytest.approx(llama["Qwen/Qwen2.5-0.5B"]["fp16"]["ppl_wikitext2"])
    assert panels["Qwen2.5-1.5B"].fp16_ppl == pytest.approx(llama["Qwen/Qwen2.5-1.5B"]["fp16"]["ppl_wikitext2"])
    assert panels["Qwen2.5-3B"].fp16_ppl is None
    assert {"Q4_K_M imatrix", "IQ4_XS imatrix"} <= {p.label for p in panels["Qwen2.5-0.5B"].points}


def test_check_flag_needs_no_matplotlib(chart, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert chart.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "check passed" in out
    assert "Qwen2.5-3B: 2 rows" in out and "fp16 not measured" in out


def test_check_flag_from_the_command_line():
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "check passed" in result.stdout


def test_check_flag_reports_missing_keys(chart, tmp_path, capsys):
    broken = tmp_path / "bench.json"
    broken.write_text(json.dumps({"environment": {}, "models_llama": {}}), encoding="utf-8")
    assert chart.main(["--check", "--bench", str(broken)]) == 2
    err = capsys.readouterr().err
    assert "environment.device" in err and "pareto_040" in err


def test_missing_matplotlib_exits_with_status_1(chart, monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    out = tmp_path / "chart.png"
    assert chart.main(["--out", str(out)]) == 1
    assert "pip install matplotlib" in capsys.readouterr().err
    assert not out.exists()


def test_png_smoke(chart, tmp_path):
    pytest.importorskip("matplotlib")
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    assert chart.main(["--out", str(first)]) == 0
    assert chart.main(["--out", str(second)]) == 0
    data = first.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", data[16:24]) == (chart.FIG_WIDTH_PX, chart.FIG_HEIGHT_PX)
    assert data == second.read_bytes()  # deterministic output
