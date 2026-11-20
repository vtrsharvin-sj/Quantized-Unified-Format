"""CLI unit tests: --help, unknown subcommand, error exit codes (Req 12.1, 12.3, 12.4, 12.5)."""

from __future__ import annotations

import pytest

from quf.cli import COMMANDS, main


def test_top_level_help_exits_zero_and_lists_subcommands(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for cmd in COMMANDS:
        assert cmd in out


@pytest.mark.parametrize("cmd", COMMANDS)
def test_subcommand_help_exits_zero(cmd, capsys):
    assert main([cmd, "--help"]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_unknown_subcommand_exits_nonzero_and_lists_valid(capsys):
    assert main(["frobnicate"]) != 0
    err = capsys.readouterr().err
    for cmd in COMMANDS:
        assert cmd in err


def test_no_subcommand_exits_nonzero(capsys):
    assert main([]) != 0


def test_info_missing_file_exits_nonzero(tmp_path, capsys):
    assert main(["info", str(tmp_path / "nope.quf")]) != 0
    assert "quf: error" in capsys.readouterr().err


def test_info_corrupt_file_exits_nonzero(tmp_path, capsys):
    p = tmp_path / "bad.quf"
    p.write_bytes(b"this is not a quf file at all" * 4)
    assert main(["info", str(p)]) != 0
    assert "quf: error" in capsys.readouterr().err


def test_convert_missing_source_exits_nonzero(tmp_path, capsys):
    code = main(["convert", str(tmp_path / "no_such_model_dir"), str(tmp_path / "o.quf")])
    assert code != 0
    assert "quf: error" in capsys.readouterr().err


def test_convert_missing_args_exits_nonzero(capsys):
    assert main(["convert"]) != 0


def test_bench_missing_file_exits_nonzero(tmp_path, capsys):
    code = main(["bench", str(tmp_path / "nope.quf"), "--out", str(tmp_path / "o.json")])
    assert code != 0
    assert "quf: error" in capsys.readouterr().err


def test_bench_requires_out(capsys):
    assert main(["bench", "m.quf"]) != 0
