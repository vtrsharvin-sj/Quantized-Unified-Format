"""Checks packaging metadata in pyproject.toml (task 13.1)."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/vtrsharvin-sj/Quantized-Model-Format"


def _project() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


def test_license_and_author():
    p = _project()
    assert p["license"] == "MIT"
    assert any(a["name"] == "vtrsharvin-sj" for a in p["authors"])
    assert "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_torch_is_base_dependency_and_cuda_extra_has_no_torch():
    p = _project()
    assert any(d.startswith("torch") for d in p["dependencies"])
    assert not any(d.startswith("torch") for d in p["optional-dependencies"].get("cuda", []))


def test_urls_classifiers_version():
    p = _project()
    assert REPO_URL in p["urls"].values()
    assert p["classifiers"]
    assert p["version"] == "0.2.0"
