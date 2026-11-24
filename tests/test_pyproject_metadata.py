"""Checks packaging metadata in pyproject.toml (tasks 13.1 and 40.3)."""

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
REPO_URL = "https://github.com/vtrsharvin-sj/Quantized-Unified-Format"

EXPECTED_URLS = {
    "Homepage": REPO_URL,
    "Repository": REPO_URL,
    "Documentation": f"{REPO_URL}/tree/main/docs",
    "Changelog": f"{REPO_URL}/blob/main/CHANGELOG.md",
    "Issues": f"{REPO_URL}/issues",
}
REQUIRED_KEYWORDS = {
    "quantization", "llm", "gptq", "int4", "4-bit", "triton", "cuda", "windows", "qwen",
}
# CI runner image -> the operating system classifier it stands for.
CI_OS_CLASSIFIERS = {
    "ubuntu-latest": "Operating System :: POSIX :: Linux",
    "windows-latest": "Operating System :: Microsoft :: Windows",
}


def _project() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


def _ci_matrix_list(key: str) -> list:
    """Return the inline list under `key:` in the CI matrix, e.g. python-version: ["3.10", ...]."""
    if not CI_WORKFLOW.is_file():
        pytest.skip("the CI workflow file is not present (e.g. an installed sdist)")
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    m = re.search(rf"^\s*{re.escape(key)}:\s*\[([^\]]*)\]", text, re.MULTILINE)
    assert m, f"no inline `{key}: [...]` list in {CI_WORKFLOW}"
    return [v.strip().strip("\"'") for v in m.group(1).split(",") if v.strip()]


def _dist_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[(]", requirement.strip(), maxsplit=1)[0].lower()


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
    assert p["version"] == "0.4.1"


def test_description_is_one_short_line():
    description = _project()["description"]
    assert description.strip()
    assert "\n" not in description
    assert len(description) <= 100


def test_keywords():
    assert REQUIRED_KEYWORDS <= set(_project()["keywords"])


def test_project_urls():
    urls = _project()["urls"]
    for name, url in EXPECTED_URLS.items():
        assert urls.get(name) == url, name
    assert all(u.startswith("https://") for u in urls.values())


def test_gpu_and_operating_system_classifiers():
    classifiers = _project()["classifiers"]
    assert "Environment :: GPU :: NVIDIA CUDA" in classifiers
    assert "Operating System :: OS Independent" not in classifiers
    os_classifiers = {c for c in classifiers if c.startswith("Operating System ::")}
    assert os_classifiers == set(CI_OS_CLASSIFIERS.values())
    # The declared operating systems are exactly the ones CI tests.
    ci_os = _ci_matrix_list("os")
    assert {CI_OS_CLASSIFIERS.get(o, o) for o in ci_os} == os_classifiers


def test_python_classifiers_match_ci_matrix():
    classifiers = _project()["classifiers"]
    declared = {
        m.group(1)
        for c in classifiers
        if (m := re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", c))
    }
    ci_versions = _ci_matrix_list("python-version")
    assert ci_versions
    assert declared == set(ci_versions)


def test_no_matplotlib_dependency():
    p = _project()
    requirements = list(p["dependencies"])
    for extra in p.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    assert "matplotlib" not in {_dist_name(r) for r in requirements}
