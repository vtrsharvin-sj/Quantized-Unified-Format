"""Static checks of README.md, which is both the GitHub README and the PyPI description.

CPU only, no network. PyPI renders the same file, so it must be plain GitHub-flavoured
Markdown with absolute links, and every repository link must point to a path that exists.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SPEC_REQUIREMENTS = ROOT / ".kiro" / "specs" / "quf-quantized-format" / "requirements.md"

REPO = "vtrsharvin-sj/Quantized-Unified-Format"
REPO_LINK = re.compile(rf"^https://github\.com/{re.escape(REPO)}/(blob|tree)/main/(.+)$")
IMAGE_PREFIX = f"https://raw.githubusercontent.com/{REPO}/main/"
BADGE_IMAGES = ("https://img.shields.io/", f"https://github.com/{REPO}/actions/workflows/ci.yml/badge.svg")
# Badge name -> accepted image URL pattern.
BADGES = {
    "PyPI version": r"https://img\.shields\.io/pypi/v/quf\b",
    "Python versions": r"https://img\.shields\.io/pypi/pyversions/quf\b",
    "license": r"https://img\.shields\.io/(?:badge/license-|pypi/l/quf\b|github/license/)",
    "CI status": (rf"https://github\.com/{re.escape(REPO)}/actions/workflows/ci\.yml/badge\.svg"
                  rf"|https://img\.shields\.io/github/actions/workflow/status/{re.escape(REPO)}/ci\.yml"),
}
BANNED_WORDS = ("best", "fastest", "greatest", "state-of-the-art", "state of the art", "revolutionary", "blazing")
SECTIONS = (
    "Why QUF", "Results at a glance", "Install", "Quickstart", "Python API", "How it works",
    "Benchmarks", "CLI reference", "Supported models", "Limitations", "Troubleshooting",
    "File format", "Development", "Changelog", "License", "Acknowledgements",
)
# Spec glossary terms that must stay out of the public README. In a repository checkout,
# every other underscore glossary term of the spec's requirements.md is checked as well.
SPEC_TERMS = ("Beat_Target", "Speed_Floor", "Pareto_Table", "Glance_Table", "Bench_JSON",
              "Copy_Paste_Test", "Public_README", "Llama_Protocol", "Fast_Path")
# Options of other tools that the README shows (pip, venv, llama-imatrix). Every other
# --option in a code block, a code span or a table row must be an option of the quf CLI.
OTHER_TOOL_OPTIONS = frozenset({"--index-url", "--ignore-installed", "--no-deps",
                                "--system-site-packages", "--chunks"})

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
OPTION = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
QUF_COMMAND = re.compile(r"^\s*(?:[A-Z][A-Z0-9_]*=\S*\s+)*quf\s+(info|convert|run|bench)\b")


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _split(text: str) -> tuple[list[str], list[tuple[int, str]]]:
    """Return (prose lines with fenced blocks removed, [(line number, info string)] of each opening fence)."""
    prose: list[str] = []
    openings: list[tuple[int, str]] = []
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        m = FENCE.match(line)
        if fence is None:
            if m:
                fence = m.group(1)
                openings.append((number, m.group(2).strip()))
            else:
                prose.append(line)
        elif m and m.group(1).startswith(fence) and not m.group(2).strip():
            fence = None
    assert fence is None, "README.md has an unclosed code fence"
    return prose, openings


def _prose() -> str:
    """README text outside fenced blocks and inline code spans."""
    prose, _ = _split(_text())
    return re.sub(r"`[^`\n]*`", "", "\n".join(prose))


def _targets(prose: str) -> list[str]:
    inline = re.findall(r"\]\(\s*<?([^)\s>]*)", prose)
    references = re.findall(r"^\s*\[[^\]]+\]:\s*<?(\S+?)>?\s*$", prose, re.MULTILINE)
    return inline + references


def _images(prose: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(\s*<?([^)\s>]*)", prose)


def _require_checkout() -> None:
    if not CI_WORKFLOW.is_file():
        pytest.skip("not a repository checkout (e.g. an installed sdist)")


def test_sections_in_order():
    headings = re.findall(r"^## (.+?)\s*$", "\n".join(_split(_text())[0]), re.MULTILINE)
    assert tuple(headings) == SECTIONS


def test_line_count():
    assert 300 <= len(_text().splitlines()) <= 450


def test_why_quf_has_four_or_five_bullets():
    section = _text().split("## Why QUF", 1)[1].split("\n## ", 1)[0]
    bullets = [line for line in section.splitlines() if line.startswith("- ")]
    assert 4 <= len(bullets) <= 5


def test_no_raw_html():
    tags = re.findall(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>|<!--", _prose())
    assert tags == []


def test_no_mermaid_and_no_alerts():
    text = _text()
    assert "```mermaid" not in text.lower()
    assert re.search(r"\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)", text, re.IGNORECASE) is None


def test_every_code_block_has_a_language():
    _, openings = _split(_text())
    assert openings, "README.md has no code blocks"
    missing = [number for number, info in openings if not info]
    assert missing == [], f"code fences without a language on lines {missing}"


def test_every_link_and_image_is_absolute_https():
    targets = _targets(_prose())
    assert targets, "README.md has no links"
    bad = [t for t in targets if not t.startswith("https://") or "#" in t]
    assert bad == [], f"relative, non-https or #anchor links: {bad}"


def test_images_are_badges_or_repository_files():
    _require_checkout()
    images = _images(_prose())
    assert any(i.startswith(IMAGE_PREFIX) for i in images), "the chart image is missing"
    for url in images:
        if url.startswith(BADGE_IMAGES):
            continue
        assert url.startswith(IMAGE_PREFIX), f"image not from the repository: {url}"
        assert (ROOT / unquote(url[len(IMAGE_PREFIX):])).is_file(), f"image path does not exist: {url}"


def test_repository_links_point_to_existing_paths():
    _require_checkout()
    checked = 0
    for url in _targets(_prose()):
        m = REPO_LINK.match(url)
        if m is None:
            continue
        kind, rel = m.groups()
        path = ROOT / unquote(rel.rstrip("/"))
        if kind == "blob":
            assert path.is_file(), f"blob link to a missing file: {url}"
        else:
            assert path.is_dir(), f"tree link to a missing folder: {url}"
        checked += 1
    assert checked, "README.md has no repository links"


def test_no_banned_superlatives():
    text = _text().lower()
    found = [w for w in BANNED_WORDS if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text)]
    assert found == []


def _version(s: str) -> tuple[int, ...]:
    return tuple(int(part) for part in s.split("."))


def test_quf_versions_are_current_or_older():
    import quf

    current = _version(re.match(r"\d+\.\d+\.\d+", quf.__version__).group(0))
    text = _text()
    mentions = re.findall(r"(?i)\bquf\s+(?:>=\s*|≥\s*)?(\d+\.\d+\.\d+)\b", text)
    history = re.search(r"^Release history:.*$", text, re.MULTILINE)
    assert history is not None, "the Release history line is missing"
    mentions += re.findall(r"\b(\d+\.\d+\.\d+)\b", history.group(0))
    newer = [v for v in mentions if _version(v) > current]
    assert newer == [], f"README names quf versions newer than {quf.__version__}: {newer}"


def _fenced_lines(text: str) -> list[str]:
    """The lines inside fenced code blocks, without the fences."""
    inside: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        m = FENCE.match(line)
        if fence is None:
            if m:
                fence = m.group(1)
        elif m and m.group(1).startswith(fence) and not m.group(2).strip():
            fence = None
        else:
            inside.append(line)
    return inside


def _section(title: str) -> list[str]:
    """Raw lines of '## title', from its heading to the next '## ' heading outside a code
    fence, without trailing blank lines."""
    lines = _text().splitlines()
    start = end = None
    fence: str | None = None
    for number, line in enumerate(lines):
        m = FENCE.match(line)
        if fence is not None:
            if m and m.group(1).startswith(fence) and not m.group(2).strip():
                fence = None
            continue
        if m:
            fence = m.group(1)
        elif line.startswith("## "):
            if start is not None:
                end = number
                break
            if line[3:].strip() == title:
                start = number
    assert start is not None, f"README.md has no '## {title}' section"
    section = lines[start:end]
    while section and not section[-1].strip():
        section.pop()
    return section


def test_badges_for_pypi_python_license_and_ci():
    head = _prose().split("\n## ", 1)[0]  # the title, the tagline and the badges
    images = _images(head)
    missing = [name for name, pattern in BADGES.items() if not any(re.match(pattern, u) for u in images)]
    assert missing == [], f"badges missing above 'Why QUF': {missing}"


def test_how_it_works_is_at_most_40_lines():
    section = _section("How it works")
    assert len(section) <= 40, f"'How it works' has {len(section)} lines, including its heading"


def _spec_terms() -> set[str]:
    terms = set(SPEC_TERMS)
    if SPEC_REQUIREMENTS.is_file():
        glossary = SPEC_REQUIREMENTS.read_text(encoding="utf-8")
        terms.update(re.findall(r"^- \*\*([A-Za-z0-9]+(?:_[A-Za-z0-9]+)+)\*\*", glossary, re.MULTILINE))
    return terms


def test_no_task_numbers_or_spec_terms():
    text = _text()
    numbers = re.findall(r"(?i)\b(?:tasks?|requirements?)\s+\d+(?:\.\d+)*\b", text)
    assert numbers == [], f"internal task or requirement numbers: {numbers}"
    terms = sorted(t for t in _spec_terms() if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text))
    assert terms == [], f"spec glossary terms: {terms}"


def _cli_options() -> dict[str, set[str]]:
    """Option strings of the quf CLI: key '' for `quf` itself, else the command name."""
    from quf.cli import _build_parser

    parser = _build_parser()
    options = {"": set(parser._option_string_actions)}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                options[name] = set(sub._option_string_actions)
    assert {"info", "convert", "run", "bench"} <= set(options)
    return options


def test_every_option_exists_in_the_cli_parser():
    options = _cli_options()
    known = set().union(*options.values())
    text = _text()
    prose, _ = _split(text)
    spans = re.findall(r"`([^`\n]+)`", "\n".join(prose))
    rows = [line for line in prose if line.lstrip().startswith("|")]
    unknown: set[str] = set()
    misplaced: set[str] = set()
    checked = 0
    for snippet in _fenced_lines(text) + spans + rows:
        command = QUF_COMMAND.match(snippet)
        for option in OPTION.findall(snippet):
            if option in OTHER_TOOL_OPTIONS:
                continue
            checked += 1
            if option not in known:
                unknown.add(option)
            elif command and option not in options[command.group(1)]:
                misplaced.add(f"quf {command.group(1)} {option}")
    assert checked, "README.md shows no quf options"
    assert unknown == set(), f"options that quf.cli does not define: {sorted(unknown)}"
    assert misplaced == set(), f"options shown with a command that does not take them: {sorted(misplaced)}"
