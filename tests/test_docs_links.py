"""Relative links in the documentation name files and headings that exist.

README.md is covered by ``test_distribution.py`` (its links are absolute so
PyPI can render it); this covers every page under ``docs/``, where a moved or
renamed note, or a retitled section, otherwise leaves a link that only a
reader finds broken.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
_FENCE = re.compile(r"^(```|~~~).*?^\1", re.S | re.M)
_LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")


def _prose(page: Path) -> str:
    return _FENCE.sub("", page.read_text(encoding="utf-8"))


def _relative_targets(page: Path) -> list[str]:
    targets = (match.group(1) for match in _LINK.finditer(_prose(page)))
    return [target for target in targets if not _EXTERNAL.match(target)]


def _anchors(page: Path) -> set[str]:
    """GitHub's heading ids: lower-cased, punctuation dropped, spaces to '-'."""
    seen: dict[str, int] = {}
    anchors = set()
    for line in _prose(page).splitlines():
        heading = _HEADING.match(line)
        if heading is None:
            continue
        slug = re.sub(r"[^\w\- ]", "", heading.group(1).strip().lower())
        slug = slug.replace(" ", "-")
        count = seen.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return anchors


PAGES = sorted(DOCS.rglob("*.md"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(page: Path) -> None:
    broken = []
    for target in _relative_targets(page):
        path, _, fragment = target.partition("#")
        destination = page.parent / path if path else page
        if not destination.exists():
            broken.append(target)
        elif (
            fragment
            and destination.suffix == ".md"
            and fragment not in _anchors(destination)
        ):
            broken.append(target)
    assert broken == []


def test_archive_index_lists_every_archived_note() -> None:
    archive = DOCS / "archive"
    listed = set(_relative_targets(archive / "README.md"))
    notes = {p.name for p in archive.glob("*.md") if p.name != "README.md"}
    assert notes - listed == set()
