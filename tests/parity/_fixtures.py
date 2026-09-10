"""Resolve stored native captures into verified paths for the CPU subset.

The fixture files live outside the repository (see ``docs/parity-cpu.md``); git
carries only their digests. Everything a test is handed here has been checked
against the manifest byte count and sha256 first, so a truncated copy or a file
from a later capture fails as itself instead of as a parity regression.

Missing fixtures FAIL, never skip: a skipped parity test is indistinguishable
from a passing one in a summary line, which is how gated suites go dark
(``tests/models/test_optional_suite_gate.py`` exists for the same reason).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from ._manifest import FixtureFile, ManifestEntry, entry_for

#: Override for the directory the fetch helper fills.
ENV_VAR = "FOLDJAX_PARITY_FIXTURES"

#: Default store on the machine the captures were taken on. Deliberately
#: outside the checkout: the files are 7-16 MB per case against a 22 MB pack.
DEFAULT_ROOT = Path("/home/jaemin/non-project/optimizing/foldjax-bench/parity-fixtures")

_CHUNK = 1 << 20


def fixtures_root() -> Path:
    override = os.environ.get(ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_ROOT


def case_dir(port: str, case: str, root: Path | None = None) -> Path:
    return (fixtures_root() if root is None else root) / port / case


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, spec: FixtureFile) -> str | None:
    """Describe how ``path`` differs from ``spec``, or ``None`` if it matches."""
    if not path.is_file():
        return f"{path} is missing"
    size = path.stat().st_size
    if size != spec.size_bytes:
        return f"{path} is {size} bytes, the manifest records {spec.size_bytes}"
    # Hashed every time rather than memoised on (path, mtime, size): a fixture
    # re-fetched mid-session can land inside one filesystem timestamp tick, and
    # a stale digest here would pass a file nobody checked. 14 MB costs ~40 ms.
    found = sha256_of(path)
    if found != spec.sha256:
        return f"{path} has sha256 {found}, the manifest records {spec.sha256}"
    return None


def fetch_command(entry: ManifestEntry) -> str:
    return f"python -m tests.parity.fetch --from {entry.capture_dir}"


def unusable_message(entry: ManifestEntry, problems: list[str], root: Path) -> str:
    lines = [
        f"CPU parity fixture unusable: {entry.port}/{entry.case} (tier {entry.tier})",
        *(f"  {problem}" for problem in problems),
        f"  fixtures root: {root} (override with {ENV_VAR})",
        "fetch the files by digest from the capture that produced them:",
        f"  {fetch_command(entry)}",
        f"  capture provenance: {entry.capture_provenance}",
    ]
    return "\n".join(lines)


def tripwire_message(entry: ManifestEntry, mismatched: list[str]) -> str:
    lines = [
        f"CPU parity fixture is stale: {entry.port}/{entry.case} (tier {entry.tier})",
        *(f"  {item}" for item in mismatched),
        "the stored capture was produced against a different schema; re-capture "
        "on GPU and update the manifest -- do not re-featurize here, that would "
        "compare this checkout against itself.",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class ResolvedCase:
    """One manifest entry whose files are present and match their digests."""

    entry: ManifestEntry
    files: Mapping[str, Path]

    def path(self, name: str) -> Path:
        try:
            return self.files[name]
        except KeyError:
            known = ", ".join(sorted(self.files))
            raise KeyError(
                f"{self.entry.port}/{self.entry.case} has no fixture file {name!r} "
                f"(manifest lists {known})"
            ) from None

    def assert_tripwire(self, observed: Mapping[str, str]) -> None:
        """Fail loudly when the checkout no longer matches the capture.

        ``observed`` is what the port reports *now* (tape schema version,
        featurizer commit); keys the manifest does not record are ignored, so a
        port can add one before every manifest carries it.
        """
        compared = [key for key in self.entry.tripwire if key in observed]
        if not compared:
            # Otherwise a port that renames what it reports keeps calling this
            # and the check quietly stops comparing anything.
            pytest.fail(
                f"{self.entry.port}/{self.entry.case}: the tripwire compared "
                f"nothing -- manifest records {sorted(self.entry.tripwire)}, "
                f"this checkout reported {sorted(observed)}",
                pytrace=False,
            )
        mismatched = [
            f"{key}: capture {self.entry.tripwire[key]!r}, "
            f"this checkout {observed[key]!r}"
            for key in compared
            if observed[key] != self.entry.tripwire[key]
        ]
        if mismatched:
            pytest.fail(tripwire_message(self.entry, mismatched), pytrace=False)


@dataclass(frozen=True)
class FixtureStore:
    """Manifest + fixture root pair a test resolves cases through."""

    root: Path
    manifest_dir: Path | None = None

    def entry(self, port: str, case: str, tier: str) -> ManifestEntry:
        return entry_for(port, case, tier, self.manifest_dir)

    def resolve(self, entry: ManifestEntry) -> ResolvedCase:
        directory = self.root / entry.port / entry.case
        problems = [
            problem
            for spec in entry.files.values()
            if (problem := check_file(directory / spec.name, spec)) is not None
        ]
        if problems:
            pytest.fail(unusable_message(entry, problems, self.root), pytrace=False)
        return ResolvedCase(
            entry=entry,
            files={name: directory / name for name in entry.files},
        )

    def case(self, port: str, case: str, *, tier: str) -> ResolvedCase:
        return self.resolve(self.entry(port, case, tier))
