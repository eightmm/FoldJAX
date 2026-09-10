"""The CPU parity subset must never silently stop existing.

Not marked ``cpu_parity`` -- deliberately. A guard that sits behind the gate it
guards proves nothing (``tests/models/openfold3/test_parity_gate_is_meaningful.py``
is ``torch_parity``-marked and so never runs where torch is absent), and a
collection gate that hides a whole suite is exactly how 101 Boltz-2 tests went
unrun for months.

Two subprocesses, because collection policy is only observable from outside the
run that is applying it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ._manifest import MANIFEST_DIR, all_entries, load_all_manifests

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET = "tests/parity"


def _environment() -> dict[str, str]:
    """The child must be told what to collect by its arguments, nothing else.

    An outer `PYTEST_ADDOPTS` would otherwise be applied twice and could invert
    the selection this test is checking.
    """
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env.setdefault("JAX_PLATFORMS", "cpu")
    return env


def _collect(*extra: str) -> list[str]:
    """Node ids `pytest --collect-only` reports for the subset directory."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            SUBSET,
            *extra,
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return [line.strip() for line in completed.stdout.splitlines() if "::" in line]


def test_the_marker_alone_still_collects_the_subset() -> None:
    """`pytest -m cpu_parity` is the shipped selection; it must find tests."""
    assert _collect("-m", "cpu_parity") != []


def test_the_subset_is_deselected_by_default() -> None:
    """A default run must not try to replay captures it has no fixtures for."""
    marked = set(_collect("-m", "cpu_parity"))
    default = set(_collect())
    assert default, "the subset directory collects nothing at all"
    assert marked.isdisjoint(default)


def test_every_shipped_manifest_validates() -> None:
    for manifest in load_all_manifests():
        assert manifest.entries


def test_every_manifest_entry_names_tests_that_are_collected() -> None:
    """A fixture whose test does not run is a digest certifying nothing."""
    collected = _collect("-m", "cpu_parity")
    for entry in all_entries():
        for node_id in entry.test_node_ids:
            assert any(
                item == node_id or item.startswith(f"{node_id}[") for item in collected
            ), f"{entry.port}/{entry.case} tier {entry.tier} names {node_id}"


def test_the_manifest_directory_holds_only_manifests_and_the_schema_note() -> None:
    """`*.json` is what the loader reads; anything else would be ignored."""
    unexpected = [
        path.name
        for path in sorted(MANIFEST_DIR.iterdir())
        if path.suffix != ".json" and path.name != "README.md"
    ]
    assert unexpected == []
