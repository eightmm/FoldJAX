"""Collection policy and fixture wiring for the CPU parity subset.

The subset is deselected by default. It is not gated with ``collect_ignore``:
that mechanism is what hid 101 Boltz-2 tests behind a missing optional import,
because a module that is never imported cannot report that it is not running.
Here every test is imported and collected, then deselected by marker, so
``--collect-only`` and ``test_gate.py`` can both see the whole set.

``addopts`` is deliberately empty (see ``pyproject.toml``), so the default
deselection has to be this hook rather than a ``-m`` expression in the config.

The option itself is registered in ``tests/conftest.py``: pytest only parses
options from conftests on the rootdir chain, so registering it here would make
``pytest --run-cpu-parity`` from the repository root fail with "unrecognized
arguments" -- verified in this tree before moving it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from ._fixtures import FixtureStore, ResolvedCase, fixtures_root

CPU_PARITY_MARKER = "cpu_parity"


def write_fixture(directory: Path, name: str, payload: bytes) -> dict[str, object]:
    """Write a stand-in fixture file and return its manifest ``files`` value."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def example_manifest(
    port: str = "demo",
    *,
    case: str = "demo_case",
    tier: str = "A",
    files: Mapping[str, Mapping[str, object]] | None = None,
    **entry_overrides: object,
) -> dict[str, object]:
    """A minimal manifest that validates, for tests to break one key at a time."""
    entry: dict[str, object] = {
        "case": case,
        "tier": tier,
        "tokens": 76,
        "capture": {
            "provenance": "/captures/demo-native-20260909/demo_case/native-A/"
            "provenance.json",
            "commit": "0c6d059",
        },
        "tripwire": {"tape_schema": "demo-v1"},
        "files": dict(files)
        if files
        else {"trunk.npz": {"sha256": "0" * 64, "bytes": 8}},
        "gpu_residual_angstrom": {"max": 0.037},
        "cpu_calibration": {
            "residual_angstrom": 0.02,
            "wall_seconds": 91.0,
            "source_commit": "0c6d059",
        },
        "tolerance": {
            "metric": "rmsd_angstrom",
            "value": 0.05,
            "set_from": "CPU calibration + margin",
        },
        "test_node_ids": [
            f"tests/parity/test_{port}_tier_{tier.lower()}.py::test_case"
        ],
    }
    entry.update(entry_overrides)
    return {"schema_version": 1, "port": port, "entries": [entry]}


def _asked_for_the_subset(config: pytest.Config) -> bool:
    """Either ``--run-cpu-parity`` or an explicit ``-m`` naming the marker.

    The marker alone selecting the subset is what ``test_gate.py`` asserts with
    ``pytest -m cpu_parity --collect-only``; without that, the gate would have
    to pass the flag to see whether the subset exists, and a subset that
    vanished would look exactly like one that was deselected.
    """
    if config.getoption("run_cpu_parity", default=False):
        return True
    return CPU_PARITY_MARKER in (config.getoption("markexpr", default="") or "")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _asked_for_the_subset(config):
        return
    deselected = [
        item for item in items if item.get_closest_marker(CPU_PARITY_MARKER) is not None
    ]
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item not in set(deselected)]


@pytest.fixture
def parity_fixture_store() -> FixtureStore:
    """The shipped manifests against the configured fixture root."""
    return FixtureStore(root=fixtures_root())


@pytest.fixture
def parity_case(
    parity_fixture_store: FixtureStore,
) -> Callable[..., ResolvedCase]:
    """Resolve one manifest entry to verified paths.

    Called from the test body rather than requested as a value, so a missing or
    stale fixture is reported as a failure of that test instead of a setup
    error attributed to the fixture.
    """

    def resolve(port: str, case: str, *, tier: str) -> ResolvedCase:
        return parity_fixture_store.case(port, case, tier=tier)

    return resolve
