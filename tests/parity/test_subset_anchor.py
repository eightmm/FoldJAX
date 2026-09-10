"""The one `cpu_parity` test that needs neither weights nor a stored capture.

It exists so the subset is never empty. `test_gate.py` asserts that
`-m cpu_parity` collects something; before the ports land their own replays,
this is what it collects, and what it exercises is the contract those replays
depend on -- resolve a manifest entry, verify the bytes, hand back a path.

Ports do NOT copy this test. They mark their replay `cpu_parity`, ask for the
`parity_case` fixture, and name their tests in their manifest entry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ._fixtures import FixtureStore, fixtures_root
from ._manifest import MANIFEST_DIR, ManifestError
from .conftest import example_manifest, write_fixture

pytestmark = pytest.mark.cpu_parity


def test_the_resolver_contract_ports_depend_on(tmp_path: Path) -> None:
    case_dir = tmp_path / "fixtures" / "demo" / "demo_case"
    spec = write_fixture(case_dir, "trunk.npz", b"stored native capture")
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "demo.json").write_text(
        json.dumps(example_manifest(files={"trunk.npz": spec}))
    )

    store = FixtureStore(root=tmp_path / "fixtures", manifest_dir=manifest_dir)
    resolved = store.case("demo", "demo_case", tier="A")

    assert resolved.path("trunk.npz").is_file()
    assert resolved.entry.tolerance_value > 0.0
    resolved.assert_tripwire({"tape_schema": "demo-v1"})


def test_the_fixture_a_port_asks_for_reads_the_shipped_manifests(
    parity_fixture_store: FixtureStore,
) -> None:
    assert parity_fixture_store.root == fixtures_root()
    with pytest.raises(ManifestError, match=str(MANIFEST_DIR)):
        parity_fixture_store.entry("no_such_port", "no_such_case", "A")
