"""The manifest loader is the only thing that enforces the fixture schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ._manifest import (
    ManifestError,
    entry_for,
    load_all_manifests,
    load_manifest,
    manifest_paths,
    parse_manifest,
)
from .conftest import example_manifest


def _write(directory: Path, payload: dict[str, Any], port: str = "demo") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{port}.json"
    path.write_text(json.dumps(payload))
    return path


def _parse(payload: dict[str, Any]) -> Any:
    return parse_manifest(payload, source="demo.json", expected_port="demo")


def test_a_complete_entry_parses_into_its_fields() -> None:
    entry = _parse(example_manifest()).entries[0]
    assert entry.key == ("demo", "demo_case", "A")
    assert entry.tokens == 76
    assert entry.tolerance_value == pytest.approx(0.05)
    assert entry.cpu_source_commit == "0c6d059"
    assert entry.files["trunk.npz"].size_bytes == 8
    assert entry.capture_dir == Path(
        "/captures/demo-native-20260909/demo_case/native-A"
    )


def test_an_unknown_top_level_key_is_rejected() -> None:
    payload = example_manifest()
    payload["fixtures"] = {}
    with pytest.raises(ManifestError, match="unknown key"):
        _parse(payload)


def test_a_typo_inside_an_entry_is_rejected_rather_than_ignored() -> None:
    payload = example_manifest()
    payload["entries"][0]["tolerences"] = {}
    with pytest.raises(ManifestError, match="unknown key"):
        _parse(payload)


def test_a_future_schema_version_is_refused() -> None:
    payload = example_manifest()
    payload["schema_version"] = 2
    with pytest.raises(ManifestError, match="this loader speaks 1"):
        _parse(payload)


def test_a_tier_outside_the_two_documented_ones_is_refused() -> None:
    payload = example_manifest(tier="C")
    with pytest.raises(ManifestError, match="tier"):
        _parse(payload)


def test_a_digest_that_is_not_sha256_is_refused() -> None:
    payload = example_manifest(files={"trunk.npz": {"sha256": "abc", "bytes": 8}})
    with pytest.raises(ManifestError, match="64 lowercase hex"):
        _parse(payload)


def test_a_zero_byte_fixture_is_refused() -> None:
    payload = example_manifest(files={"trunk.npz": {"sha256": "0" * 64, "bytes": 0}})
    with pytest.raises(ManifestError, match="bytes"):
        _parse(payload)


def test_a_file_name_with_a_path_separator_is_refused() -> None:
    payload = example_manifest(files={"a/trunk.npz": {"sha256": "0" * 64, "bytes": 8}})
    with pytest.raises(ManifestError, match="plain file name"):
        _parse(payload)


def test_a_relative_capture_path_is_refused() -> None:
    payload = example_manifest(capture={"provenance": "captures/x.json", "commit": "a"})
    with pytest.raises(ManifestError, match="absolute path"):
        _parse(payload)


def test_an_empty_tripwire_is_refused() -> None:
    payload = example_manifest(tripwire={})
    with pytest.raises(ManifestError, match="at least one key"):
        _parse(payload)


def test_a_zero_tolerance_is_refused() -> None:
    """CPU XLA is a third implementation; nothing here is bitwise."""
    payload = example_manifest(
        tolerance={"metric": "rmsd_angstrom", "value": 0.0, "set_from": "wishful"}
    )
    with pytest.raises(ManifestError, match="must be > 0"):
        _parse(payload)


def test_a_tolerance_below_its_own_calibration_is_refused() -> None:
    payload = example_manifest(
        tolerance={"metric": "rmsd_angstrom", "value": 0.01, "set_from": "GPU residual"}
    )
    with pytest.raises(ManifestError, match="below its own CPU calibration"):
        _parse(payload)


def test_an_entry_without_a_cpu_calibration_is_refused() -> None:
    payload = example_manifest()
    del payload["entries"][0]["cpu_calibration"]
    with pytest.raises(ManifestError, match="cpu_calibration"):
        _parse(payload)


def test_a_calibration_without_a_source_commit_is_refused() -> None:
    payload = example_manifest(
        cpu_calibration={
            "residual_angstrom": 0.02,
            "wall_seconds": 91.0,
            "source_commit": "",
        }
    )
    with pytest.raises(ManifestError, match="source_commit"):
        _parse(payload)


def test_a_node_id_outside_the_subset_is_refused() -> None:
    payload = example_manifest(test_node_ids=["tests/models/test_x.py::test_y"])
    with pytest.raises(ManifestError, match="node id"):
        _parse(payload)


def test_two_entries_for_one_case_and_tier_are_refused() -> None:
    payload = example_manifest()
    payload["entries"].append(dict(payload["entries"][0]))
    with pytest.raises(ManifestError, match="repeats case"):
        _parse(payload)


def test_the_same_case_at_both_tiers_is_allowed() -> None:
    payload = example_manifest()
    second = dict(payload["entries"][0])
    second["tier"] = "B"
    payload["entries"].append(second)
    assert [entry.tier for entry in _parse(payload).entries] == ["A", "B"]


def test_the_port_must_match_the_file_it_is_stored_in(tmp_path: Path) -> None:
    _write(tmp_path, example_manifest(port="protenix"), port="demo")
    with pytest.raises(ManifestError, match="but the file is demo.json"):
        load_manifest("demo", tmp_path)


def test_a_missing_port_manifest_names_the_path_it_looked_for(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="no CPU-parity manifest"):
        load_manifest("demo", tmp_path)


def test_invalid_json_is_reported_as_such(tmp_path: Path) -> None:
    (tmp_path / "demo.json").write_text("{")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest("demo", tmp_path)


def test_the_readme_beside_the_manifests_is_not_read_as_one(tmp_path: Path) -> None:
    _write(tmp_path, example_manifest())
    (tmp_path / "README.md").write_text("# schema")
    assert [path.name for path in manifest_paths(tmp_path)] == ["demo.json"]
    assert [manifest.port for manifest in load_all_manifests(tmp_path)] == ["demo"]


def test_entry_for_lists_what_the_port_does_have(tmp_path: Path) -> None:
    _write(tmp_path, example_manifest())
    assert entry_for("demo", "demo_case", "A", tmp_path).tokens == 76
    with pytest.raises(ManifestError, match="demo_case/tier-A"):
        entry_for("demo", "other", "A", tmp_path)
