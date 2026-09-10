"""A fixture is handed to a test only after it matches the manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ._fixtures import DEFAULT_ROOT, ENV_VAR, FixtureStore, check_file, fixtures_root
from .conftest import example_manifest, write_fixture

#: `pytest.fail` raises this; a missing fixture must land here and not in a skip.
Failed = pytest.fail.Exception

PAYLOAD = b"stored native capture"


def _store(tmp_path: Path, **entry_overrides: object) -> tuple[FixtureStore, Path]:
    case_dir = tmp_path / "fixtures" / "demo" / "demo_case"
    spec = write_fixture(case_dir, "trunk.npz", PAYLOAD)
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = example_manifest(files={"trunk.npz": spec}, **entry_overrides)
    (manifest_dir / "demo.json").write_text(json.dumps(payload))
    store = FixtureStore(root=tmp_path / "fixtures", manifest_dir=manifest_dir)
    return store, case_dir


def test_the_root_comes_from_the_environment_or_the_documented_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert fixtures_root() == DEFAULT_ROOT
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert fixtures_root() == tmp_path


def test_a_matching_file_resolves_to_its_path(tmp_path: Path) -> None:
    store, case_dir = _store(tmp_path)
    resolved = store.case("demo", "demo_case", tier="A")
    assert resolved.path("trunk.npz") == case_dir / "trunk.npz"
    assert resolved.entry.tolerance_value == pytest.approx(0.05)


def test_asking_for_a_file_the_manifest_does_not_list_names_what_it_does(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    resolved = store.case("demo", "demo_case", tier="A")
    with pytest.raises(KeyError, match="trunk.npz"):
        resolved.path("tape.npz")


def test_a_missing_file_fails_with_the_fetch_command(tmp_path: Path) -> None:
    store, case_dir = _store(tmp_path)
    (case_dir / "trunk.npz").unlink()
    with pytest.raises(Failed) as error:
        store.case("demo", "demo_case", tier="A")
    message = str(error.value)
    assert "python -m tests.parity.fetch --from " in message
    assert "/captures/demo-native-20260909/demo_case/native-A" in message
    assert ENV_VAR in message


def test_a_truncated_file_fails_instead_of_being_replayed(tmp_path: Path) -> None:
    store, case_dir = _store(tmp_path)
    (case_dir / "trunk.npz").write_bytes(PAYLOAD[:-1])
    with pytest.raises(Failed, match="bytes"):
        store.case("demo", "demo_case", tier="A")


def test_a_same_size_file_from_another_capture_fails_on_its_digest(
    tmp_path: Path,
) -> None:
    store, case_dir = _store(tmp_path)
    (case_dir / "trunk.npz").write_bytes(b"X" * len(PAYLOAD))
    with pytest.raises(Failed, match="sha256"):
        store.case("demo", "demo_case", tier="A")


def test_a_rewritten_file_is_re_hashed_rather_than_served_from_the_cache(
    tmp_path: Path,
) -> None:
    store, case_dir = _store(tmp_path)
    store.case("demo", "demo_case", tier="A")
    (case_dir / "trunk.npz").write_bytes(b"X" * len(PAYLOAD))
    with pytest.raises(Failed, match="sha256"):
        store.case("demo", "demo_case", tier="A")


def test_check_file_reports_the_first_thing_that_is_wrong(tmp_path: Path) -> None:
    store, case_dir = _store(tmp_path)
    spec = store.entry("demo", "demo_case", "A").files["trunk.npz"]
    assert check_file(case_dir / "trunk.npz", spec) is None
    assert "is missing" in str(check_file(tmp_path / "absent.npz", spec))


def test_a_tripwire_mismatch_fails_and_says_not_to_re_featurize(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    resolved = store.case("demo", "demo_case", tier="A")
    resolved.assert_tripwire({"tape_schema": "demo-v1", "unrelated": "whatever"})
    with pytest.raises(Failed) as error:
        resolved.assert_tripwire({"tape_schema": "demo-v2"})
    assert "re-featurize" in str(error.value)


def test_a_tripwire_that_compares_nothing_is_itself_a_failure(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    resolved = store.case("demo", "demo_case", tier="A")
    with pytest.raises(Failed, match="compared nothing"):
        resolved.assert_tripwire({"renamed_key": "demo-v1"})
