"""`python -m tests.parity.fetch` copies by content, out of a read-only capture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from .conftest import example_manifest
from .fetch import main

PAYLOAD = b"stored native capture"
DECOY = b"X" * len(PAYLOAD)


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    """A capture directory: the wanted file under a name of its own, plus noise."""
    root = tmp_path / "capture" / "demo_case" / "native-A"
    root.mkdir(parents=True)
    (root / "trunk-00.npz").write_bytes(PAYLOAD)
    (root / "same-size-decoy.npz").write_bytes(DECOY)
    (root / "checkpoint").mkdir()
    (root / "checkpoint" / "weights.bin").write_bytes(b"w" * 4096)
    return root


@pytest.fixture
def manifest_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "manifest"
    directory.mkdir()
    spec = {"sha256": hashlib.sha256(PAYLOAD).hexdigest(), "bytes": len(PAYLOAD)}
    (directory / "demo.json").write_text(
        json.dumps(example_manifest(files={"trunk.npz": spec}))
    )
    return directory


def _argv(capture: Path, manifest_dir: Path, dest: Path, *extra: str) -> list[str]:
    return [
        "--from",
        str(capture),
        "--manifest-dir",
        str(manifest_dir),
        "--dest",
        str(dest),
        *extra,
    ]


def test_a_renamed_file_is_found_by_its_digest(
    capture: Path, manifest_dir: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "fixtures"
    assert main(_argv(capture, manifest_dir, dest)) == 0
    assert (dest / "demo" / "demo_case" / "trunk.npz").read_bytes() == PAYLOAD


def test_the_capture_directory_is_left_exactly_as_it_was(
    capture: Path, manifest_dir: Path, tmp_path: Path
) -> None:
    before = sorted(path.name for path in capture.rglob("*"))
    main(_argv(capture, manifest_dir, tmp_path / "fixtures"))
    assert sorted(path.name for path in capture.rglob("*")) == before


def test_a_second_run_reports_the_file_as_present(
    capture: Path,
    manifest_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dest = tmp_path / "fixtures"
    main(_argv(capture, manifest_dir, dest))
    capsys.readouterr()
    assert main(_argv(capture, manifest_dir, dest)) == 0
    assert "present  demo/demo_case/trunk.npz" in capsys.readouterr().out


def test_a_file_the_capture_does_not_hold_exits_nonzero(
    manifest_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(_argv(empty, manifest_dir, tmp_path / "fixtures")) == 1
    captured = capsys.readouterr()
    assert "NOT FOUND demo/demo_case/trunk.npz" in captured.err
    assert "provenance.json" in captured.err


def test_a_dry_run_writes_nothing(
    capture: Path, manifest_dir: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "fixtures"
    assert main(_argv(capture, manifest_dir, dest, "--dry-run")) == 0
    assert not (dest / "demo" / "demo_case" / "trunk.npz").exists()


def test_a_source_that_is_not_a_directory_is_refused(
    manifest_dir: Path, tmp_path: Path
) -> None:
    assert main(_argv(tmp_path / "nowhere", manifest_dir, tmp_path / "fixtures")) == 1


def test_selecting_another_port_selects_nothing_and_says_so(
    capture: Path, manifest_dir: Path, tmp_path: Path
) -> None:
    argv = _argv(capture, manifest_dir, tmp_path / "fixtures", "--port", "protenix")
    assert main(argv) == 1


def test_no_partial_file_survives_a_completed_fetch(
    capture: Path, manifest_dir: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "fixtures"
    main(_argv(capture, manifest_dir, dest))
    assert [path.name for path in (dest / "demo" / "demo_case").iterdir()] == [
        "trunk.npz"
    ]
