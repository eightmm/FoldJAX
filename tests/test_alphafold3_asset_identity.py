"""Persistent AF3 tuning identities describe bytes, not an NFS mount."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foldjax.backends import _tokamax_autotune as tuning
from foldjax.backends import alphafold3 as backend
from foldjax.models.alphafold3 import build


@pytest.fixture
def assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "af3.bin").write_bytes(b"parameter payload")
    runner = tmp_path / "runner.py"
    runner.write_bytes(b"runner payload")
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "model.py").write_bytes(b"model payload")
    (source / "nested" / "kernel.py").write_bytes(b"kernel payload")
    monkeypatch.setattr(backend, "VENDORED_RUNNER", runner)
    monkeypatch.setattr(build, "source_package", lambda: source)
    return {"weights": weights, "runner": runner, "source": source}


def _source_key(weights: Path) -> str:
    snapshot = backend._managed_content_snapshot(weights)
    assert snapshot is not None
    # This is the source_sha256 construction used by Tokamax's runtime identity.
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize("field", ("device", "inode", "ctime", "mode", "mtime"))
def test_persistent_identity_ignores_node_local_stat_fields(
    assets: dict[str, Path], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    weights = assets["weights"]
    before = _source_key(weights)
    anchor = backend._managed_asset_snapshot(weights)
    original = backend.path_stat_identity
    field_index = {"mode": 0, "device": 1, "inode": 2, "mtime": 4, "ctime": 5}

    def changed_stat(path: Path) -> dict[str, Any] | None:
        identity = copy.deepcopy(original(path))
        if identity is not None:
            identity["stat_signature"]["root"][field_index[field]] += 1
        return identity

    monkeypatch.setattr(backend, "path_stat_identity", changed_stat)

    assert backend._managed_asset_snapshot(weights) != anchor
    assert _source_key(weights) == before


def test_persistent_identity_survives_a_different_install_path(
    assets: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _source_key(assets["weights"])
    relocated = tmp_path / "another-node"
    relocated.mkdir()
    shutil.copytree(assets["weights"], relocated / "weights")
    shutil.copytree(assets["source"], relocated / "source")
    shutil.copyfile(assets["runner"], relocated / "runner.py")
    monkeypatch.setattr(backend, "VENDORED_RUNNER", relocated / "runner.py")
    monkeypatch.setattr(build, "source_package", lambda: relocated / "source")

    assert _source_key(relocated / "weights") == before


def test_tokamax_runtime_identity_uses_the_portable_source_digest(
    assets: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tuning, "_tokamax_api", lambda: SimpleNamespace(__version__="test")
    )
    monkeypatch.setattr(tuning, "_distribution_version", lambda _name: "test")
    monkeypatch.setattr(tuning, "_platform_version", lambda _name: "test")
    identity = tuning._runtime_identity(
        device=SimpleNamespace(platform="gpu", device_kind="mock-gpu"),
        source_snapshot=backend._managed_content_snapshot(assets["weights"]),
        config_identity=(),
        buckets=(256,),
        padding=True,
    )

    assert identity["source_sha256"] == _source_key(assets["weights"])
    assert str(assets["weights"]) not in json.dumps(identity)


@pytest.mark.parametrize("asset", ("weights", "runner", "source"))
def test_persistent_identity_changes_with_asset_content(
    assets: dict[str, Path], asset: str
) -> None:
    before = _source_key(assets["weights"])
    target = {
        "weights": assets["weights"] / "af3.bin",
        "runner": assets["runner"],
        "source": assets["source"] / "nested" / "kernel.py",
    }[asset]
    info = target.stat()
    payload = target.read_bytes()
    replacement = target.with_name(target.name + ".replacement")
    replacement.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
    # Replacing avoids assuming that the filesystem advances ctime between
    # consecutive writes; size and mtime remain identical deliberately.
    replacement.replace(target)

    assert target.stat().st_size == info.st_size
    assert _source_key(assets["weights"]) != before


def test_source_identity_binds_relative_file_names(assets: dict[str, Path]) -> None:
    before = _source_key(assets["weights"])
    source = assets["source"]
    (source / "nested" / "kernel.py").rename(source / "nested" / "renamed.py")

    assert _source_key(assets["weights"]) != before


def test_generated_bytecode_does_not_change_source_identity(
    assets: dict[str, Path],
) -> None:
    before = _source_key(assets["weights"])
    source = assets["source"]
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "model.cpython-313.pyc").write_bytes(b"cache")
    (source / "nested" / "legacy.pyc").write_bytes(b"cache")
    (source / "nested" / "legacy.pyo").write_bytes(b"cache")

    assert _source_key(assets["weights"]) == before


def test_parameter_hash_is_reused_while_other_assets_change(
    assets: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: Counter[Path] = Counter()
    original = backend.sha256_file

    def counted_hash(path: Path) -> str:
        calls[path] += 1
        return original(path)

    monkeypatch.setattr(backend, "sha256_file", counted_hash)
    weights = assets["weights"]
    first = _source_key(weights)
    assert _source_key(weights) == first
    (assets["source"] / "nested" / "kernel.py").write_bytes(b"changed source")
    assert _source_key(weights) != first

    assert calls[weights / "af3.bin"] == 1
    assert calls[assets["source"] / "nested" / "kernel.py"] == 2


def test_change_while_hashing_does_not_claim_a_stable_identity(
    assets: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = backend.sha256_file
    target = assets["weights"] / "af3.bin"

    def changing_hash(path: Path) -> str:
        result = original(path)
        if path == target:
            path.write_bytes(b"changed during hashing")
        return result

    monkeypatch.setattr(backend, "sha256_file", changing_hash)

    assert backend._managed_content_snapshot(assets["weights"]) is None


def test_change_to_an_earlier_asset_while_hashing_another_is_rejected(
    assets: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = backend.sha256_file

    def changing_hash(path: Path) -> str:
        result = original(path)
        if path == assets["runner"]:
            (assets["weights"] / "af3.bin").write_bytes(b"changed after weight hash")
        return result

    monkeypatch.setattr(backend, "sha256_file", changing_hash)

    assert backend._managed_content_snapshot(assets["weights"]) is None


def test_missing_parameter_bundle_has_no_portable_identity(
    assets: dict[str, Path],
) -> None:
    (assets["weights"] / "af3.bin").unlink()

    assert backend._managed_content_snapshot(assets["weights"]) is None
