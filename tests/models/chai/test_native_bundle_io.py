from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.bridge import bundle_io


def _state(count: int, offset: int) -> dict[str, np.ndarray]:
    return {
        f"tensor_{index:04d}": np.asarray([offset + index], dtype=np.float32)
        for index in range(count)
    }


@pytest.fixture()
def source_paths(tmp_path: Path) -> dict[str, Path]:
    sources = {}
    for index, component in enumerate(bundle_io.COMPONENT_TENSOR_COUNTS):
        path = tmp_path / f"{component}.pt"
        path.write_bytes(f"official-{index}-{component}".encode())
        sources[component] = path
    return sources


@pytest.fixture()
def native_bundle(tmp_path: Path, source_paths, monkeypatch) -> Path:
    states = {
        component: _state(count, index * 10_000)
        for index, (component, count) in enumerate(
            bundle_io.COMPONENT_TENSOR_COUNTS.items()
        )
    }

    def fake_load(path):
        return states[Path(path).stem]

    monkeypatch.setattr(bundle_io, "load_component_state_dict", fake_load)
    destination = tmp_path / "chai-native"
    bundle_io.export_native_bundle(
        source_paths,
        destination,
        chai_source="chai-lab/chai-1",
        chai_release="v2",
    )
    return destination


def test_export_and_load_strict_six_component_bundle(
    native_bundle: Path, source_paths
) -> None:
    loaded = bundle_io.load_native_bundle(native_bundle)
    assert tuple(loaded.components) == tuple(bundle_io.COMPONENT_TENSOR_COUNTS)
    assert loaded.manifest["schema_version"] == bundle_io.BUNDLE_SCHEMA_VERSION
    assert loaded.manifest["chai"] == {
        "source": "chai-lab/chai-1",
        "release": "v2",
    }
    assert {path.name for path in native_bundle.iterdir()} == {
        bundle_io.BUNDLE_MANIFEST_FILENAME,
        *(f"{name}.npz" for name in bundle_io.COMPONENT_TENSOR_COUNTS),
    }
    for name, expected_count in bundle_io.COMPONENT_TENSOR_COUNTS.items():
        assert len(loaded.components[name]) == expected_count
        metadata = loaded.manifest["components"][name]
        assert metadata["tensor_count"] == expected_count
        assert (
            metadata["source_sha256"]
            == hashlib.sha256(source_paths[name].read_bytes()).hexdigest()
        )
        assert (
            metadata["native_archive_sha256"]
            == hashlib.sha256((native_bundle / f"{name}.npz").read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("missing", ["feature_embedding", "diffusion_module"])
def test_load_rejects_missing_component(native_bundle: Path, missing: str) -> None:
    (native_bundle / f"{missing}.npz").unlink()
    with pytest.raises(ValueError, match="bundle files mismatch"):
        bundle_io.load_native_bundle(native_bundle)


def test_load_rejects_extra_component(native_bundle: Path) -> None:
    (native_bundle / "unexpected.npz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="bundle files mismatch"):
        bundle_io.load_native_bundle(native_bundle)


def test_load_rejects_swapped_component_archives(native_bundle: Path) -> None:
    first = native_bundle / "feature_embedding.npz"
    second = native_bundle / "bond_loss_input_proj.npz"
    first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
    first.write_bytes(second_bytes)
    second.write_bytes(first_bytes)
    with pytest.raises(ValueError, match="corrupt or swapped"):
        bundle_io.load_native_bundle(native_bundle)


def test_load_rejects_corrupt_archive(native_bundle: Path) -> None:
    archive = native_bundle / "confidence_head.npz"
    data = bytearray(archive.read_bytes())
    data[-1] ^= 0xFF
    archive.write_bytes(data)
    with pytest.raises(ValueError, match="corrupt or swapped"):
        bundle_io.load_native_bundle(native_bundle)


def test_load_rejects_manifest_component_count_tampering(native_bundle: Path) -> None:
    manifest_path = native_bundle / bundle_io.BUNDLE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["components"]["trunk"]["tensor_count"] -= 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tensor count declaration"):
        bundle_io.load_native_bundle(native_bundle)


def test_export_rejects_missing_extra_and_wrong_tensor_counts(
    tmp_path: Path, source_paths, monkeypatch
) -> None:
    missing = dict(source_paths)
    missing.pop("trunk")
    with pytest.raises(ValueError, match="source components mismatch"):
        bundle_io.export_native_bundle(
            missing,
            tmp_path / "missing",
            chai_source="source",
            chai_release="release",
        )

    extra = {**source_paths, "extra": tmp_path / "extra.pt"}
    with pytest.raises(ValueError, match="source components mismatch"):
        bundle_io.export_native_bundle(
            extra,
            tmp_path / "extra",
            chai_source="source",
            chai_release="release",
        )

    monkeypatch.setattr(
        bundle_io,
        "load_component_state_dict",
        lambda path: _state(
            bundle_io.COMPONENT_TENSOR_COUNTS[Path(path).stem]
            - (Path(path).stem == "token_embedder"),
            0,
        ),
    )
    with pytest.raises(ValueError, match="token_embedder tensor count"):
        bundle_io.export_native_bundle(
            source_paths,
            tmp_path / "wrong-count",
            chai_source="source",
            chai_release="release",
        )


def test_export_refuses_existing_destination(tmp_path: Path, source_paths) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        bundle_io.export_native_bundle(
            source_paths,
            destination,
            chai_source="source",
            chai_release="release",
        )


def test_runtime_load_is_torch_free_in_subprocess(native_bundle: Path) -> None:
    script = """
import builtins
import sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('runtime bundle load imported torch')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from foldjax.models.chai.bridge.bundle_io import load_native_bundle
bundle = load_native_bundle(sys.argv[1])
assert len(bundle.components) == 6
assert 'torch' not in sys.modules
"""
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, env.get("PYTHONPATH")))
    )
    subprocess.run(
        [sys.executable, "-c", script, str(native_bundle)],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )


def test_loader_does_not_reference_torch(monkeypatch, native_bundle: Path) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch import attempted")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert len(bundle_io.load_native_bundle(native_bundle).components) == 6
