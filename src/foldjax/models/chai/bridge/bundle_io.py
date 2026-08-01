"""Strict six-component native Chai bundle export and Torch-free loading."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models.chai.bridge.component_io import (
    load_component_state_dict,
    load_native_component_state_dict,
    save_native_component_state_dict,
)

BUNDLE_FORMAT = "chai-jax-native-bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_FILENAME = "manifest.json"

COMPONENT_TENSOR_COUNTS = {
    "feature_embedding": 16,
    "bond_loss_input_proj": 1,
    "token_embedder": 43,
    "trunk": 1398,
    "diffusion_module": 343,
    "confidence_head": 106,
}


@dataclass(frozen=True)
class NativeBundle:
    """Validated bundle metadata and component state dictionaries."""

    manifest: dict[str, Any]
    components: dict[str, dict[str, np.ndarray]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata_value(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_source_mapping(sources: Mapping[str, str | Path]) -> dict[str, Path]:
    expected = set(COMPONENT_TENSOR_COUNTS)
    actual = set(sources)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"source components mismatch: missing={missing}, extra={extra}"
        )

    validated: dict[str, Path] = {}
    for component in COMPONENT_TENSOR_COUNTS:
        path = Path(sources[component])
        expected_filename = f"{component}.pt"
        if path.name != expected_filename:
            raise ValueError(
                f"source component filename mismatch for {component}: "
                f"expected {expected_filename!r}, got {path.name!r}"
            )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source component is not a regular file: {component}")
        validated[component] = path
    return validated


def export_native_bundle(
    sources: Mapping[str, str | Path],
    destination: str | Path,
    *,
    chai_source: str,
    chai_release: str,
) -> dict[str, Any]:
    """Convert exactly six official components into an atomic native bundle.

    ``sources`` is keyed by the extension-free component names in
    :data:`COMPONENT_TENSOR_COUNTS`. Existing destinations are never replaced.
    Torch is used only through the lazy bridge loader during this export path.
    """
    source_paths = _validate_source_mapping(sources)
    chai_source = _validate_metadata_value("chai_source", chai_source)
    chai_release = _validate_metadata_value("chai_release", chai_release)

    destination_path = Path(destination)
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(f"bundle destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
        )
    )

    try:
        component_manifest: dict[str, dict[str, Any]] = {}
        for component, expected_count in COMPONENT_TENSOR_COUNTS.items():
            source_path = source_paths[component]
            source_sha256 = _sha256_file(source_path)
            state = load_component_state_dict(source_path)
            actual_count = len(state)
            if actual_count != expected_count:
                raise ValueError(
                    f"{component} tensor count mismatch: "
                    f"expected {expected_count}, got {actual_count}"
                )

            archive_name = f"{component}.npz"
            archive_path = temporary / archive_name
            save_native_component_state_dict(state, archive_path, component=component)
            component_manifest[component] = {
                "archive": archive_name,
                "tensor_count": expected_count,
                "source_filename": source_path.name,
                "source_sha256": source_sha256,
                "native_archive_sha256": _sha256_file(archive_path),
            }

        manifest: dict[str, Any] = {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "chai": {"source": chai_source, "release": chai_release},
            "components": component_manifest,
        }
        manifest_path = temporary / BUNDLE_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination_path)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _load_and_validate_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / BUNDLE_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid native bundle manifest") from error
    if not isinstance(manifest, dict):
        raise ValueError("invalid native bundle manifest")
    if set(manifest) != {"format", "schema_version", "chai", "components"}:
        raise ValueError("native bundle manifest fields mismatch")
    if manifest["format"] != BUNDLE_FORMAT:
        raise ValueError("invalid native bundle format")
    if manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported native bundle schema: {manifest['schema_version']!r}"
        )

    chai = manifest["chai"]
    if not isinstance(chai, dict) or set(chai) != {"source", "release"}:
        raise ValueError("invalid Chai source/release metadata")
    _validate_metadata_value("chai.source", chai["source"])
    _validate_metadata_value("chai.release", chai["release"])

    components = manifest["components"]
    if not isinstance(components, dict) or set(components) != set(
        COMPONENT_TENSOR_COUNTS
    ):
        raise ValueError("bundle manifest components mismatch")
    expected_fields = {
        "archive",
        "tensor_count",
        "source_filename",
        "source_sha256",
        "native_archive_sha256",
    }
    for component, expected_count in COMPONENT_TENSOR_COUNTS.items():
        metadata = components[component]
        if not isinstance(metadata, dict) or set(metadata) != expected_fields:
            raise ValueError(f"invalid component manifest: {component}")
        if metadata["archive"] != f"{component}.npz":
            raise ValueError(f"component archive mismatch: {component}")
        if metadata["source_filename"] != f"{component}.pt":
            raise ValueError(f"source filename mismatch: {component}")
        if metadata["tensor_count"] != expected_count:
            raise ValueError(f"tensor count declaration mismatch: {component}")
        if not _is_sha256(metadata["source_sha256"]):
            raise ValueError(f"invalid source checksum: {component}")
        if not _is_sha256(metadata["native_archive_sha256"]):
            raise ValueError(f"invalid native archive checksum: {component}")
    return manifest


def load_native_bundle(path: str | Path) -> NativeBundle:
    """Load a fully validated native bundle without importing Torch."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("native bundle path must be a regular directory")
    expected_files = {
        BUNDLE_MANIFEST_FILENAME,
        *(f"{component}.npz" for component in COMPONENT_TENSOR_COUNTS),
    }
    actual_files = {entry.name for entry in root.iterdir()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(f"bundle files mismatch: missing={missing}, extra={extra}")
    for filename in expected_files:
        file_path = root / filename
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"bundle entry is not a regular file: {filename}")

    manifest = _load_and_validate_manifest(root)
    states: dict[str, dict[str, np.ndarray]] = {}
    for component, expected_count in COMPONENT_TENSOR_COUNTS.items():
        archive = root / f"{component}.npz"
        metadata = manifest["components"][component]
        if _sha256_file(archive) != metadata["native_archive_sha256"]:
            raise ValueError(f"native archive is corrupt or swapped: {component}")
        try:
            state = load_native_component_state_dict(
                archive, expected_component=component
            )
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f"invalid native archive: {component}") from error
        if len(state) != expected_count:
            raise ValueError(
                f"{component} tensor count mismatch: "
                f"expected {expected_count}, got {len(state)}"
            )
        states[component] = state
    return NativeBundle(manifest=manifest, components=states)
