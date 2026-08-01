"""Torch-free native archive for Chai reference conformers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

_FORMAT = "chai-jax-conformers"
# Version 2 packs every residue's arrays into four contiguous blocks. Version 1
# stored four npz members per residue, which for the released asset meant
# 172,700 members: reading one costs a seek, a header parse, and a zipfile
# handle, so loading took ~203 s and dominated Chai's runtime. Version 1
# archives are still readable so an existing one keeps working.
_VERSION = 2
_LEGACY_VERSION = 1
_MANIFEST = "__chai_jax_conformer_manifest__"


class ConformerData(NamedTuple):
    position: np.ndarray
    element: np.ndarray
    charge: np.ndarray
    atom_names: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]
    symmetries: np.ndarray


def _array_metadata(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(memoryview(contiguous)).hexdigest(),
    }


def _validate(name: str, conformer: ConformerData) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("conformer residue names must be non-empty strings")
    atom_count = conformer.position.shape[0]
    if conformer.position.shape != (atom_count, 3):
        raise ValueError(f"invalid conformer position shape: {name}")
    if (
        conformer.element.shape != (atom_count,)
        or conformer.charge.shape != (atom_count,)
        or len(conformer.atom_names) != atom_count
        or conformer.symmetries.ndim != 2
        or conformer.symmetries.shape[0] != atom_count
    ):
        raise ValueError(f"conformer atom count mismatch: {name}")
    if any(
        left < 0 or right < 0 or left >= atom_count or right >= atom_count
        for left, right in conformer.bonds
    ):
        raise ValueError(f"conformer bond index is out of range: {name}")


def save_native_conformers(
    conformers: Mapping[str, ConformerData],
    path: str | Path,
    *,
    asset_version: str,
    source_sha256: str,
) -> None:
    """Atomically save checksummed conformers without object arrays."""
    if not conformers:
        raise ValueError("at least one conformer is required")
    if not asset_version:
        raise ValueError("asset_version is required")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")

    names = sorted(conformers)
    positions, elements, charges, symmetries = [], [], [], []
    atom_counts, symmetry_widths, atom_names, bonds = [], [], [], []
    # Only `symmetries` varies in dtype across the released archive (1,880 of
    # 43,175 are int32, the rest int64). Concatenating promotes them all, so the
    # original is recorded and restored rather than silently widened.
    symmetry_dtypes = []
    for name in names:
        conformer = conformers[name]
        _validate(name, conformer)
        position = np.ascontiguousarray(conformer.position)
        symmetry = np.ascontiguousarray(conformer.symmetries)
        positions.append(position)
        elements.append(np.ascontiguousarray(conformer.element))
        charges.append(np.ascontiguousarray(conformer.charge))
        # Ragged in its second axis, so it is stored flat and reshaped on load.
        symmetries.append(symmetry.reshape(-1))
        atom_counts.append(int(position.shape[0]))
        symmetry_widths.append(int(symmetry.shape[1]))
        symmetry_dtypes.append(symmetry.dtype.str)
        atom_names.append(list(conformer.atom_names))
        bonds.append([list(bond) for bond in conformer.bonds])

    arrays = {
        "position": np.concatenate(positions, axis=0),
        "element": np.concatenate(elements, axis=0),
        "charge": np.concatenate(charges, axis=0),
        "symmetries": np.concatenate(symmetries, axis=0),
    }
    residue_manifest = {
        "names": names,
        "atom_counts": atom_counts,
        "symmetry_widths": symmetry_widths,
        "symmetry_dtypes": symmetry_dtypes,
        "atom_names": atom_names,
        "bonds": bonds,
        "arrays": {key: _array_metadata(value) for key, value in arrays.items()},
    }

    manifest = {
        "format": _FORMAT,
        "version": _VERSION,
        "asset_version": asset_version,
        "source_sha256": source_sha256,
        "packed": residue_manifest,
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        np.savez(
            output,
            **{
                _MANIFEST: np.frombuffer(manifest_bytes, np.uint8),
                **arrays,
            },
        )
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_native_conformers(
    path: str | Path,
    *,
    expected_asset_version: str | None = None,
) -> dict[str, ConformerData]:
    """Load and verify the native conformer archive without PyTorch/RDKit."""
    with np.load(path, allow_pickle=False) as archive:
        if _MANIFEST not in archive.files:
            raise ValueError("native conformer manifest is missing")
        try:
            manifest = json.loads(archive[_MANIFEST].tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("native conformer manifest is invalid") from error
        version = manifest.get("version")
        if manifest.get("format") != _FORMAT or version not in (
            _VERSION,
            _LEGACY_VERSION,
        ):
            raise ValueError("unsupported native conformer format")
        if (
            expected_asset_version is not None
            and manifest.get("asset_version") != expected_asset_version
        ):
            raise ValueError(
                "native conformer asset version mismatch: "
                f"expected {expected_asset_version!r}, "
                f"got {manifest.get('asset_version')!r}"
            )
        if version == _VERSION:
            return _read_packed(archive, manifest)
        return _read_legacy(archive, manifest)


def _read_packed(archive, manifest: dict) -> dict[str, ConformerData]:
    """Read the packed layout: four array reads, then slicing."""
    packed = manifest.get("packed")
    if not isinstance(packed, dict) or not packed.get("names"):
        raise ValueError("native conformer residue manifest is invalid")
    if set(archive.files) - {_MANIFEST} != set(packed["arrays"]):
        raise ValueError("native conformer array names mismatch")

    arrays = {}
    for name, expected in packed["arrays"].items():
        value = np.array(archive[name], copy=True)
        if _array_metadata(value) != expected:
            raise ValueError(f"native conformer array metadata mismatch: {name}")
        arrays[name] = value

    names = packed["names"]
    atom_counts = np.asarray(packed["atom_counts"], dtype=np.int64)
    widths = np.asarray(packed["symmetry_widths"], dtype=np.int64)
    atom_bounds = np.concatenate([[0], np.cumsum(atom_counts)])
    symmetry_bounds = np.concatenate([[0], np.cumsum(atom_counts * widths)])
    if atom_bounds[-1] != arrays["position"].shape[0]:
        raise ValueError("native conformer atom counts do not span the archive")
    if symmetry_bounds[-1] != arrays["symmetries"].shape[0]:
        raise ValueError("native conformer symmetries do not span the archive")

    output: dict[str, ConformerData] = {}
    for index, name in enumerate(names):
        start, stop = atom_bounds[index], atom_bounds[index + 1]
        sym_start, sym_stop = symmetry_bounds[index], symmetry_bounds[index + 1]
        conformer = ConformerData(
            position=arrays["position"][start:stop],
            element=arrays["element"][start:stop],
            charge=arrays["charge"][start:stop],
            atom_names=tuple(packed["atom_names"][index]),
            bonds=tuple(tuple(pair) for pair in packed["bonds"][index]),
            symmetries=arrays["symmetries"][sym_start:sym_stop]
            .reshape(int(atom_counts[index]), int(widths[index]))
            .astype(packed["symmetry_dtypes"][index], copy=False),
        )
        _validate(name, conformer)
        output[name] = conformer
    return output


def _read_legacy(archive, manifest: dict) -> dict[str, ConformerData]:
    """Read version 1, which stored four npz members per residue."""
    residues = manifest.get("residues")
    if not isinstance(residues, dict) or not residues:
        raise ValueError("native conformer residue manifest is invalid")
    expected_arrays = {
        f"{metadata['prefix']}.{array_name}"
        for metadata in residues.values()
        for array_name in metadata["arrays"]
    }
    if set(archive.files) - {_MANIFEST} != expected_arrays:
        raise ValueError("native conformer array names mismatch")

    output: dict[str, ConformerData] = {}
    for name, metadata in residues.items():
        prefix = metadata["prefix"]
        values = {}
        for array_name, expected in metadata["arrays"].items():
            value = np.array(archive[f"{prefix}.{array_name}"], copy=True)
            if _array_metadata(value) != expected:
                raise ValueError(
                    f"native conformer array metadata mismatch: {name}.{array_name}"
                )
            values[array_name] = value
        conformer = ConformerData(
            position=values["position"],
            element=values["element"],
            charge=values["charge"],
            atom_names=tuple(metadata["atom_names"]),
            bonds=tuple(tuple(pair) for pair in metadata["bonds"]),
            symmetries=values["symmetries"],
        )
        _validate(name, conformer)
        output[name] = conformer
    return output
