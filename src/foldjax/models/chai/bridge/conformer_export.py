"""One-time bridge from Chai's antipickle conformers to native assets."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models.chai.bridge.antipickle_io import load as load_antipickle
from foldjax.models.chai.bridge.conformer_io import (
    ConformerData,
    save_native_conformers,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    """Detach a Torch tensor while also allowing array-like bridge fixtures."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


@dataclass(frozen=True, slots=True)
class _RawConformer:
    """Chai's ``ConformerData`` as it appears in the archive.

    Field names are the archive's, so the decoder can rebuild each record
    without importing Chai. Arrays arrive as numpy because Chai's own torch
    adapter serialises tensors through numpy.
    """

    position: Any
    element: Any
    charge: Any
    atom_names: Any
    bonds: Any
    symmetries: Any


def _load_via_chai_lab(chai_source_directory: str | Path, source_path: Path) -> Any:
    """Decode with Chai's own adapters, for verifying the native reader."""
    checkout = Path(chai_source_directory).resolve()
    if not (checkout / "chai_lab/data/sources").is_dir():
        raise ValueError(
            "chai_source_directory must be a Chai checkout containing chai_lab"
        )
    checkout_text = str(checkout)
    sys.path.insert(0, checkout_text)
    try:
        antipickle = importlib.import_module("antipickle")
        rdkit_source = importlib.import_module("chai_lab.data.sources.rdkit")
        return antipickle.load(source_path, adapters=rdkit_source._get_adapters())
    finally:
        if sys.path and sys.path[0] == checkout_text:
            sys.path.pop(0)


def export_native_conformers(
    source: str | Path,
    destination: str | Path,
    *,
    chai_source_directory: str | Path | None = None,
    asset_version: str = "conformers_v1",
) -> int:
    """Export Chai's official conformer antipickle into the native archive.

    The archive is decoded directly (see :mod:`antipickle_io`), so neither
    torch, antipickle, nor a Chai checkout is required. Passing
    ``chai_source_directory`` falls back to Chai's own adapters, which is useful
    only for cross-checking this reader against upstream.
    """
    source_path = Path(source).resolve()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"conformer source is not a regular file: {source_path}")

    if chai_source_directory is None:
        raw = load_antipickle(source_path, dataclasses={"conf": _RawConformer})
    else:
        raw = _load_via_chai_lab(chai_source_directory, source_path)

    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("official conformer asset must contain a non-empty mapping")
    conformers = {
        name: ConformerData(
            position=_to_numpy(value.position),
            element=_to_numpy(value.element),
            charge=_to_numpy(value.charge),
            atom_names=tuple(value.atom_names),
            bonds=tuple(tuple(pair) for pair in value.bonds),
            symmetries=_to_numpy(value.symmetries),
        )
        for name, value in raw.items()
    }
    save_native_conformers(
        conformers,
        destination,
        asset_version=asset_version,
        source_sha256=_sha256_file(source_path),
    )
    return len(conformers)
