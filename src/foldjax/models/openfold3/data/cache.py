"""Public, pickle-free preprocessing-cache writers for OpenFold3."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

_DATABASE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def _npz_target(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _write_npz_atomic(target: Path, payload: Mapping[str, np.ndarray]) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}-",
        suffix=".npz",
        dir=target.parent,
        delete=False,
    ) as temporary:
        staged = Path(temporary.name)
    try:
        np.savez_compressed(staged, **payload)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return target


def _msa_character_matrix(value: Any, *, database: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"U", "S"}:
        raise TypeError(
            f"MSA {database!r} sequences must be strings, not {array.dtype}"
        )
    if array.ndim == 1:
        if array.dtype.kind == "S":
            array = np.char.decode(array, "utf-8")
        sequences = [str(sequence) for sequence in array]
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError(f"MSA {database!r} sequences must have equal length")
        array = np.asarray([list(sequence) for sequence in sequences], dtype="U1")
    elif array.ndim == 2:
        array = array.astype("U1", copy=False)
    else:
        raise ValueError(f"MSA {database!r} must be a sequence list or 2D array")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError(f"MSA {database!r} cannot be empty")
    return array


def save_preparsed_msas(
    msas: Mapping[str, Mapping[str, Any]], path: str | Path
) -> Path:
    """Write MSA arrays in FoldJAX's safe pre-parsed ``.npz`` format.

    Each database maps to ``msa`` (either equal-length sequence strings or a
    character matrix), an integer ``deletion_matrix`` of the same shape, and an
    optional one-dimensional string ``metadata`` array. The output never contains
    object arrays and is loaded with ``allow_pickle=False``.
    """
    if not msas:
        raise ValueError("at least one MSA database is required")
    payload: dict[str, np.ndarray] = {}
    for database, values in msas.items():
        if not isinstance(database, str) or not _DATABASE_NAME.fullmatch(database):
            raise ValueError(
                f"MSA database name {database!r} may contain only letters, numbers, "
                "underscore, dot, and hyphen"
            )
        if not isinstance(values, Mapping):
            raise TypeError(f"MSA {database!r} must be a mapping of arrays")
        if "msa" not in values or "deletion_matrix" not in values:
            raise ValueError(
                f"MSA {database!r} requires 'msa' and 'deletion_matrix'"
            )
        msa = _msa_character_matrix(values["msa"], database=database)
        deletion = np.asarray(values["deletion_matrix"])
        if deletion.dtype.kind not in {"i", "u"} or deletion.shape != msa.shape:
            raise ValueError(
                f"MSA {database!r} deletion_matrix must be an integer array with "
                f"shape {msa.shape}"
            )
        metadata = np.asarray(values.get("metadata", [""] * msa.shape[0]))
        if (
            metadata.ndim != 1
            or metadata.shape[0] != msa.shape[0]
            or metadata.dtype.kind not in {"U", "S"}
        ):
            raise ValueError(
                f"MSA {database!r} metadata must contain one string per row"
            )
        payload[f"{database}__msa"] = msa
        payload[f"{database}__deletion_matrix"] = deletion
        payload[f"{database}__metadata"] = metadata.astype(str, copy=False)
    return _write_npz_atomic(_npz_target(path), payload)


def _template_value(entry: Mapping[str, Any], name: str, default: Any = None) -> Any:
    return entry[name] if name in entry else default


def save_template_cache(
    entries: Mapping[str, Mapping[str, Any]], path: str | Path
) -> Path:
    """Write a pickle-free local-template cache.

    Every key is ``<entry_id>_<chain_id>``. Each value requires an integer
    ``idx_map`` of 1-based ``(query_residue, template_residue)`` pairs and may
    provide ``release_date`` and ``cif_path``. Entries are normalized to manifest
    order; relative CIF paths remain relative to the cache file.
    """
    if not entries:
        raise ValueError("at least one template cache entry is required")
    normalized: dict[str, dict[str, Any]] = {}
    for index, (template_id, entry) in enumerate(entries.items()):
        if (
            not isinstance(template_id, str)
            or template_id.count("_") != 1
            or any(not part for part in template_id.split("_"))
        ):
            raise ValueError(
                f"template ID {template_id!r} must be '<entry_id>_<chain_id>'"
            )
        if not isinstance(entry, Mapping):
            raise TypeError(f"template cache entry {template_id!r} must be a mapping")
        if "idx_map" not in entry:
            raise ValueError(f"template cache entry {template_id!r} lacks idx_map")
        index_map = np.asarray(entry["idx_map"])
        if index_map.dtype.kind not in {"i", "u"} or (
            index_map.ndim != 2 or index_map.shape[1:] != (2,)
        ):
            raise ValueError(
                f"template cache entry {template_id!r} idx_map must be an integer "
                "array with shape (N, 2)"
            )
        index_map = index_map[(index_map > 0).all(axis=1)]
        if not len(index_map):
            raise ValueError(
                f"template cache entry {template_id!r} has no aligned residue pairs"
            )
        release_date = _template_value(entry, "release_date", "")
        if isinstance(release_date, datetime | date):
            release_date = release_date.isoformat()
        if not isinstance(release_date, str):
            raise TypeError(
                f"template cache entry {template_id!r} release_date must be a string"
            )
        value: dict[str, Any] = {
            "index": index,
            "release_date": release_date,
            "idx_map": index_map.astype(np.int64, copy=False).tolist(),
        }
        cif_path = _template_value(entry, "cif_path")
        if cif_path is not None:
            value["cif_path"] = str(Path(cif_path))
        normalized[template_id] = value
    payload = {
        "entries_json": np.asarray(
            json.dumps(normalized, separators=(",", ":"), sort_keys=False)
        )
    }
    return _write_npz_atomic(_npz_target(path), payload)
