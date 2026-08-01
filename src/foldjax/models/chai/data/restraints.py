"""Torch-free parsing and feature generation for Chai manual restraints."""

from __future__ import annotations

import csv
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

_AA_1_TO_3: Final = {
    one: three
    for one, three in zip(
        "ARNDCQEGHILKMFPSTWYV",
        (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ),
        strict=True,
    )
}
_AA_1_TO_3["X"] = "UNK"
_CSV_COLUMNS: Final = {
    "restraint_id",
    "chainA",
    "res_idxA",
    "chainB",
    "res_idxB",
    "connection_type",
    "confidence",
    "min_distance_angstrom",
    "max_distance_angstrom",
    "comment",
}
_CONTACT_KEYS: Final = {
    "left_residue_subchain_id",
    "right_residue_subchain_id",
    "left_residue_index",
    "right_residue_index",
    "left_residue_name",
    "right_residue_name",
    "distance_threshold",
}
_POCKET_KEYS: Final = {
    "pocket_chain_subchain_id",
    "pocket_token_subchain_id",
    "pocket_token_residue_index",
    "pocket_token_residue_name",
    "pocket_distance_threshold",
}
_DOCKING_KEYS: Final = {
    "subchain_ids",
    "noise_sigma",
    "dropout_prob",
    "atom_center_mask",
    "atom_center_coords",
}


def _float(row: Mapping[str, str], name: str, *, default: float | None = None) -> float:
    value = row[name].strip()
    if not value and default is not None:
        return default
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"invalid {name}: {value!r}") from error
    if not np.isfinite(result):
        raise ValueError(f"invalid {name}: {value!r}")
    return result


def _selector(value: str, *, allow_empty: bool) -> tuple[str, int]:
    value = value.strip()
    if not value:
        if allow_empty:
            return "", 0
        raise ValueError("restraint residue selector must not be empty")
    if value.endswith("@") or value.count("@") > 1:
        raise ValueError(f"invalid restraint residue selector: {value!r}")
    residue = value.split("@", 1)[0]
    if not residue:
        raise ValueError("atom-only restraints cannot produce token restraint features")
    name, suffix = residue[0], residue[1:]
    if name not in _AA_1_TO_3:
        raise ValueError(f"unsupported restraint residue code: {name!r}")
    try:
        position = int(suffix) if suffix else 1
    except ValueError as error:
        raise ValueError(f"invalid restraint residue selector: {value!r}") from error
    if position < 1:
        raise ValueError(f"restraint residue index must be positive: {value!r}")
    return _AA_1_TO_3[name], position - 1


def parse_restraints_csv(path: str | Path) -> dict[str, list[dict[str, Any] | None]]:
    """Parse Chai's public CSV contract into inference restraint inputs.

    Covalent rows are intentionally left to the chemistry pipeline. Chai's public
    CSV has no docking row type, so docking remains a programmatic feature input.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not _CSV_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("restraints CSV is missing required columns")
        contacts: list[dict[str, Any]] = []
        pockets: list[dict[str, Any]] = []
        ids: set[str] = set()
        for row in reader:
            restraint_id = row["restraint_id"].strip()
            if not restraint_id or restraint_id in ids:
                raise ValueError(
                    f"restraint_id must be nonempty and unique: {restraint_id!r}"
                )
            ids.add(restraint_id)
            chain_a, chain_b = row["chainA"].strip(), row["chainB"].strip()
            if not chain_a or not chain_b:
                raise ValueError("restraints require chainA and chainB")
            confidence = _float(row, "confidence", default=1.0)
            minimum = _float(row, "min_distance_angstrom", default=0.0)
            maximum = _float(row, "max_distance_angstrom")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("restraint confidence must be in [0, 1]")
            if minimum < 0.0 or maximum < minimum:
                raise ValueError("restraint distances must satisfy 0 <= min <= max")
            kind = row["connection_type"].strip().lower()
            if kind == "covalent":
                continue
            if kind == "contact":
                name_a, index_a = _selector(row["res_idxA"], allow_empty=False)
                name_b, index_b = _selector(row["res_idxB"], allow_empty=False)
                contacts.append(
                    {
                        "left_residue_subchain_id": chain_a,
                        "right_residue_subchain_id": chain_b,
                        "left_residue_index": index_a,
                        "right_residue_index": index_b,
                        "left_residue_name": name_a,
                        "right_residue_name": name_b,
                        "distance_threshold": maximum,
                    }
                )
            elif kind == "pocket":
                if row["res_idxA"].strip():
                    raise ValueError("pocket chainA must not specify a residue")
                name_b, index_b = _selector(row["res_idxB"], allow_empty=False)
                pockets.append(
                    {
                        "pocket_chain_subchain_id": chain_a,
                        "pocket_token_subchain_id": chain_b,
                        "pocket_token_residue_index": index_b,
                        "pocket_token_residue_name": name_b,
                        "pocket_distance_threshold": maximum,
                    }
                )
            else:
                raise ValueError(f"unsupported restraint connection_type: {kind!r}")
    return {
        "contact_constraints": contacts or [None],
        "docking_constraints": [None],
        "pocket_constraints": pockets or [None],
    }


def _constraints(value: Any, name: str) -> list[Mapping[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype("O"):
            raise TypeError(f"{name} must contain mappings")
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list")
    if len(value) == 1 and isinstance(value[0], (list, tuple, np.ndarray)):
        value = value[0].tolist() if isinstance(value[0], np.ndarray) else value[0]
    if not value or all(item is None for item in value):
        return None
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f"{name} must contain only mappings or only null")
    return list(value)


def _require_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing, extra = expected - value.keys(), value.keys() - expected
    if missing:
        raise ValueError(f"{name} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown keys: {sorted(extra)}")


def _decode(code: np.ndarray) -> str:
    values = [int(value) for value in code if int(value) != 255]
    try:
        return bytes(values).decode("ascii")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("subchain and residue tensorcodes must be ASCII") from error


def _asym_for_subchain(subchain: str, subchains: np.ndarray, asyms: np.ndarray) -> int:
    matches = np.asarray([_decode(code) == subchain for code in subchains], bool)
    unique = np.unique(asyms[matches])
    if unique.size != 1 or int(unique[0]) == 0:
        raise ValueError(
            f"subchain {subchain!r} must resolve to exactly one non-padding asym_id"
        )
    return int(unique[0])


def _unique_residue(
    *,
    asym: int,
    index: int,
    name: str,
    asyms: np.ndarray,
    residue_indices: np.ndarray,
    residue_names: np.ndarray,
) -> np.ndarray:
    mask = (asyms == asym) & (residue_indices == index)
    positions = np.flatnonzero(mask)
    if positions.size != 1:
        raise ValueError(
            f"restraint residue must resolve uniquely: asym={asym}, index={index}"
        )
    actual = _decode(residue_names[positions[0]])
    if actual != name:
        raise ValueError(
            f"restraint residue name mismatch: expected {actual!r}, got {name!r}"
        )
    return mask


def manual_restraint_features(
    inputs: Mapping[str, Any],
    *,
    batch: int,
    tokens: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the three released Chai restraint features."""
    contact = np.full((batch, tokens, tokens, 1), -1.0, np.float32)
    docking = np.full((batch, tokens, tokens), 5, np.int64)
    pocket = np.full((batch, tokens, tokens, 1), -1.0, np.float32)
    contact_groups = _constraints(
        inputs.get("contact_constraints"), "contact_constraints"
    )
    docking_groups = _constraints(
        inputs.get("docking_constraints"), "docking_constraints"
    )
    pocket_groups = _constraints(inputs.get("pocket_constraints"), "pocket_constraints")
    if not any((contact_groups, docking_groups, pocket_groups)):
        return contact, docking, pocket
    if batch != 1:
        raise ValueError("manual restraints support batch size 1 only")

    asyms = np.asarray(inputs["token_asym_id"])[0]
    residue_indices = np.asarray(inputs["token_residue_index"])[0]
    subchains = np.asarray(inputs["subchain_id"])[0]
    residue_names = np.asarray(inputs["token_residue_name"])[0]

    for group in contact_groups or ():
        _require_keys(group, _CONTACT_KEYS, "contact constraint")
        left_asym = _asym_for_subchain(
            str(group["left_residue_subchain_id"]), subchains, asyms
        )
        right_asym = _asym_for_subchain(
            str(group["right_residue_subchain_id"]), subchains, asyms
        )
        left = _unique_residue(
            asym=left_asym,
            index=int(group["left_residue_index"]),
            name=str(group["left_residue_name"]),
            asyms=asyms,
            residue_indices=residue_indices,
            residue_names=residue_names,
        )
        right = _unique_residue(
            asym=right_asym,
            index=int(group["right_residue_index"]),
            name=str(group["right_residue_name"]),
            asyms=asyms,
            residue_indices=residue_indices,
            residue_names=residue_names,
        )
        contact[0, left, right, 0] = np.float32(group["distance_threshold"])

    for group in pocket_groups or ():
        _require_keys(group, _POCKET_KEYS, "pocket constraint")
        chain_asym = _asym_for_subchain(
            str(group["pocket_chain_subchain_id"]), subchains, asyms
        )
        token_asym = _asym_for_subchain(
            str(group["pocket_token_subchain_id"]), subchains, asyms
        )
        token = _unique_residue(
            asym=token_asym,
            index=int(group["pocket_token_residue_index"]),
            name=str(group["pocket_token_residue_name"]),
            asyms=asyms,
            residue_indices=residue_indices,
            residue_names=residue_names,
        )
        pocket[0, token, asyms == chain_asym, 0] = np.float32(
            group["pocket_distance_threshold"]
        )

    if docking_groups:
        distances = np.zeros((tokens, tokens), np.float32)
        mask = np.zeros((tokens, tokens), bool)
        rng = np.random.default_rng(secrets.randbits(128)) if rng is None else rng
        for group in docking_groups:
            _require_keys(group, _DOCKING_KEYS, "docking constraint")
            chain_ids = list(group["subchain_ids"])
            coords = [
                np.asarray(value, np.float32) for value in group["atom_center_coords"]
            ]
            coord_masks = [
                np.asarray(value, bool) for value in group["atom_center_mask"]
            ]
            if not (len(chain_ids) == len(coords) == len(coord_masks)) or not chain_ids:
                raise ValueError(
                    "docking constraint chain, coordinate, and mask counts differ"
                )
            sigma = float(group["noise_sigma"])
            probability = float(group["dropout_prob"])
            if sigma < 0.0 or not 0.0 <= probability <= 1.0:
                raise ValueError("docking noise_sigma and dropout_prob are invalid")
            coords = [
                value + rng.normal(0.0, sigma, value.shape).astype(np.float32)
                for value in coords
            ]
            chain_asyms = [
                _asym_for_subchain(str(value), subchains, asyms) for value in chain_ids
            ]
            for i in range(len(chain_ids)):
                for j in range(i, len(chain_ids)):
                    rows, cols = (
                        np.flatnonzero(asyms == chain_asyms[i]),
                        np.flatnonzero(asyms == chain_asyms[j]),
                    )
                    if coords[i].shape != (rows.size, 3) or coords[j].shape != (
                        cols.size,
                        3,
                    ):
                        raise ValueError(
                            "docking coordinates must match every token in each chain"
                        )
                    if coord_masks[i].shape != (rows.size,) or coord_masks[j].shape != (
                        cols.size,
                    ):
                        raise ValueError(
                            "docking coordinate masks must match chain token counts"
                        )
                    pair = np.sqrt(
                        np.sum(
                            (coords[i][:, None] - coords[j][None]) ** 2,
                            axis=-1,
                            dtype=np.float32,
                        )
                    )
                    pair_mask = coord_masks[i][:, None] & coord_masks[j][None]
                    distances[np.ix_(rows, cols)] = pair
                    distances[np.ix_(cols, rows)] = pair.T
                    mask[np.ix_(rows, cols)] = pair_mask
                    mask[np.ix_(cols, rows)] = pair_mask.T
        docking[0] = np.searchsorted(
            np.asarray([0.0, 4.0, 8.0, 16.0], np.float32), distances
        ).astype(np.int64)
        docking[0, ~mask] = 5
        dropout = float(docking_groups[0]["dropout_prob"])
        drop_tokens = rng.random(tokens) < dropout
        docking[0, drop_tokens[:, None] | drop_tokens[None, :]] = 5
    return contact, docking, pocket


__all__ = ["manual_restraint_features", "parse_restraints_csv"]
