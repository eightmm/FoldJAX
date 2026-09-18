"""Strict, identity-preserving comparison of two written mmCIF structures.

The metric describes coordinates that writers emitted.  It deliberately does
not claim that either atom set is the raw network atom mask: that information
is not available from a CIF alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from bench.structures import kabsch

SCHEMA_VERSION = "1.0"
_MISSING = frozenset({"", ".", "?"})
_PROTEIN_COMPONENTS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "ASX",
        "CYS",
        "GLN",
        "GLU",
        "GLX",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "MSE",
        "PHE",
        "PRO",
        "PYL",
        "SEC",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "UNK",
        "VAL",
    }
)


@dataclass(frozen=True)
class WrittenAtoms:
    """Validated first-model primary-altloc atom rows from one CIF."""

    path: Path
    keys: tuple[tuple[str, str, str, str, str], ...]
    coordinates: np.ndarray
    chains: np.ndarray
    entity_instances: np.ndarray
    elements: np.ndarray
    ca_mask: np.ndarray
    metadata: dict[str, Any]


def _normal(value: object) -> str:
    """Return a CIF value, mapping mmCIF absent markers to an empty string."""
    if value is None or value is False:
        return ""
    value = str(value)
    return "" if value in _MISSING else value


def _column(category: dict[str, list[object]], name: str, count: int) -> list[str]:
    values = category.get(name)
    if values is None:
        return [""] * count
    if len(values) != count:
        raise ValueError(f"_atom_site.{name} has {len(values)} rows, expected {count}")
    return [_normal(value) for value in values]


def _identity_value(
    label: str, auth: str, field: str, path: Path, fallback_counts: Counter[str]
) -> str:
    """Use label identity first; make auth fallback and missing identities explicit."""
    if label:
        return label
    if auth:
        fallback_counts[field] += 1
        return auth
    raise ValueError(f"{path}: missing required {field} identity (label and auth)")


def _polymer_entity_types(block: gemmi.cif.Block) -> dict[str, str]:
    entity_poly = block.get_mmcif_category("_entity_poly.")
    if not entity_poly:
        return {}
    entity_ids = [_normal(value) for value in entity_poly.get("entity_id", ())]
    types = [_normal(value) for value in entity_poly.get("type", ())]
    if types and len(types) != len(entity_ids):
        raise ValueError("_entity_poly.type has a different row count from entity_id")
    return {
        entity_id: types[index] if types else ""
        for index, entity_id in enumerate(entity_ids)
        if entity_id
    }


def read_written_atoms(path: str | Path) -> WrittenAtoms:
    """Read first-model, primary-altloc atoms and reject ambiguous identities."""
    path = Path(path)
    block = gemmi.cif.read_file(str(path)).sole_block()
    category = block.get_mmcif_category("_atom_site.")
    if not category:
        raise ValueError(f"{path}: missing _atom_site category")
    count = len(next(iter(category.values()), ()))
    if count == 0:
        raise ValueError(f"{path}: _atom_site has no rows")

    columns = {
        name: _column(category, name, count)
        for name in (
            "pdbx_PDB_model_num",
            "label_alt_id",
            "label_asym_id",
            "auth_asym_id",
            "label_seq_id",
            "auth_seq_id",
            "pdbx_PDB_ins_code",
            "label_comp_id",
            "auth_comp_id",
            "label_atom_id",
            "auth_atom_id",
            "label_entity_id",
            "type_symbol",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "occupancy",
        )
    }
    models = columns["pdbx_PDB_model_num"]
    selected_model = next((model for model in models if model), "1")
    fallback_counts: Counter[str] = Counter()
    dropped_model = dropped_altloc = 0
    keys: list[tuple[str, str, str, str, str]] = []
    coords: list[np.ndarray] = []
    chains: list[str] = []
    instances: list[str] = []
    elements: list[str] = []
    ca: list[bool] = []
    polymer_types = _polymer_entity_types(block)
    has_polymer_types = any(polymer_types.values())
    missing_entity_metadata = not bool(polymer_types)

    for index in range(count):
        model = models[index] or "1"
        if model != selected_model:
            dropped_model += 1
            continue
        altloc = columns["label_alt_id"][index]
        if altloc not in ("", "A"):
            dropped_altloc += 1
            continue
        chain = _identity_value(
            columns["label_asym_id"][index],
            columns["auth_asym_id"][index],
            "asym_id",
            path,
            fallback_counts,
        )
        sequence = _identity_value(
            columns["label_seq_id"][index],
            columns["auth_seq_id"][index],
            "seq_id",
            path,
            fallback_counts,
        )
        component = _identity_value(
            columns["label_comp_id"][index],
            columns["auth_comp_id"][index],
            "comp_id",
            path,
            fallback_counts,
        )
        atom = _identity_value(
            columns["label_atom_id"][index],
            columns["auth_atom_id"][index],
            "atom_id",
            path,
            fallback_counts,
        )
        key = (chain, sequence, columns["pdbx_PDB_ins_code"][index], component, atom)
        keys.append(key)
        try:
            point = np.asarray(
                [columns[name][index] for name in ("Cartn_x", "Cartn_y", "Cartn_z")],
                dtype=np.float64,
            )
        except ValueError as error:
            raise ValueError(
                f"{path}: coordinates are not numeric at row {index + 1}"
            ) from error
        if not np.isfinite(point).all():
            raise ValueError(f"{path}: nonfinite coordinates at row {index + 1}")
        coords.append(point)
        chains.append(chain)
        entity = columns["label_entity_id"][index]
        is_polymer = entity in polymer_types
        if is_polymer:
            instances.append(chain)
        else:
            instances.append(
                "|".join(
                    (
                        chain,
                        entity or "unconfirmed",
                        sequence,
                        columns["pdbx_PDB_ins_code"][index],
                        component,
                    )
                )
            )
            missing_entity_metadata |= not bool(entity)
        if has_polymer_types:
            is_protein = "polypeptide" in polymer_types.get(entity, "").lower()
            ca_policy = "entity_poly.type polypeptide"
        else:
            is_protein = component.upper() in _PROTEIN_COMPONENTS and (
                not polymer_types or is_polymer
            )
            ca_policy = "conservative protein-component fallback"
        element = columns["type_symbol"][index].upper()
        if not element:
            raise ValueError(f"{path}: missing required type_symbol at row {index + 1}")
        elements.append(element)
        ca.append(atom.upper() == "CA" and element == "C" and is_protein)

    duplicates = [key for key, number in Counter(keys).items() if number > 1]
    if duplicates:
        raise ValueError(f"{path}: duplicate strict atom identity {duplicates[0]!r}")
    if not keys:
        raise ValueError(f"{path}: no atoms after first-model/altloc selection")
    return WrittenAtoms(
        path=path,
        keys=tuple(keys),
        coordinates=np.asarray(coords, dtype=np.float64),
        chains=np.asarray(chains, dtype=str),
        entity_instances=np.asarray(instances, dtype=str),
        elements=np.asarray(elements, dtype=str),
        ca_mask=np.asarray(ca, dtype=bool),
        metadata={
            "selected_model": selected_model,
            "model_policy": "first model encountered; later models excluded",
            "altloc_policy": (
                "blank/unknown or A accepted; other alternate locations excluded"
            ),
            "dropped_model_rows": dropped_model,
            "dropped_altloc_rows": dropped_altloc,
            "hydrogen_policy": "included when written",
            "zero_occupancy_policy": "included when written",
            "auth_fallback_rows": dict(sorted(fallback_counts.items())),
            "ca_selection_policy": f"atom name CA, element C, and {ca_policy}",
            "entity_instance_policy": (
                "polymer=chain; nonpolymer=chain/entity/residue/insertion/component"
            ),
            "entity_metadata_uncertainty": missing_entity_metadata,
        },
    )


def _rmsd(displacements: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(displacements[mask] ** 2)))


def _group_rmsd(displacements: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        label: _rmsd(displacements, labels == label)
        for label in sorted(set(labels.tolist()))
    }


def _group_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        label: int(np.count_nonzero(labels == label))
        for label in sorted(set(labels.tolist()))
    }


def compare_written_structures(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Compare strict matching written atom identities under one proper fit."""
    left_atoms = read_written_atoms(left)
    right_atoms = read_written_atoms(right)
    left_index = {key: index for index, key in enumerate(left_atoms.keys)}
    right_index = {key: index for index, key in enumerate(right_atoms.keys)}
    left_only = sorted(set(left_index) - set(right_index))
    right_only = sorted(set(right_index) - set(left_index))
    if left_only or right_only:
        raise ValueError(
            "strict atom identity mismatch: "
            f"left_only={len(left_only)}, right_only={len(right_only)}"
        )
    keys = sorted(left_index)
    left_rows = np.asarray([left_index[key] for key in keys])
    right_rows = np.asarray([right_index[key] for key in keys])
    if not np.array_equal(
        left_atoms.elements[left_rows], right_atoms.elements[right_rows]
    ):
        raise ValueError(
            "strict atom annotation mismatch: "
            "element differs for matching atom identity"
        )
    if not np.array_equal(
        left_atoms.ca_mask[left_rows], right_atoms.ca_mask[right_rows]
    ):
        raise ValueError(
            "ambiguous CA selection: polymer/protein metadata "
            "differs for matching atom identity"
        )
    p = left_atoms.coordinates[left_rows]
    q = right_atoms.coordinates[right_rows]
    rotation, pm, qm = kabsch(p, q)
    displacements = np.linalg.norm((p - pm) @ rotation.T - (q - qm), axis=1)
    ca_mask = left_atoms.ca_mask[left_rows]
    chain_labels = left_atoms.chains[left_rows]
    entity_labels = left_atoms.entity_instances[left_rows]
    ca_applicable = bool(ca_mask.any())
    metrics = {
        "all_atom_rmsd": _rmsd(displacements, np.ones(len(displacements), dtype=bool)),
        "ca_rmsd_same_fit": _rmsd(displacements, ca_mask) if ca_applicable else None,
        "per_chain_same_fit": _group_rmsd(displacements, chain_labels),
        "per_entity_instance_same_fit": _group_rmsd(displacements, entity_labels),
        "per_chain_atom_counts": _group_counts(chain_labels),
        "per_entity_instance_atom_counts": _group_counts(entity_labels),
        "atom_displacement_p95": float(np.quantile(displacements, 0.95)),
        "atom_displacement_max": float(displacements.max()),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "written_coordinates",
        "units": "angstrom",
        "inputs": {
            "left": {
                "sha256": hashlib.sha256(left_atoms.path.read_bytes()).hexdigest()
            },
            "right": {
                "sha256": hashlib.sha256(right_atoms.path.read_bytes()).hexdigest()
            },
        },
        "policies": {
            "primary_fit": (
                "strict identity all-atom proper Kabsch fit; "
                "one transform reused for every metric"
            ),
            "raw_network_atom_mask": "not_available_from_written_cif",
            "homomer_permutation_diagnostic": (
                "not_run; primary metric retains written chain identities"
            ),
            "per_entity_instance_comparison": (
                "left-defined grouping; right entity annotation is "
                "not independently verified"
            ),
            "left_to_right_transform": {
                "rotation": rotation.tolist(),
                "translation": (qm - pm @ rotation.T).tolist(),
                "coordinate_convention": "right = left @ rotation.T + translation",
            },
            "left": left_atoms.metadata,
            "right": right_atoms.metadata,
        },
        "atom_identity": {
            "strict_match": True,
            "key_fields": [
                "asym_id",
                "seq_id",
                "pdbx_PDB_ins_code",
                "comp_id",
                "atom_id",
            ],
            "matched_atoms": len(keys),
        },
        "coverage": {
            "atom": {
                "left": len(left_atoms.keys),
                "right": len(right_atoms.keys),
                "matched": len(keys),
            },
            "ca": {
                "left": int(left_atoms.ca_mask.sum()),
                "right": int(right_atoms.ca_mask.sum()),
                "matched": int(ca_mask.sum()),
                "applicable": ca_applicable,
            },
            "entity_instances": {
                "left": len(set(left_atoms.entity_instances.tolist())),
                "right": len(set(right_atoms.entity_instances.tolist())),
            },
        },
        "missing_or_not_applicable": {
            "ca_metrics": None if ca_applicable else "no protein carbon-alpha atoms"
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_written_structures(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
