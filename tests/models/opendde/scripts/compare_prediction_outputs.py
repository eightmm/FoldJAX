"""Compare two OpenDDE prediction CIF/summary output pairs.

The comparison is deliberately independent of JAX and PyTorch.  Gemmi reads
mmCIF columns by name, NumPy performs the rigid alignments, and the resulting
JSON records both structural sanity checks and the RNG caveat needed when two
frameworks did not consume an identical random tape.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

type AtomKey = tuple[str, str, str, str, str, str, str]

ATOM_KEY_FIELDS = (
    "pdbx_PDB_model_num",
    "label_asym_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "label_comp_id",
    "label_atom_id",
    "label_alt_id",
)
DNA_COMPONENTS = frozenset({"DA", "DC", "DG", "DT", "DI"})
CLASH_THRESHOLD_ANGSTROM = 1.1
CLASH_ROW_CHUNK = 512
RNG_INTERPRETATION = (
    "Unless both runs consumed an identical shared random tape, framework RNG "
    "streams can differ even at the same numeric seed. RMSDs in that case are "
    "sample-to-sample comparisons, not a numerical parity claim."
)


@dataclass(frozen=True)
class CifAtoms:
    """Atom rows and the metadata required by the comparison."""

    path: Path
    keys: tuple[AtomKey, ...]
    coordinates: np.ndarray
    chains: np.ndarray
    component_ids: np.ndarray
    atom_names: np.ndarray
    entity_ids: np.ndarray
    group_pdb: np.ndarray
    polymer_chains: frozenset[str]


def _normalize_cif_value(value: object) -> str:
    """Normalize the two mmCIF missing markers to one identity value."""

    if value is None or value is False:
        return ""
    text = str(value)
    return "" if text in {".", "?"} else text


def _column(
    category: dict[str, list[object]],
    name: str,
    *,
    count: int,
    default: str | None = None,
) -> list[str]:
    values = category.get(name)
    if values is None:
        if default is None:
            raise ValueError(f"_atom_site.{name} is required")
        return [default] * count
    if len(values) != count:
        raise ValueError(f"_atom_site.{name} has {len(values)} rows, expected {count}")
    return [_normalize_cif_value(value) for value in values]


def load_cif_atoms(path: str | Path) -> CifAtoms:
    """Load atom rows by mmCIF tag and reject ambiguous/non-finite inputs."""

    path = Path(path)
    block = gemmi.cif.read_file(str(path)).sole_block()
    atom_site = block.get_mmcif_category("_atom_site.")
    if not atom_site:
        raise ValueError(f"{path}: missing _atom_site category")
    first_column = next(iter(atom_site.values()))
    count = len(first_column)
    if count == 0:
        raise ValueError(f"{path}: _atom_site has no atom rows")

    model_numbers = _column(atom_site, "pdbx_PDB_model_num", count=count, default="1")
    chains = _column(atom_site, "label_asym_id", count=count)
    sequence_ids = _column(atom_site, "label_seq_id", count=count, default="")
    insertion_codes = _column(atom_site, "pdbx_PDB_ins_code", count=count, default="")
    component_ids = _column(atom_site, "label_comp_id", count=count)
    atom_names = _column(atom_site, "label_atom_id", count=count)
    alt_ids = _column(atom_site, "label_alt_id", count=count, default="")
    entity_ids = _column(atom_site, "label_entity_id", count=count, default="")
    group_pdb = _column(atom_site, "group_PDB", count=count, default="ATOM")

    keys = tuple(
        zip(
            model_numbers,
            chains,
            sequence_ids,
            insertion_codes,
            component_ids,
            atom_names,
            alt_ids,
            strict=True,
        )
    )
    duplicate_keys = [key for key, n_rows in Counter(keys).items() if n_rows > 1]
    if duplicate_keys:
        raise ValueError(
            f"{path}: atom keys are not unique; first duplicate is "
            f"{duplicate_keys[0]!r}"
        )

    try:
        coordinates = np.column_stack(
            [
                np.asarray(_column(atom_site, name, count=count), dtype=np.float64)
                for name in ("Cartn_x", "Cartn_y", "Cartn_z")
            ]
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: coordinates are not numeric") from error
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{path}: coordinates contain NaN or infinity")

    entity_poly = block.get_mmcif_category("_entity_poly.")
    polymer_entity_ids = {
        _normalize_cif_value(value) for value in entity_poly.get("entity_id", ())
    }
    chain_array = np.asarray(chains, dtype=str)
    entity_array = np.asarray(entity_ids, dtype=str)
    group_array = np.asarray(group_pdb, dtype=str)
    polymer_chains: set[str] = set()
    for chain in np.unique(chain_array):
        mask = chain_array == chain
        if polymer_entity_ids:
            is_polymer = any(
                entity_id in polymer_entity_ids for entity_id in entity_array[mask]
            )
        else:
            # Prediction CIFs normally carry _entity_poly.  This conservative
            # fallback still handles simple ATOM-only polymer structures.
            is_polymer = bool(np.all(group_array[mask] == "ATOM"))
        if is_polymer:
            polymer_chains.add(str(chain))

    return CifAtoms(
        path=path,
        keys=keys,
        coordinates=coordinates,
        chains=chain_array,
        component_ids=np.asarray(component_ids, dtype=str),
        atom_names=np.asarray(atom_names, dtype=str),
        entity_ids=entity_array,
        group_pdb=group_array,
        polymer_chains=frozenset(polymer_chains),
    )


def _match_candidate(reference: CifAtoms, candidate: CifAtoms) -> np.ndarray:
    reference_keys = set(reference.keys)
    candidate_keys = set(candidate.keys)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        extra = sorted(candidate_keys - reference_keys)
        raise ValueError(
            "atom key sets differ: "
            f"candidate is missing {len(missing)} and has {len(extra)} extra keys; "
            f"first missing={missing[:1]!r}, first extra={extra[:1]!r}"
        )
    candidate_index = {key: index for index, key in enumerate(candidate.keys)}
    return np.asarray([candidate_index[key] for key in reference.keys], dtype=np.int64)


def _rmsd(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(left - right), axis=-1))))


def _kabsch_align(moving: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigidly align ``moving`` onto ``target`` without permitting reflection."""

    if moving.shape != target.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError(
            f"Kabsch inputs must have equal [N, 3] shapes, got "
            f"{moving.shape} and {target.shape}"
        )
    if moving.shape[0] == 0:
        raise ValueError("Kabsch alignment requires at least one atom")
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    moving_centered = moving - moving_center
    target_centered = target - target_center
    left, _, right_t = np.linalg.svd(moving_centered.T @ target_centered)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(right_t.T @ left.T) < 0.0:
        correction[-1, -1] = -1.0
    rotation = right_t.T @ correction @ left.T
    return moving_centered @ rotation.T + target_center


def _subset_rmsd(
    reference_coordinates: np.ndarray,
    candidate_coordinates: np.ndarray,
    globally_aligned_candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    reference_subset = reference_coordinates[mask]
    candidate_subset = candidate_coordinates[mask]
    return {
        "atom_count": int(mask.sum()),
        "global_transform_rmsd": _rmsd(
            globally_aligned_candidate[mask], reference_subset
        ),
        "independent_kabsch_rmsd": _rmsd(
            _kabsch_align(candidate_subset, reference_subset), reference_subset
        ),
    }


def _rmsd_report(
    reference: CifAtoms,
    candidate_coordinates: np.ndarray,
) -> dict[str, Any]:
    reference_coordinates = reference.coordinates
    globally_aligned = _kabsch_align(candidate_coordinates, reference_coordinates)
    per_chain: dict[str, dict[str, float | int]] = {}
    for chain in sorted(np.unique(reference.chains)):
        mask = reference.chains == chain
        per_chain[str(chain)] = _subset_rmsd(
            reference_coordinates, candidate_coordinates, globally_aligned, mask
        )

    markers: dict[str, dict[str, float | int]] = {}
    protein_ca = (reference.chains == "A") & (reference.atom_names == "CA")
    if np.any(protein_ca):
        markers["A:CA"] = _subset_rmsd(
            reference_coordinates,
            candidate_coordinates,
            globally_aligned,
            protein_ca,
        )
    for chain in sorted(np.unique(reference.chains)):
        chain_mask = reference.chains == chain
        if not any(
            component in DNA_COMPONENTS
            for component in reference.component_ids[chain_mask]
        ):
            continue
        c4_prime = chain_mask & (reference.atom_names == "C4'")
        if np.any(c4_prime):
            markers[f"{chain}:C4'"] = _subset_rmsd(
                reference_coordinates,
                candidate_coordinates,
                globally_aligned,
                c4_prime,
            )

    return {
        "all_atom": {
            "atom_count": int(reference_coordinates.shape[0]),
            "raw_rmsd": _rmsd(candidate_coordinates, reference_coordinates),
            "kabsch_rmsd": _rmsd(globally_aligned, reference_coordinates),
        },
        "per_chain": per_chain,
        "markers": markers,
    }


def _structure_counts(atoms: CifAtoms) -> dict[str, Any]:
    per_chain: dict[str, dict[str, int | bool]] = {}
    for chain in sorted(np.unique(atoms.chains)):
        indices = np.flatnonzero(atoms.chains == chain)
        residues = {
            (
                atoms.keys[index][2],
                atoms.keys[index][3],
                atoms.keys[index][4],
            )
            for index in indices
        }
        per_chain[str(chain)] = {
            "atom_count": int(indices.size),
            "cif_residue_count": len(residues),
            "is_polymer": str(chain) in atoms.polymer_chains,
        }
    residues = {(key[1], key[2], key[3], key[4]) for key in atoms.keys}
    return {
        "atom_count": len(atoms.keys),
        "cif_residue_count": len(residues),
        "chain_count": len(per_chain),
        "model_count": len({key[0] for key in atoms.keys}),
        "per_chain": per_chain,
    }


def _load_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: summary JSON must contain an object")
    return payload


def _score_array(value: object, *, path: Path, key: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: score {key!r} is not numeric") from error
    if array.size == 0:
        raise ValueError(f"{path}: score {key!r} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: score {key!r} contains NaN or infinity")
    return array


def _score_report(
    reference_path: str | Path,
    candidate_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)
    reference = _load_summary(reference_path)
    candidate = _load_summary(candidate_path)
    if set(reference) != set(candidate):
        raise ValueError(
            "summary key sets differ: "
            f"candidate is missing {sorted(set(reference) - set(candidate))} and "
            f"has extras {sorted(set(candidate) - set(reference))}"
        )

    scalar_scores: dict[str, dict[str, float]] = {}
    array_scores: dict[str, dict[str, Any]] = {}
    boolean_scores: dict[str, dict[str, bool]] = {}
    for key in sorted(reference):
        reference_value = reference[key]
        candidate_value = candidate[key]
        if isinstance(reference_value, bool) or isinstance(candidate_value, bool):
            if not isinstance(reference_value, bool) or not isinstance(
                candidate_value, bool
            ):
                raise ValueError(f"summary score {key!r} changes boolean type")
            boolean_scores[key] = {
                "reference": reference_value,
                "candidate": candidate_value,
                "equal": reference_value == candidate_value,
            }
            continue
        reference_array = _score_array(reference_value, path=reference_path, key=key)
        candidate_array = _score_array(candidate_value, path=candidate_path, key=key)
        if reference_array.shape != candidate_array.shape:
            raise ValueError(
                f"summary score {key!r} changes shape from "
                f"{reference_array.shape} to {candidate_array.shape}"
            )
        difference = candidate_array - reference_array
        if reference_array.ndim == 0:
            delta = float(difference)
            scalar_scores[key] = {
                "reference": float(reference_array),
                "candidate": float(candidate_array),
                "delta": delta,
                "abs_delta": abs(delta),
            }
        else:
            array_scores[key] = {
                "shape": list(reference_array.shape),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "max_abs": float(np.max(np.abs(difference))),
                "mean_signed_delta": float(np.mean(difference)),
            }
    return (
        {
            "scalar": scalar_scores,
            "array": array_scores,
            "boolean": boolean_scores,
        },
        reference,
        candidate,
    )


def _close_pair_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    threshold: float,
) -> tuple[int, float]:
    squared_threshold = threshold * threshold
    close_pairs = 0
    minimum_squared_distance = np.inf
    for start in range(0, left.shape[0], CLASH_ROW_CHUNK):
        difference = left[start : start + CLASH_ROW_CHUNK, None, :] - right[None, :, :]
        squared_distance = np.sum(np.square(difference), axis=-1)
        close_pairs += int(np.count_nonzero(squared_distance < squared_threshold))
        minimum_squared_distance = min(
            minimum_squared_distance, float(np.min(squared_distance))
        )
    return close_pairs, float(np.sqrt(minimum_squared_distance))


def _clash_geometry(atoms: CifAtoms) -> dict[str, Any]:
    chains = sorted(str(chain) for chain in np.unique(atoms.chains))
    pair_metrics: dict[str, dict[str, float | int | bool]] = {}
    computed_has_clash = False
    for left_chain, right_chain in combinations(chains, 2):
        left = atoms.coordinates[atoms.chains == left_chain]
        right = atoms.coordinates[atoms.chains == right_chain]
        close_pairs, minimum_distance = _close_pair_metrics(
            left, right, threshold=CLASH_THRESHOLD_ANGSTROM
        )
        relative_close_pairs = close_pairs / max(min(len(left), len(right)), 1)
        severe = close_pairs > 100 or relative_close_pairs > 0.5
        evaluated = (
            left_chain in atoms.polymer_chains and right_chain in atoms.polymer_chains
        )
        computed_has_clash = computed_has_clash or (evaluated and severe)
        pair_metrics[f"{left_chain}:{right_chain}"] = {
            "close_pair_count": close_pairs,
            "relative_close_pair_count": relative_close_pairs,
            "minimum_distance_angstrom": minimum_distance,
            "evaluated_for_has_clash": evaluated,
            "pair_has_clash": bool(evaluated and severe),
        }
    return {
        "polymer_chains": sorted(atoms.polymer_chains),
        "computed_has_clash": bool(computed_has_clash),
        "chain_pairs": pair_metrics,
    }


def compare_prediction_outputs(
    *,
    reference_cif: str | Path,
    candidate_cif: str | Path,
    reference_summary: str | Path,
    candidate_summary: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Validate and compare two prediction output pairs, then write JSON."""

    reference_atoms = load_cif_atoms(reference_cif)
    candidate_atoms = load_cif_atoms(candidate_cif)
    candidate_order = _match_candidate(reference_atoms, candidate_atoms)
    candidate_coordinates = candidate_atoms.coordinates[candidate_order]

    scores, reference_scores, candidate_scores = _score_report(
        reference_summary, candidate_summary
    )
    if "has_clash" not in reference_scores or "has_clash" not in candidate_scores:
        raise ValueError("both summaries must contain has_clash")
    if not isinstance(reference_scores["has_clash"], bool) or not isinstance(
        candidate_scores["has_clash"], bool
    ):
        raise ValueError("summary has_clash values must be boolean")

    reference_clash = _clash_geometry(reference_atoms)
    candidate_clash = _clash_geometry(candidate_atoms)
    reference_summary_clash = reference_scores["has_clash"]
    candidate_summary_clash = candidate_scores["has_clash"]
    reference_clash["summary_has_clash"] = reference_summary_clash
    reference_clash["summary_consistent"] = (
        reference_summary_clash == reference_clash["computed_has_clash"]
    )
    candidate_clash["summary_has_clash"] = candidate_summary_clash
    candidate_clash["summary_consistent"] = (
        candidate_summary_clash == candidate_clash["computed_has_clash"]
    )

    report: dict[str, Any] = {
        "inputs": {
            "reference_cif": str(Path(reference_cif)),
            "candidate_cif": str(Path(candidate_cif)),
            "reference_summary": str(Path(reference_summary)),
            "candidate_summary": str(Path(candidate_summary)),
        },
        "interpretation": {"rmsd_rng_caveat": RNG_INTERPRETATION},
        "checks": {
            "atom_keys_unique_and_equal": True,
            "coordinates_finite": True,
            "scores_finite": True,
            "reference_summary_has_clash_consistent": reference_clash[
                "summary_consistent"
            ],
            "candidate_summary_has_clash_consistent": candidate_clash[
                "summary_consistent"
            ],
        },
        "atom_identity": {
            "key_fields": list(ATOM_KEY_FIELDS),
            "atom_count": len(reference_atoms.keys),
            "candidate_rows_reordered": reference_atoms.keys != candidate_atoms.keys,
        },
        "counts": {
            "reference": _structure_counts(reference_atoms),
            "candidate": _structure_counts(candidate_atoms),
        },
        "rmsd_angstrom": _rmsd_report(reference_atoms, candidate_coordinates),
        "scores": scores,
        "clash": {
            "threshold_angstrom": CLASH_THRESHOLD_ANGSTROM,
            "criterion": (
                "For each polymer-chain pair: close_pair_count > 100 or "
                "close_pair_count / min(chain atom counts) > 0.5; OR over pairs."
            ),
            "reference": reference_clash,
            "candidate": candidate_clash,
            "summary_has_clash_equal": (
                reference_summary_clash == candidate_summary_clash
            ),
        },
    }

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cif", type=Path, required=True)
    parser.add_argument("--candidate-cif", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    compare_prediction_outputs(
        reference_cif=args.reference_cif,
        candidate_cif=args.candidate_cif,
        reference_summary=args.reference_summary,
        candidate_summary=args.candidate_summary,
        out=args.out,
    )


if __name__ == "__main__":
    main()
