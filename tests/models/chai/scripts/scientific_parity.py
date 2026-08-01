"""Scientific comparison primitives for Chai Torch and Chai-JAX outputs.

The artifact format intentionally records the inference contract next to the
arrays.  Comparing outputs produced from different inputs, source weights, or
sampler settings is an error rather than a numerical result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

SCHEMA_VERSION = 1
CONTRACT_KEYS = (
    "fixture_id",
    "fasta_sha256",
    "source_weight_sha256",
    "config",
    "branches",
    "reference_augmentation",
)
NUMERIC_PREPARED_TOLERANCES = {
    # Reference-augmentation replay uses identical rigid transforms, while the
    # official and native conformer archives can differ by one final FP32 ULP.
    "features.AtomRefPos": {"atol": 1e-7, "rtol": 1e-6},
    "features.InverseSquaredBlockedAtomPairDistances": {
        "atol": 2e-7,
        "rtol": 1e-6,
    },
    # The native NumPy geometry path and official Torch geometry differ only
    # by final FP32 rounding for the same template coordinates.
    "features.TemplateUnitVector": {
        "atol": 2e-7,
        "rtol": 1e-6,
    },
}


class ContractMismatchError(ValueError):
    """Raised when two runs do not represent the same scientific experiment."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    # Official Torch feature generators retain a terminal channel dimension
    # of one; the native JAX feature bridge removes it before embedding. Both
    # layouts address the same scalar at every preceding index.
    if array.ndim and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.dtype.kind == "b":
        array = array.astype(np.bool_, copy=False)
    elif array.dtype.kind in "iu":
        array = array.astype("<i8", copy=False)
    elif array.dtype.kind == "f":
        array = array.astype("<f4", copy=False)
    elif array.dtype.kind in "SU":
        array = array.astype("U", copy=False)
    else:
        raise TypeError(f"unsupported array dtype for provenance: {array.dtype}")
    return np.ascontiguousarray(array)


def semantic_array_sha256(value: Any) -> str:
    """Hash values after backend-neutral dtype canonicalization."""
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    if array.dtype.kind == "U":
        digest.update("\0".join(array.reshape(-1).tolist()).encode())
    else:
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tensor_manifest(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        result[name] = {
            "shape": list(array.shape),
            "semantic_sha256": semantic_array_sha256(array),
        }
    return result


def source_weight_manifest_from_native_bundle(
    bundle_path: str | Path,
) -> dict[str, str]:
    manifest = json.loads((Path(bundle_path) / "manifest.json").read_text())
    return {
        name: item["source_sha256"]
        for name, item in sorted(manifest["components"].items())
    }


def source_weight_manifest_from_torch(model_directory: str | Path) -> dict[str, str]:
    root = Path(model_directory)
    filenames = {
        "feature_embedding": "feature_embedding.pt",
        "bond_loss_input_proj": "bond_loss_input_proj.pt",
        "token_embedder": "token_embedder.pt",
        "trunk": "trunk.pt",
        "diffusion_module": "diffusion_module.pt",
        "confidence_head": "confidence_head.pt",
    }
    return {name: sha256_file(root / filename) for name, filename in filenames.items()}


def read_cif_coordinates(
    cif_paths: Sequence[str | Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Read coordinates and stable atom identities in atom_site row order."""
    all_coords: list[np.ndarray] = []
    identities: np.ndarray | None = None
    for path in cif_paths:
        block = gemmi.cif.read(str(path)).sole_block()
        columns = {
            name: list(block.find_values(tag))
            for name, tag in {
                "chain": "_atom_site.label_asym_id",
                "seq": "_atom_site.label_seq_id",
                "component": "_atom_site.label_comp_id",
                "atom": "_atom_site.label_atom_id",
                "x": "_atom_site.Cartn_x",
                "y": "_atom_site.Cartn_y",
                "z": "_atom_site.Cartn_z",
            }.items()
        }
        lengths = {len(value) for value in columns.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError(f"invalid or empty atom_site loop: {path}")
        occurrence: dict[tuple[str, ...], int] = {}
        rows = []
        for chain, seq, component, atom in zip(
            columns["chain"],
            columns["seq"],
            columns["component"],
            columns["atom"],
            strict=True,
        ):
            base = (chain, seq, component, atom)
            ordinal = occurrence.get(base, 0)
            occurrence[base] = ordinal + 1
            rows.append((*base, str(ordinal)))
        current = np.asarray(rows, dtype="U")
        if identities is None:
            identities = current
        elif not np.array_equal(current, identities):
            raise ValueError("candidate CIF files have different atom identities")
        all_coords.append(
            np.stack(
                [
                    np.asarray(columns[axis], dtype=np.float64)
                    for axis in ("x", "y", "z")
                ],
                axis=-1,
            )
        )
    assert identities is not None
    return np.stack(all_coords), identities


def save_artifact(
    path: str | Path,
    *,
    coords: np.ndarray,
    atom_ids: np.ndarray,
    pae: np.ndarray,
    pde: np.ndarray,
    plddt: np.ndarray,
    ranking: Mapping[str, Any],
    metadata: Mapping[str, Any],
    prepared_arrays: Mapping[str, Any] | None = None,
) -> None:
    arrays = {
        "coords": np.asarray(coords, dtype=np.float32),
        "atom_ids": np.asarray(atom_ids, dtype="U"),
        "pae": np.asarray(pae, dtype=np.float32),
        "pde": np.asarray(pde, dtype=np.float32),
        "plddt": np.asarray(plddt, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    arrays.update(
        {f"ranking__{key}": np.asarray(value) for key, value in ranking.items()}
    )
    if prepared_arrays is not None:
        arrays.update(
            {
                f"prepared__{key}": np.asarray(value)
                for key, value in prepared_arrays.items()
            }
        )
    np.savez_compressed(path, **arrays)


def load_artifact(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    result["metadata"] = json.loads(str(result.pop("metadata_json")))
    result["ranking"] = {
        name.removeprefix("ranking__"): result.pop(name)
        for name in list(result)
        if name.startswith("ranking__")
    }
    result["prepared_arrays"] = {
        name.removeprefix("prepared__"): result.pop(name)
        for name in list(result)
        if name.startswith("prepared__")
    }
    return result


def validate_contract(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_arrays: Mapping[str, np.ndarray] | None = None,
    right_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    errors = []
    for key in CONTRACT_KEYS:
        if left.get(key) != right.get(key):
            errors.append(key)
    left_prepared = left.get("prepared_tensors", {})
    right_prepared = right.get("prepared_tensors", {})
    common = sorted(
        name
        for name in set(left_prepared) & set(right_prepared)
        if name.startswith("features.")
    )
    mismatched = []
    left_arrays = {} if left_arrays is None else left_arrays
    right_arrays = {} if right_arrays is None else right_arrays
    for name in common:
        if (
            left_prepared[name]["semantic_sha256"]
            == right_prepared[name]["semantic_sha256"]
        ):
            continue
        tolerance = NUMERIC_PREPARED_TOLERANCES.get(name)
        if tolerance is None or name not in left_arrays or name not in right_arrays:
            mismatched.append(name)
            continue
        if not np.allclose(
            left_arrays[name],
            right_arrays[name],
            atol=tolerance["atol"],
            rtol=tolerance["rtol"],
        ):
            mismatched.append(name)
    if not common:
        errors.append("prepared_tensors(no common model features)")
    if mismatched:
        errors.append("prepared_tensors(" + ", ".join(mismatched) + ")")
    if errors:
        raise ContractMismatchError("inference contract mismatch: " + "; ".join(errors))


def _align_atoms(
    left_ids: np.ndarray, right_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_keys = [tuple(row) for row in np.asarray(left_ids)]
    right_lookup = {
        tuple(row): index for index, row in enumerate(np.asarray(right_ids))
    }
    if len(right_lookup) != len(right_ids):
        raise ValueError("right artifact has duplicate atom identities")
    missing_right = [key for key in left_keys if key not in right_lookup]
    missing_left = sorted(set(right_lookup) - set(left_keys))
    if missing_right or missing_left:
        raise ValueError(
            "atom identity sets differ: "
            f"missing_right={len(missing_right)}, missing_left={len(missing_left)}"
        )
    left_indices = np.arange(len(left_keys))
    right_indices = np.asarray([right_lookup[key] for key in left_keys])
    ca_mask = np.asarray([key[3] == "CA" for key in left_keys])
    return left_indices, right_indices, ca_mask


def kabsch_transform(
    reference: np.ndarray, mobile: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if (
        reference.shape != mobile.shape
        or reference.ndim != 2
        or reference.shape[1] != 3
    ):
        raise ValueError("Kabsch inputs must both have shape (atom, 3)")
    if reference.shape[0] == 0:
        raise ValueError("Kabsch alignment requires at least one atom")
    ref_centroid = reference.mean(axis=0)
    mob_centroid = mobile.mean(axis=0)
    covariance = (mobile - mob_centroid).T @ (reference - ref_centroid)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    translation = ref_centroid - mob_centroid @ rotation
    return rotation, translation


def _coordinate_metrics(
    reference: np.ndarray, mobile: np.ndarray, ca_mask: np.ndarray
) -> dict[str, Any]:
    rotation, translation = kabsch_transform(reference, mobile)
    aligned = mobile @ rotation + translation
    pre = np.linalg.norm(mobile - reference, axis=-1)
    post = np.linalg.norm(aligned - reference, axis=-1)
    result: dict[str, Any] = {
        "all_atom_kabsch_rmsd": float(np.sqrt(np.mean(post**2))),
        "aligned_coordinate_mae": float(np.mean(post)),
        "aligned_coordinate_max": float(np.max(post)),
        "raw_coordinate_mae": float(np.mean(pre)),
        "raw_coordinate_max": float(np.max(pre)),
        "atom_count": int(reference.shape[0]),
        "ca_atom_count": int(ca_mask.sum()),
    }
    if np.any(ca_mask):
        ca_rotation, ca_translation = kabsch_transform(
            reference[ca_mask], mobile[ca_mask]
        )
        ca_delta = mobile[ca_mask] @ ca_rotation + ca_translation - reference[ca_mask]
        result["ca_kabsch_rmsd"] = float(np.sqrt(np.mean(np.sum(ca_delta**2, axis=-1))))
    else:
        result["ca_kabsch_rmsd"] = None
    if reference.shape[0] > 1:
        ref_dist = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)
        mob_dist = np.linalg.norm(mobile[:, None] - mobile[None, :], axis=-1)
        triangle = np.triu_indices(reference.shape[0], k=1)
        distance_delta = np.abs(ref_dist[triangle] - mob_dist[triangle])
        result["pair_distance_mae"] = float(np.mean(distance_delta))
        result["pair_distance_max"] = float(np.max(distance_delta))
    else:
        result["pair_distance_mae"] = 0.0
        result["pair_distance_max"] = 0.0
    return result


def _tensor_drift(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    if left.shape != right.shape:
        raise ValueError(f"tensor shapes differ: {left.shape} != {right.shape}")
    delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    flat_left = np.asarray(left, dtype=np.float64).reshape(-1)
    flat_right = np.asarray(right, dtype=np.float64).reshape(-1)
    correlation: float | None
    if flat_left.size > 1 and np.std(flat_left) > 0 and np.std(flat_right) > 0:
        correlation = float(np.corrcoef(flat_left, flat_right)[0, 1])
    else:
        correlation = None
    return {
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "correlation": correlation,
    }


def compare_artifacts(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    validate_contract(
        left["metadata"],
        right["metadata"],
        left.get("prepared_arrays"),
        right.get("prepared_arrays"),
    )
    left_indices, right_indices, ca_mask = _align_atoms(
        left["atom_ids"], right["atom_ids"]
    )
    if left["coords"].shape[0] != right["coords"].shape[0]:
        raise ValueError("candidate counts differ")
    coordinates = []
    for candidate in range(left["coords"].shape[0]):
        coordinates.append(
            _coordinate_metrics(
                left["coords"][candidate, left_indices],
                right["coords"][candidate, right_indices],
                ca_mask,
            )
        )
    tensor_drift = {
        name: _tensor_drift(left[name], right[name]) for name in ("pae", "pde", "plddt")
    }
    ranking_keys = sorted(set(left["ranking"]) & set(right["ranking"]))
    ranking_drift = {
        name: _tensor_drift(left["ranking"][name], right["ranking"][name])
        for name in ranking_keys
    }
    left_prepared_arrays = left.get("prepared_arrays", {})
    right_prepared_arrays = right.get("prepared_arrays", {})
    prepared_numeric_drift = {
        name: _tensor_drift(left_prepared_arrays[name], right_prepared_arrays[name])
        for name in sorted(set(left_prepared_arrays) & set(right_prepared_arrays))
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "left_backend": left["metadata"].get("backend"),
        "right_backend": right["metadata"].get("backend"),
        "fixture_id": left["metadata"]["fixture_id"],
        "sampler_randomness": left["metadata"].get("sampler_randomness"),
        "coordinates": coordinates,
        "confidence_tensor_drift": tensor_drift,
        "ranking_and_clash_drift": ranking_drift,
        "prepared_numeric_drift": prepared_numeric_drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_artifacts(load_artifact(args.left), load_artifact(args.right))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
