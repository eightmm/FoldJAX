"""Strict, mask-aware parity summaries for atom coordinates and features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bench import structures


def _coordinate_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (samples, atoms, 3)")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one atom/entity")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must contain only finite numeric coordinates")
    return array.astype(np.float64)


def _labels(value: Sequence[Any], name: str, atoms: int) -> list[Any]:
    array = np.asarray(value, dtype=object)
    if array.ndim != 1 or array.size != atoms:
        raise ValueError(f"{name} must have one label per atom")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one entity")
    labels = array.tolist()
    if any(label is None or label == "" for label in labels):
        raise ValueError(f"{name} must not contain empty labels")
    try:
        set(labels)
    except TypeError as error:
        raise ValueError(f"{name} must contain hashable labels") from error
    return labels


def _keys(value: Sequence[Any], name: str, atoms: int) -> list[Any]:
    keys = list(value)
    if len(keys) != atoms:
        raise ValueError(f"{name} must have one key per atom")
    try:
        unique = set(keys)
    except TypeError as error:
        raise ValueError(f"{name} must contain hashable keys") from error
    if len(unique) != len(keys):
        raise ValueError(f"{name} contains duplicate atom keys")
    return keys


def _mask(value: Any, name: str, samples: int, atoms: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (samples, atoms) or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array with shape (samples, atoms)")
    if not array.any():
        raise ValueError(f"{name} contains no valid atoms")
    return array


def _json_value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def compare_entity_parity(
    left_coordinates: Any,
    right_coordinates: Any,
    left_atom_keys: Sequence[Any],
    right_atom_keys: Sequence[Any],
    left_entity_labels: Sequence[Any],
    right_entity_labels: Sequence[Any],
    left_mask: Any,
    right_mask: Any,
) -> dict[str, Any]:
    """Compare coordinate samples after exact key, entity, and mask matching.

    The returned dictionaries contain only Python scalars and lists, so they can
    be written directly with :mod:`json`.  Entity RMSDs are measured under the
    one global superposition fitted on the valid atoms of each sample.
    """
    left = _coordinate_array(left_coordinates, "left_coordinates")
    right = _coordinate_array(right_coordinates, "right_coordinates")
    if left.shape[0] != right.shape[0]:
        raise ValueError("coordinate arrays must have the same sample count")

    left_keys = _keys(left_atom_keys, "left_atom_keys", left.shape[1])
    right_keys = _keys(right_atom_keys, "right_atom_keys", right.shape[1])
    if set(left_keys) != set(right_keys):
        raise ValueError("atom key sets must match exactly")
    right_index = {key: index for index, key in enumerate(right_keys)}
    right_order = np.asarray([right_index[key] for key in left_keys])

    left_labels = _labels(left_entity_labels, "left_entity_labels", left.shape[1])
    right_labels = _labels(right_entity_labels, "right_entity_labels", right.shape[1])
    right_labels = [right_labels[index] for index in right_order]
    if left_labels != right_labels:
        raise ValueError("entity labels must match after atom-key reordering")

    left_valid = _mask(left_mask, "left_mask", left.shape[0], left.shape[1])
    right_valid = _mask(right_mask, "right_mask", right.shape[0], right.shape[1])[
        :, right_order
    ]
    if not np.array_equal(left_valid, right_valid):
        raise ValueError("masks must match after atom-key reordering")

    right = right[:, right_order, :]
    entity_order = list(
        dict.fromkeys(
            label for atom, label in enumerate(left_labels) if left_valid[:, atom].any()
        )
    )
    if not entity_order:
        raise ValueError("entity labels contain no valid entities")
    entity_rmsd = {label: [] for label in entity_order}
    global_rmsd: list[float] = []
    atom_counts: list[int] = []
    coordinates_equal: list[bool] = []
    for sample_index in range(left.shape[0]):
        valid = left_valid[sample_index]
        if not valid.any():
            raise ValueError(f"sample {sample_index} contains no valid atoms")
        aligned_left, aligned_right = structures.superpose(
            left[sample_index, valid], right[sample_index, valid]
        )
        squared_residuals = np.sum((aligned_left - aligned_right) ** 2, axis=-1)
        valid_labels = [label for label, include in zip(left_labels, valid) if include]
        for label in entity_order:
            residuals = np.asarray(
                [
                    residual
                    for candidate, residual in zip(valid_labels, squared_residuals)
                    if candidate == label
                ]
            )
            if residuals.size == 0:
                raise ValueError(
                    f"sample {sample_index} has no valid atoms for entity {label!r}"
                )
            else:
                entity_rmsd[label].append(float(np.sqrt(np.mean(residuals))))
        global_rmsd.append(float(np.sqrt(np.mean(squared_residuals))))
        atom_counts.append(int(valid.sum()))
        coordinates_equal.append(
            bool(np.array_equal(left[sample_index, valid], right[sample_index, valid]))
        )
    return {
        "global_rmsd": global_rmsd,
        "global_max_rmsd": float(max(global_rmsd)),
        "entity_rmsd": {
            _json_value(label): values for label, values in entity_rmsd.items()
        },
        "entity_max_rmsd": {
            _json_value(label): float(
                max(value for value in values if value is not None)
            )
            for label, values in entity_rmsd.items()
        },
        "atom_counts": atom_counts,
        "coordinates_equal": coordinates_equal,
    }


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float | None:
    if not (
        np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number)
    ):
        return None
    if left.size == 0:
        return 0.0
    if np.issubdtype(left.dtype, np.integer) and np.issubdtype(right.dtype, np.integer):
        differences = np.subtract(left.astype(object), right.astype(object))
        return float(
            max(abs(difference) for difference in np.asarray(differences).flat)
        )
    if np.issubdtype(left.dtype, np.complexfloating) or np.issubdtype(
        right.dtype, np.complexfloating
    ):
        return float(
            np.max(np.abs(left.astype(np.complex128) - right.astype(np.complex128)))
        )
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def compare_feature_dicts(
    left_features: Mapping[str, Any], right_features: Mapping[str, Any]
) -> dict[str, Any]:
    """Report exact feature-dictionary agreement and safe numeric errors."""
    left_keys = set(left_features)
    right_keys = set(right_features)
    for side, features in (("left", left_features), ("right", right_features)):
        for key, value in features.items():
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise ValueError(f"{side} feature {key!r} contains nonfinite values")
    report: dict[str, Any] = {
        "missing_from_left": sorted(right_keys - left_keys),
        "missing_from_right": sorted(left_keys - right_keys),
        "shape_mismatches": {},
        "dtype_mismatches": {},
        "value_mismatches": {},
        "max_absolute_error": {},
    }
    for key in sorted(left_keys & right_keys):
        left = np.asarray(left_features[key])
        right = np.asarray(right_features[key])
        shape_equal = left.shape == right.shape
        dtype_equal = left.dtype == right.dtype
        if not shape_equal:
            report["shape_mismatches"][key] = {
                "left": list(left.shape),
                "right": list(right.shape),
            }
        if not dtype_equal:
            report["dtype_mismatches"][key] = {
                "left": str(left.dtype),
                "right": str(right.dtype),
            }
        if not shape_equal:
            continue
        if not np.array_equal(left, right):
            report["value_mismatches"][key] = int(
                np.size(left) - np.count_nonzero(left == right)
            )
        error = _max_abs_error(left, right)
        if error is not None:
            report["max_absolute_error"][key] = error
    report["equal"] = not any(
        report[name]
        for name in (
            "missing_from_left",
            "missing_from_right",
            "shape_mismatches",
            "dtype_mismatches",
            "value_mismatches",
        )
    )
    return report
