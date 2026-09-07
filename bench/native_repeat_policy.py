"""Prospectively frozen, fixed-tape native-repeat engineering allowances.

Adapters must independently validate the comparison identities and explicitly
canonicalize confidence leaves before constructing records. ``provenance``
names the validated native comparison target, including its source, not the
candidate's executing framework. Each execution's source/runtime and validation
evidence belong in its immutable report, identified by ``report_sha256``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bench.entity_parity import compare_entity_parity

_IDENTITIES = {"input", "checkpoint", "source", "effective_policy", "tape"}
_POLICY = {
    "version": 1,
    "samples": 5,
    "repeats": 3,
    "factor": 3.0,
    "rmsd_floor": 0.05,
    "atol": 1e-4,
    "rtol": 1e-4,
}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    report_sha256: str
    provenance: Mapping[str, str]
    coordinates: np.ndarray
    atom_keys: Sequence[str]
    entity_labels: Sequence[str]
    mask: np.ndarray
    confidence: Mapping[str, np.ndarray]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(_digest([value.dtype.str, value.shape]).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _binding(run: RunRecord) -> dict[str, Any]:
    if not isinstance(run.run_id, str) or not run.run_id:
        raise ValueError("run_id must be a nonempty string")
    if not _sha256(run.report_sha256):
        raise ValueError("report_sha256 must identify the validated execution report")
    if not _IDENTITIES.issubset(run.provenance) or not all(
        isinstance(key, str) and _sha256(value) for key, value in run.provenance.items()
    ):
        raise ValueError(
            "provenance requires validated input/checkpoint/source/"
            "effective_policy/tape SHA256 identities"
        )
    coordinates, mask = np.asarray(run.coordinates), np.asarray(run.mask)
    if coordinates.ndim != 3 or coordinates.shape[0] != 5:
        raise ValueError("coordinates require five sample-index-paired outputs")
    if any(not isinstance(key, str) or not key for key in run.atom_keys):
        raise ValueError("atom keys must be nonempty strings in native order")
    if any(not isinstance(label, str) or not label for label in run.entity_labels):
        raise ValueError("entity labels must be nonempty strings")
    # This also rejects duplicate keys, malformed masks and nonfinite coordinates.
    compare_entity_parity(
        coordinates,
        coordinates,
        run.atom_keys,
        run.atom_keys,
        run.entity_labels,
        run.entity_labels,
        mask,
        mask,
    )
    labels = np.asarray(run.entity_labels)
    for label in set(run.entity_labels):
        if not mask[:, labels == label].any(axis=1).all():
            raise ValueError(
                f"each sample must contain valid atoms for entity {label!r}"
            )
    if not run.confidence or not all(
        isinstance(name, str) and name for name in run.confidence
    ):
        raise ValueError("confidence must contain all named raw and extracted leaves")
    arrays = {name: np.asarray(value) for name, value in run.confidence.items()}
    if any(array.dtype.kind not in "fbiuUS" for array in arrays.values()):
        raise ValueError("confidence rejects object, complex and unknown dtypes")
    if not any(array.dtype.kind == "f" for array in arrays.values()):
        raise ValueError("confidence requires at least one floating leaf")
    return {
        "run_id": run.run_id,
        "report_sha256": run.report_sha256,
        "provenance": dict(run.provenance),
        "coordinates": _array_digest(coordinates),
        "mask": _array_digest(mask),
        "atom_keys": _digest(list(run.atom_keys)),
        "entity_labels": _digest(list(run.entity_labels)),
        "confidence": {name: _array_digest(value) for name, value in arrays.items()},
    }


def _compare(
    left: RunRecord, right: RunRecord, additions: Mapping[str, float]
) -> dict[str, Any]:
    if dict(left.provenance) != dict(right.provenance):
        raise ValueError("validated comparison provenance differs (including tape)")
    if list(left.atom_keys) != list(right.atom_keys):
        raise ValueError("atom keys and order must match exactly; no reordering")
    if set(left.confidence) != set(right.confidence):
        raise ValueError("confidence leaf sets must match exactly")
    coordinates = compare_entity_parity(
        left.coordinates,
        right.coordinates,
        left.atom_keys,
        right.atom_keys,
        left.entity_labels,
        right.entity_labels,
        left.mask,
        right.mask,
    )
    leaves = {}
    for name in sorted(left.confidence):
        ref, candidate = (
            np.asarray(left.confidence[name]),
            np.asarray(right.confidence[name]),
        )
        if ref.shape != candidate.shape or ref.dtype != candidate.dtype:
            raise ValueError(f"confidence {name}: shape and dtype must match exactly")
        if ref.dtype.kind != "f":
            if not np.array_equal(ref, candidate):
                raise ValueError(
                    f"confidence {name}: nonfloating values must match exactly"
                )
            leaves[name] = {
                "exact": True,
                "max_abs_error": None,
                "strict_pass": True,
                "practical_pass": True,
            }
            continue
        if not all(
            np.array_equal(check(ref), check(candidate))
            for check in (np.isnan, np.isposinf, np.isneginf)
        ):
            raise ValueError(
                f"confidence {name}: NaN and signed infinity positions differ"
            )
        valid = np.isfinite(ref)
        ref, candidate = (
            ref[valid].astype(np.float64),
            candidate[valid].astype(np.float64),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            error = np.abs(candidate - ref)
        if not np.isfinite(error).all():
            raise ValueError(f"confidence {name}: nonfinite numeric difference")
        strict = 1e-4 + 1e-4 * np.abs(ref)
        practical = strict + additions.get(name, 0.0)
        leaves[name] = {
            "max_abs_error": float(error.max(initial=0.0)),
            "finite_entries": int(ref.size),
            "strict_pass": bool(np.all(error <= strict)),
            "practical_pass": bool(np.all(error <= practical)),
            "strict_failures": int(np.count_nonzero(error > strict)),
            "practical_failures": int(np.count_nonzero(error > practical)),
            "practical_limit_min": float(practical.min()) if ref.size else None,
            "practical_limit_max": float(practical.max()) if ref.size else None,
        }
    return {
        "left": left.run_id,
        "right": right.run_id,
        "coordinates": coordinates,
        "confidence": leaves,
        "strict_structure_pass": all(
            value <= 0.05 for value in coordinates["entity_max_rmsd"].values()
        ),
        "strict_confidence_pass": all(leaf["strict_pass"] for leaf in leaves.values()),
    }


def calibrate_native_repeats(runs: Sequence[RunRecord]) -> dict[str, Any]:
    """Freeze every native pair and its allowances before candidate evaluation.

    The first run is the frozen candidate reference. The JSON-serializable return
    value is content sealed; persist it before running candidates. This helper
    cannot enforce external execution chronology or validate external reports.
    """
    if len(runs) < 3:
        raise ValueError("calibration requires at least three native executions")
    bindings = [_binding(run) for run in runs]
    if len({run.run_id for run in runs}) != len(runs) or len(
        {run.report_sha256 for run in runs}
    ) != len(runs):
        raise ValueError("native executions require distinct run IDs and reports")
    pairs = [
        _compare(left, right, {}) for left, right in itertools.combinations(runs, 2)
    ]
    entities = {
        label: max(pair["coordinates"]["entity_max_rmsd"][label] for pair in pairs)
        for label in pairs[0]["coordinates"]["entity_max_rmsd"]
    }
    confidence = {
        name: max(pair["confidence"][name]["max_abs_error"] for pair in pairs)
        for name, leaf in pairs[0]["confidence"].items()
        if leaf["max_abs_error"] is not None
    }
    result = {
        "policy": dict(_POLICY),
        "native_runs": bindings,
        "native_pairs": pairs,
        "entity_native_max_rmsd": entities,
        "entity_limits": {
            label: max(0.05, 3.0 * value) for label, value in entities.items()
        },
        "confidence_native_max_abs": confidence,
        "confidence_additions": {
            name: 3.0 * value for name, value in confidence.items()
        },
    }
    if not all(np.isfinite(value) for value in result["confidence_additions"].values()):
        raise ValueError("calibration confidence allowance overflow")
    reference = _compare(runs[0], runs[0], result["confidence_additions"])
    result["reference_run_id"] = runs[0].run_id
    result["confidence_reference_limits"] = {
        name: {"min": leaf["practical_limit_min"], "max": leaf["practical_limit_max"]}
        for name, leaf in reference["confidence"].items()
        if name in confidence
    }
    result["calibration_sha256"] = _digest(result)
    return result


def evaluate_candidate(
    calibration: Mapping[str, Any], reference: RunRecord, candidate: RunRecord
) -> dict[str, Any]:
    """Evaluate the original and practical gates without changing calibration."""
    sealed = dict(calibration)
    expected = sealed.pop("calibration_sha256", None)
    if _digest(sealed) != expected or sealed.get("policy") != _POLICY:
        raise ValueError("calibration was changed or has an unsupported policy")
    reference_binding, candidate_binding = _binding(reference), _binding(candidate)
    if reference_binding != sealed["native_runs"][0]:
        raise ValueError(
            "reference does not match a frozen native report and array binding"
        )
    result = _compare(reference, candidate, sealed["confidence_additions"])
    result.update(
        {
            "calibration_sha256": expected,
            "reference": reference_binding,
            "candidate": candidate_binding,
            "entity_limits": dict(sealed["entity_limits"]),
            "confidence_additions": dict(sealed["confidence_additions"]),
            "practical_structure_pass": all(
                value <= sealed["entity_limits"][label]
                for label, value in result["coordinates"]["entity_max_rmsd"].items()
            ),
            "practical_confidence_pass": all(
                leaf["practical_pass"] for leaf in result["confidence"].values()
            ),
        }
    )
    result["strict_pass"] = (
        result["strict_structure_pass"] and result["strict_confidence_pass"]
    )
    result["practical_pass"] = (
        result["practical_structure_pass"] and result["practical_confidence_pass"]
    )
    return result
