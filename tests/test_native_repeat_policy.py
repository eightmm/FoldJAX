import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from bench.native_repeat_policy import (
    RunRecord,
    calibrate_native_repeats,
    evaluate_candidate,
)


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _run(name, confidence=0.0, displacement=0.0):
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    )
    coordinates = np.broadcast_to(points, (5, 4, 3)).copy()
    coordinates[:, 3, 0] += displacement
    return RunRecord(
        run_id=name,
        report_sha256=_hash(name),
        provenance={
            key: _hash(key)
            for key in ("input", "checkpoint", "source", "effective_policy", "tape")
        },
        coordinates=coordinates,
        atom_keys=["A:1", "A:2", "A:3", "L:1"],
        entity_labels=["A", "A", "A", "L"],
        mask=np.ones((5, 4), dtype=bool),
        confidence={
            "raw/pae": np.full((5, 4), confidence, np.float64),
            "extracted/score": np.full(5, confidence, np.float64),
            "chain_id": np.array(["A", "L"]),
            "valid": np.array([True, False]),
            "count": np.array([2**63 + 1], np.uint64),
        },
    )


def _panel():
    return [_run("n0"), _run("n1", 0.01, 0.03), _run("n2", -0.02, -0.04)]


def test_calibration_keeps_all_pairs_and_pointwise_numeric_three_factor():
    runs = _panel()
    frozen = calibrate_native_repeats(runs)
    assert [(pair["left"], pair["right"]) for pair in frozen["native_pairs"]] == [
        ("n0", "n1"),
        ("n0", "n2"),
        ("n1", "n2"),
    ]
    assert frozen["confidence_native_max_abs"]["raw/pae"] == pytest.approx(0.03)
    assert frozen["confidence_additions"]["raw/pae"] == pytest.approx(0.09)
    assert frozen["confidence_reference_limits"]["raw/pae"] == pytest.approx(
        {"min": 0.0901, "max": 0.0901}
    )
    for label, native_max in frozen["entity_native_max_rmsd"].items():
        assert frozen["entity_limits"][label] == max(0.05, 3 * native_max)
    report = evaluate_candidate(frozen, runs[0], _run("candidate", 0.09))
    assert report["practical_pass"]
    assert not report["strict_pass"]
    assert report["confidence"]["raw/pae"]["practical_limit_min"] == pytest.approx(
        0.0901
    )
    assert json.loads(json.dumps(frozen)) == frozen
    assert json.loads(json.dumps(report)) == report


def test_single_entry_failure_is_not_hidden_by_average_or_other_leaves():
    runs = _panel()
    frozen = calibrate_native_repeats(runs)
    candidate = _run("candidate")
    candidate.confidence["raw/pae"][4, 3] = 0.09011
    report = evaluate_candidate(frozen, runs[0], candidate)
    assert not report["practical_pass"]
    assert report["confidence"]["raw/pae"]["practical_failures"] == 1
    assert report["confidence"]["extracted/score"]["practical_pass"]


def test_pointwise_relative_allowance_uses_reference_not_global_maximum():
    runs = [_run(f"n{i}") for i in range(3)]
    for run in runs:
        run.confidence["raw/pae"][:, 0] = 100.0
    frozen = calibrate_native_repeats(runs)
    candidate = replace(_run("candidate"), confidence=copy.deepcopy(runs[0].confidence))
    candidate.confidence["raw/pae"][:, 0] += 0.01
    candidate.confidence["raw/pae"][:, 1] += 0.001
    report = evaluate_candidate(frozen, runs[0], candidate)
    assert report["confidence"]["raw/pae"]["practical_failures"] == 5
    assert not report["practical_pass"]


def test_whole_system_fit_keeps_entity_displacement_visible():
    runs = [_run(f"n{i}") for i in range(3)]
    frozen = calibrate_native_repeats(runs)
    report = evaluate_candidate(frozen, runs[0], _run("candidate", displacement=0.5))
    assert report["coordinates"]["entity_max_rmsd"]["L"] > 0.05
    assert not report["practical_structure_pass"]


def test_sample_indices_are_not_rematched():
    runs = [_run(f"n{i}") for i in range(3)]
    for run in runs:
        run.coordinates[:, 3, 0] += np.arange(5)
    frozen = calibrate_native_repeats(runs)
    candidate = replace(_run("candidate"), coordinates=runs[0].coordinates[::-1].copy())
    assert not evaluate_candidate(frozen, runs[0], candidate)[
        "practical_structure_pass"
    ]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_requires_at_least_three_native_executions(count):
    with pytest.raises(ValueError, match="at least three"):
        calibrate_native_repeats(_panel()[:count])


def test_duplicate_report_is_not_a_new_execution():
    runs = _panel()
    runs[2] = replace(runs[2], report_sha256=runs[1].report_sha256)
    with pytest.raises(ValueError, match="distinct"):
        calibrate_native_repeats(runs)


def test_four_repeats_keep_six_pairs_and_first_reference_is_frozen():
    runs = _panel() + [_run("n3", 0.03)]
    frozen = calibrate_native_repeats(runs)
    assert len(frozen["native_pairs"]) == 6
    with pytest.raises(ValueError, match="frozen native"):
        evaluate_candidate(frozen, runs[1], _run("candidate"))


@pytest.mark.parametrize(
    "key", ["input", "checkpoint", "source", "effective_policy", "tape"]
)
def test_each_provenance_identity_is_required_and_must_match(key):
    runs = _panel()
    runs[2].provenance[key] = _hash("different")
    with pytest.raises(ValueError, match="provenance differs"):
        calibrate_native_repeats(runs)
    del runs[2].provenance[key]
    with pytest.raises(ValueError, match="requires validated"):
        calibrate_native_repeats(runs)


@pytest.mark.parametrize(
    "mutation", ["gate", "negative_gate", "nan_gate", "native_pair", "array", "report"]
)
def test_frozen_calibration_and_reference_bindings_cannot_silently_change(mutation):
    runs = _panel()
    frozen = calibrate_native_repeats(runs)
    if mutation == "gate":
        frozen["entity_limits"]["L"] = 100.0
    elif mutation == "negative_gate":
        frozen["entity_limits"]["L"] = -1.0
    elif mutation == "nan_gate":
        frozen["entity_limits"]["L"] = float("nan")
    elif mutation == "native_pair":
        frozen["native_pairs"][0]["confidence"]["raw/pae"]["max_abs_error"] = 10
    elif mutation == "array":
        runs[0].confidence["raw/pae"][0, 0] = 1
    else:
        runs[0] = replace(runs[0], report_sha256=_hash("other"))
    with pytest.raises(ValueError, match="calibration was changed|frozen native|JSON"):
        evaluate_candidate(frozen, runs[0], _run("candidate"))


@pytest.mark.parametrize(
    "mutation",
    ["missing_leaf", "order", "mask", "missing_entity", "nan_coords", "four_samples"],
)
def test_candidate_correspondence_and_validity_fail_closed(mutation):
    runs = _panel()
    frozen = calibrate_native_repeats(runs)
    candidate = _run("candidate")
    if mutation == "missing_leaf":
        del candidate.confidence["raw/pae"]
    elif mutation == "order":
        candidate = replace(candidate, atom_keys=list(reversed(candidate.atom_keys)))
    elif mutation == "mask":
        candidate.mask[0, 0] = False
    elif mutation == "missing_entity":
        candidate.mask[:, 3] = False
    elif mutation == "nan_coords":
        candidate.coordinates[0, 0, 0] = np.nan
    else:
        candidate = replace(candidate, coordinates=candidate.coordinates[:4])
    with pytest.raises(ValueError):
        evaluate_candidate(frozen, runs[0], candidate)


def test_negative_coordinates_are_valid_not_negative_distances():
    runs = _panel()
    for run in runs:
        run.coordinates[:] -= 20
    frozen = calibrate_native_repeats(runs)
    assert evaluate_candidate(frozen, runs[0], _run("candidate"))["practical_pass"]


@pytest.mark.parametrize(
    "leaf,value",
    [
        ("count", np.array([2**63], np.uint64)),
        ("valid", np.array([True, True])),
        ("chain_id", np.array(["L", "A"])),
        ("raw/pae", np.ones((5, 4), dtype=object)),
        ("raw/pae", np.ones((5, 4), dtype=complex)),
        ("raw/pae", np.ones((5, 4), dtype=np.float32)),
    ],
)
def test_nonnumeric_exactness_and_unknown_dtype_rejection(leaf, value):
    runs = _panel()
    frozen = calibrate_native_repeats(runs)
    candidate = _run("candidate")
    candidate.confidence[leaf] = value
    with pytest.raises(ValueError):
        evaluate_candidate(frozen, runs[0], candidate)


def test_matching_undefined_positions_allowed_but_signed_infinity_changes_rejected():
    runs = _panel()
    for run in runs:
        run.confidence["undefined"] = np.array([np.nan, np.inf, -np.inf, 0.0])
    frozen = calibrate_native_repeats(runs)
    candidate = _run("candidate")
    candidate.confidence["undefined"] = runs[0].confidence["undefined"].copy()
    assert evaluate_candidate(frozen, runs[0], candidate)["practical_pass"]
    candidate.confidence["undefined"][1] = -np.inf
    with pytest.raises(ValueError, match="positions differ"):
        evaluate_candidate(frozen, runs[0], candidate)


def test_candidate_tape_mismatch_and_no_silent_recalibration():
    runs = _panel()
    frozen = calibrate_native_repeats(runs)
    before = json.dumps(frozen, sort_keys=True)
    failed = evaluate_candidate(frozen, runs[0], _run("bad", 2.0))
    assert not failed["practical_pass"]
    assert before == json.dumps(frozen, sort_keys=True)
    candidate = _run("different_tape")
    candidate.provenance["tape"] = _hash("new_tape")
    with pytest.raises(ValueError, match="provenance differs"):
        evaluate_candidate(frozen, runs[0], candidate)
