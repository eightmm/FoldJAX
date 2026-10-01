"""Metric functions of ``bench/stage_parity.py``: NumPy only, no weights.

Each metric is checked against a value worked out by hand, plus one case it
must not hide (a reflection a Kabsch fit could "undo", a rearrangement a
multiset check should forgive, an arithmetic change it should not).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench import stage_parity as sp


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def test_exact_match_fraction_counts_equal_elements() -> None:
    assert sp.exact_match_fraction([1, 2, 3, 4], [1, 2, 0, 4]) == 0.75
    assert sp.exact_match_fraction(np.zeros((0,), int), np.zeros((0,), int)) == 1.0
    assert sp.exact_match_fraction(["CA", "CB"], ["CA", "CG"]) == 0.5


def test_exact_match_fraction_refuses_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        sp.exact_match_fraction([1, 2], [1, 2, 3])


def test_max_abs_and_relative_rms_by_hand() -> None:
    native = np.array([3.0, 4.0])
    port = np.array([3.0, 5.0])
    assert sp.max_abs(port, native) == 1.0
    # rms(diff) = sqrt(1/2); rms(native) = sqrt(25/2)  ->  1/5
    assert sp.relative_rms(port, native) == pytest.approx(0.2)
    assert sp.relative_rms(native, native) == 0.0


def test_relative_rms_of_a_zero_reference() -> None:
    assert sp.relative_rms(np.zeros(3), np.zeros(3)) == 0.0
    assert sp.relative_rms(np.ones(3), np.zeros(3)) == float("inf")


def test_metrics_are_computed_in_float64() -> None:
    native = np.array([16777216.0], np.float32)  # 2**24
    port = np.array([16777218.0], np.float32)
    assert sp.max_abs(port, native) == 2.0


def test_array_parity_picks_the_metric_by_kind() -> None:
    categorical = sp.array_parity(np.array([1, 2, 3]), np.array([1, 2, 4]))
    assert categorical["exact_match_fraction"] == pytest.approx(2 / 3)
    assert "max_abs" not in categorical

    floats = sp.array_parity(np.array([1.0, 2.0]), np.array([1.0, 2.5]))
    assert floats["max_abs"] == 0.5
    assert "exact_match_fraction" not in floats

    # A mask stored as float32 on one side and int64 on the other is compared
    # numerically and exactly.
    mixed = sp.array_parity(np.array([1.0, 0.0]), np.array([1, 0]))
    assert mixed["max_abs"] == 0.0
    assert mixed["exact_match_fraction"] == 1.0

    mismatch = sp.array_parity(np.zeros(2), np.zeros(3))
    assert mismatch["shape_equal"] is False
    assert "max_abs" not in mismatch


def test_summarize_features_reports_the_worst_array() -> None:
    records = {
        "a": sp.array_parity(np.array([1, 2]), np.array([1, 2])),
        "b": sp.array_parity(np.array([1, 2]), np.array([1, 3])),
        "c": sp.array_parity(np.array([0.0, 1.0]), np.array([0.0, 1.25])),
        "d": sp.array_parity(np.zeros(2), np.zeros(4)),
    }
    summary = sp.summarize_features(records)
    assert summary["min_exact_match_fraction"] == 0.5
    assert summary["arrays_not_exact"] == ["b"]
    assert summary["max_float_max_abs"] == 0.25
    assert summary["shape_mismatches"] == ["d"]


def test_kabsch_rmsd_removes_a_rigid_motion_only() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(40, 3)) * 5.0
    moved = reference @ _rotation(np.array([1.0, 2.0, 0.5]), 1.1).T + [3.0, -2.0, 7.0]
    assert sp.kabsch_rmsd(moved, reference) == pytest.approx(0.0, abs=1e-9)

    # Moving one atom is a non-rigid change: the fit cannot remove it entirely.
    shifted = reference.copy()
    shifted[0] += [0.0, 0.0, 1.0]
    assert 0.0 < sp.kabsch_rmsd(shifted, reference) < 1.0


def test_kabsch_rmsd_does_not_fit_a_reflection() -> None:
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(30, 3)) * 4.0
    mirrored = reference * np.array([1.0, 1.0, -1.0])
    assert sp.kabsch_rmsd(mirrored, reference) > 0.5


def test_kabsch_rmsd_uses_only_the_masked_atoms() -> None:
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(10, 3))
    port = reference.copy()
    port[-1] += 50.0
    mask = np.ones(10, bool)
    mask[-1] = False
    assert sp.kabsch_rmsd(port, reference, mask) == pytest.approx(0.0, abs=1e-9)
    assert sp.kabsch_rmsd(port, reference) > 1.0
    with pytest.raises(ValueError, match="at least three atoms"):
        sp.kabsch_rmsd(port, reference, np.eye(10, dtype=bool)[0])


def test_per_sample_rmsd_and_unaligned_displacement() -> None:
    rng = np.random.default_rng(3)
    native = rng.normal(size=(2, 12, 3))
    port = native.copy()
    port[1, 4] += [0.0, 0.3, 0.4]
    values = sp.per_sample_rmsd(port, native)
    assert values[0] == pytest.approx(0.0, abs=1e-12)
    assert values[1] > 0.0
    assert sp.max_unaligned_displacement(port, native) == pytest.approx(0.5)


def test_ca_mask_from_names_requires_carbon() -> None:
    names = np.array(["N", "CA", "C", " CA ", "CA"])
    elements = np.array(["N", "C", "C", "C", "CA"])  # last: calcium named CA
    assert sp.ca_mask_from_names(names, elements).tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]


def test_kabsch_rmsd_measures_a_subset_without_refitting() -> None:
    rng = np.random.default_rng(6)
    reference = rng.normal(size=(20, 3)) * 3.0
    port = reference.copy()
    port[0] += [2.0, 0.0, 0.0]
    subset = np.zeros(20, bool)
    subset[10:] = True
    # Fitted on all atoms, the outlier tilts the fit, so the untouched subset
    # is not at zero -- unlike a refit on the subset alone.
    no_refit = sp.kabsch_rmsd(port, reference, None, subset)
    refit = sp.kabsch_rmsd(port, reference, subset)
    assert refit == pytest.approx(0.0, abs=1e-9)
    assert no_refit > 1e-3


def test_coordinate_metrics_reports_ca_and_all_atom() -> None:
    rng = np.random.default_rng(4)
    native = rng.normal(size=(1, 8, 3)) * 3.0
    port = native.copy()
    shift = np.array([0.3, -0.2, 0.5])
    port[0] += shift  # a pure translation: zero after the fit
    port[0, 7] += [1.0, 0.0, 0.0]
    ca = np.array([True, False, True, False, True, False, True, False])
    record = sp.coordinate_metrics(port, native, ca_mask=ca)
    assert record["ca_atoms"] == 4
    assert record["all_atom_rmsd_angstrom"][0] > 0.0
    # The CA value comes from the same all-atom fit, so the non-CA outlier
    # still leaves a (smaller) CA residual.
    assert 0.0 < record["ca_rmsd_angstrom"][0] < record["all_atom_rmsd_angstrom"][0]
    assert record["max_unaligned_displacement_angstrom"] == pytest.approx(
        np.linalg.norm(shift + [1.0, 0.0, 0.0])
    )


def test_multiset_delta_forgives_rearrangement_but_not_arithmetic() -> None:
    rng = np.random.default_rng(5)
    weight = rng.normal(size=(4, 6)).astype(np.float32)
    bias = rng.normal(size=(6,)).astype(np.float32)
    # Transpose + concatenate: the same numbers, arranged differently.
    rearranged = [np.concatenate([weight.T.ravel(), bias])]
    same = sp.multiset_delta([weight, bias], rearranged)
    assert same["comparable"] and same["sorted_max_abs"] == 0.0

    scaled = sp.multiset_delta([weight, bias], [weight * 1.001, bias])
    assert scaled["sorted_max_abs"] > 0.0

    dropped = sp.multiset_delta([weight, bias], [weight])
    assert dropped["comparable"] is False

    # A boolean flag the converter adds is counted, not compared.
    flagged = sp.multiset_delta([weight, bias], rearranged + [np.ones(3, bool)])
    assert flagged["comparable"] and flagged["sorted_max_abs"] == 0.0
    assert flagged["port_non_float_elements"] == 3


def test_multiset_containment_allows_unused_checkpoint_tensors() -> None:
    rng = np.random.default_rng(7)
    used = rng.normal(size=(5, 4)).astype(np.float32)
    unused = rng.normal(size=(7,)).astype(np.float32)
    # Converted = the used tensor, transposed; the checkpoint also holds a
    # tensor inference never reads.
    clean = sp.multiset_containment([used, unused], [used.T])
    assert clean["port_values_absent_from_checkpoint"] == 0
    assert clean["max_distance_to_nearest_checkpoint_value"] == 0.0
    assert clean["multiplicity_excess"] == 0
    assert clean["checkpoint_float_elements"] == 27

    shifted = used.copy()
    shifted[0, 0] += np.float32(0.5)
    moved = sp.multiset_containment([used, unused], [shifted])
    assert moved["port_values_absent_from_checkpoint"] == 1
    assert 0.0 < moved["max_distance_to_nearest_checkpoint_value"] <= 0.5

    # A tensor used twice is present, but beyond its multiplicity.
    twice = sp.multiset_containment([used], [used, used])
    assert twice["port_values_absent_from_checkpoint"] == 0
    assert twice["multiplicity_excess"] == used.size


def test_tracking_state_records_reads_not_membership_tests() -> None:
    state = sp.TrackingState({"a": np.ones(2), "b": np.zeros(3), "c": np.ones(1)})
    assert "b" in state
    _ = state["a"]
    _ = state.get("c")
    _ = state.get("missing")
    coverage = sp.checkpoint_coverage(state)
    assert coverage["tensors_read"] == 2
    assert coverage["elements_read"] == 3
    assert coverage["unread_tensors"] == ["b"]


def test_compare_parameter_trees_judges_in_the_stored_dtype() -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    mapped = {"block": {"w": np.array([1.0, 1.001], np.float32)}, "x": np.array([2.0])}
    # bf16 cannot tell 1.0 from 1.001, so a bf16 store of the mapped values is
    # exact in its own dtype.
    stored = {
        "block": {"w": mapped["block"]["w"].astype(ml_dtypes.bfloat16)},
        "x": np.array([2.5]),
    }
    report = sp.compare_parameter_trees(stored, mapped)
    assert report["groups"]["block"]["max_abs"] == 0.0
    assert report["groups"]["x"]["max_abs"] == 0.5
    assert report["max_abs"] == 0.5

    with pytest.raises(ValueError, match="parameter trees differ"):
        sp.compare_parameter_trees({"a": np.ones(1)}, {"b": np.ones(1)})


def test_stage_record_and_table_round_trip(tmp_path) -> None:
    report = {
        "model": "demo",
        "stages": {
            "S1_features": sp.not_captured("No port features stored.", needs="x"),
            "S3_trunk": sp.stage_record(
                "measured",
                condition={"backend": "cpu", "matmul_precision": "highest"},
                headline={"metric": "relative RMS", "value": 0.0123456},
                runtime_s=1.5,
            ),
        },
    }
    (tmp_path / "demo.json").write_text(json.dumps(sp._jsonable(report)))
    table = sp.render_table(tmp_path)
    assert "| demo | S1_features | not_captured |" in table
    assert "0.01235" in table
    assert "backend=cpu; matmul_precision=highest" in table
    # A metric spelled with |d| must not split the Markdown row.
    report["stages"]["S3_trunk"]["headline"]["metric"] = "max |d|"
    (tmp_path / "demo.json").write_text(json.dumps(sp._jsonable(report)))
    row = next(
        line for line in sp.render_table(tmp_path).splitlines() if "S3_trunk" in line
    )
    assert row.count(" | ") == 6
    assert "max \\|d\\|" in row
    with pytest.raises(ValueError, match="unknown stage status"):
        sp.stage_record("skipped")


def test_run_stage_records_a_failure_instead_of_raising() -> None:
    def broken() -> dict:
        pytest.fail("fixture missing", pytrace=False)

    record = sp.run_stage("demo", broken)
    assert record["status"] == "error"
    assert "fixture missing" in record["notes"]
