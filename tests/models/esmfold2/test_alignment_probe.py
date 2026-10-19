from __future__ import annotations

import numpy as np
import pytest

from bench import esmfold2_alignment_probe as probe


@pytest.fixture(scope="module")
def panel():
    covariance, labels = probe.synthetic_panel(samples=4, seed=17)
    report, arrays = probe.covariance_report(covariance, labels, sweeps=(1, 5, 10, 20))
    return covariance, labels, report, arrays


def test_instrumented_ten_sweeps_match_the_actual_production_rotation(panel):
    _, _, report, arrays = panel
    np.testing.assert_array_equal(
        arrays["rotation_production32"], arrays["jacobi32_s10_rotation"]
    )
    assert report["runs"]["jacobi32_s10"]["probe_vs_production_max_abs"] == 0
    assert report["platform"] == "cpu"
    assert report["full_model_admission"] is None
    assert len(report["production_source_sha256"]) == 64
    assert len(report["probe_source_sha256"]) == 64


def test_residual_and_objective_expose_convergence_without_new_model_gates(panel):
    _, _, report, arrays = panel
    assert (
        report["runs"]["jacobi32_s01"]["metrics"]["eigen_residual_normalized"]["max"]
        > 1e-3
    )
    for sweeps in (5, 10, 20):
        prefix = f"jacobi32_s{sweeps:02}"
        assert np.max(arrays[f"{prefix}_eigen_residual_normalized"]) < 1e-5
        assert np.max(np.abs(arrays[f"{prefix}_objective_normalized_deficit"])) < 2e-6
        assert np.max(arrays[f"{prefix}_proper_determinant_error"]) < 2e-6
    assert (
        report["runs"]["jacobi64_s10"]["metrics"]["eigen_residual_normalized"]["max"]
        < 1e-12
    )
    assert "same FP32-rounded input" in report["oracle_scope"]


def test_degenerate_rotation_differences_do_not_mean_a_worse_objective(panel):
    _, labels, report, arrays = panel
    mask = np.isin(labels, ("rank_one", "reflection_equal"))
    gap = arrays["jacobi32_s10_top_eigen_gap_normalized"][mask]
    assert np.all(gap < probe.WEAK_GAP)
    assert report["weak_gap_cases"] >= int(mask.sum())
    assert arrays["jacobi32_s10_rotation_max_abs_error"][mask].max() > 0.1
    assert (
        np.abs(arrays["jacobi32_s10_objective_normalized_deficit"][mask]).max() < 2e-6
    )
    assert arrays["jacobi32_s10_eigen_residual_normalized"][mask].max() < 1e-5


def test_scaling_panel_uses_finite_covariances_and_preserves_random_rotations(panel):
    covariance, labels, _, arrays = panel
    assert np.isfinite(covariance).all()
    expected = arrays["rotation_production32"][labels == "random"]
    for scale in (1e-25, 1e-12, 1e12, 1e25):
        np.testing.assert_allclose(
            arrays["rotation_production32"][labels == f"scale_{scale:g}"],
            expected,
            atol=2e-5,
            rtol=2e-5,
        )


def test_capture_preserves_matrix_order_and_rounds_the_reference_input(tmp_path):
    path = tmp_path / "covariance.npz"
    covariance = np.random.default_rng(6).normal(size=(2, 5, 3, 3))
    labels = np.asarray([f"step{i // 5}-sample{i % 5}" for i in range(10)])
    np.savez(path, correlation=covariance, labels=labels.reshape((2, 5)))
    values, actual_labels, source_shape = probe.load_covariances(path)
    assert source_shape == covariance.shape
    assert values.dtype == np.float32
    np.testing.assert_array_equal(
        values, covariance.astype(np.float32).reshape(-1, 3, 3)
    )
    np.testing.assert_array_equal(actual_labels, labels)


@pytest.mark.parametrize("fault", ("missing", "shape", "integer", "nan", "labels"))
def test_invalid_captures_fail_before_any_solver(tmp_path, fault):
    path = tmp_path / "bad.npz"
    values = {"correlation": np.ones((2, 3, 3), np.float32)}
    if fault == "missing":
        values = {"other": values["correlation"]}
    elif fault == "shape":
        values["correlation"] = np.ones((2, 4, 4), np.float32)
    elif fault == "integer":
        values["correlation"] = np.ones((2, 3, 3), np.int32)
    elif fault == "nan":
        values["correlation"][0, 0, 0] = np.nan
    else:
        values["labels"] = np.asarray(["only-one"])
    np.savez(path, **values)
    with pytest.raises(ValueError):
        probe.load_covariances(path)


@pytest.mark.parametrize("sweeps", ((), (0,), (True,), (1.5,)))
def test_invalid_sweep_requests_do_not_run_a_solver(sweeps):
    with pytest.raises(ValueError, match="sweep counts"):
        probe.covariance_report(np.eye(3, dtype=np.float32)[None], ["case"], sweeps)


def test_centering_stress_reports_a_shared_fp32_error_separately():
    rows = probe.centering_report()
    assert len(rows) == 9
    normal = [row for row in rows if row["scale"] == 1 and row["translation"] == 0][0]
    assert normal["svd_jacobi_valid_max_abs_error"] < 2e-5
    translated = [row for row in rows if row["translation"] == 1e6]
    assert all(row["svd_valid_max_abs_error"] > 0.01 for row in translated)
    assert all(row["jacobi_valid_max_abs_error"] > 0.01 for row in translated)
    assert all("same rounded FP32 coordinates" in row["reference"] for row in rows)
