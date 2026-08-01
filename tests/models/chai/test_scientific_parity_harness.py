from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parent / "scripts" / "scientific_parity.py"
    spec = importlib.util.spec_from_file_location("scientific_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = _load_module()


def _load_sampler_module():
    scripts = Path(__file__).resolve().parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "benchmark_sampler_parity", scripts / "benchmark_sampler_parity.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _load_branch_module():
    scripts = Path(__file__).resolve().parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "benchmark_branch_effect", scripts / "benchmark_branch_effect.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _metadata() -> dict:
    return {
        "fixture_id": "synthetic",
        "fasta_sha256": "fasta",
        "source_weight_sha256": {"trunk": "weight"},
        "config": {"recycles": 1, "timesteps": 2, "samples": 1, "seed": 0},
        "branches": {"msa": False, "template": False, "restraint": False},
        "prepared_tensors": {
            "features.token": {"shape": [1, 3], "semantic_sha256": "tensor"}
        },
        "backend": "torch",
        "sampler_randomness": {"matching": "seed-and-distribution"},
    }


def _artifact(coords: np.ndarray, *, backend: str = "torch") -> dict:
    metadata = _metadata()
    metadata["backend"] = backend
    return {
        "metadata": metadata,
        "coords": coords[None].astype(np.float32),
        "atom_ids": np.asarray(
            [
                ["A", "1", "ALA", "CA", "0"],
                ["A", "1", "ALA", "N", "0"],
                ["A", "1", "ALA", "C", "0"],
            ]
        ),
        "pae": np.zeros((1, 1, 1), np.float32),
        "pde": np.zeros((1, 1, 1), np.float32),
        "plddt": np.asarray([[0.8]], np.float32),
        "ranking": {
            "aggregate_score": np.asarray([[0.5]], np.float32),
            "has_inter_chain_clashes": np.asarray([[False]]),
        },
    }


def test_kabsch_metrics_remove_rigid_transform() -> None:
    reference = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    mobile = reference @ rotation + np.asarray([4, -2, 3], np.float32)

    report = parity.compare_artifacts(
        _artifact(reference), _artifact(mobile, backend="jax")
    )

    assert report["coordinates"][0]["all_atom_kabsch_rmsd"] < 1e-6
    assert report["coordinates"][0]["ca_kabsch_rmsd"] < 1e-6
    assert report["coordinates"][0]["raw_coordinate_mae"] > 1.0


def test_comparison_aligns_reordered_atom_identities() -> None:
    reference = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    left = _artifact(reference)
    right = _artifact(reference[[2, 0, 1]], backend="jax")
    right["atom_ids"] = right["atom_ids"][[2, 0, 1]]

    report = parity.compare_artifacts(left, right)

    assert report["coordinates"][0]["all_atom_kabsch_rmsd"] < 1e-6


def test_contract_mismatch_fails_before_numerical_comparison() -> None:
    coords = np.zeros((3, 3), np.float32)
    left = _artifact(coords)
    right = _artifact(coords, backend="jax")
    right["metadata"]["config"]["timesteps"] = 200

    with pytest.raises(parity.ContractMismatchError, match="config"):
        parity.compare_artifacts(left, right)


def test_prepared_tensor_mismatch_is_a_release_blocker() -> None:
    coords = np.zeros((3, 3), np.float32)
    left = _artifact(coords)
    right = _artifact(coords, backend="jax")
    right["metadata"]["prepared_tensors"]["features.token"]["semantic_sha256"] = (
        "different"
    )

    with pytest.raises(parity.ContractMismatchError, match="features.token"):
        parity.compare_artifacts(left, right)


def test_semantic_hash_normalizes_backend_integer_width() -> None:
    left = np.asarray([[1, 2], [3, 4]], np.int32)
    right = left.astype(np.int64)
    assert parity.semantic_array_sha256(left) == parity.semantic_array_sha256(right)


def test_semantic_hash_preserves_float_feature_drift() -> None:
    value = np.asarray([0.12345], np.float32)
    within = value + np.asarray([8e-8], np.float32)
    assert parity.semantic_array_sha256(value) != parity.semantic_array_sha256(within)


def test_declared_distance_feature_tolerance_requires_raw_arrays() -> None:
    name = "features.InverseSquaredBlockedAtomPairDistances"
    left = _metadata()
    right = _metadata()
    left["prepared_tensors"] = {name: {"shape": [2], "semantic_sha256": "left"}}
    right["prepared_tensors"] = {name: {"shape": [2], "semantic_sha256": "right"}}
    left_array = np.asarray([0.1, 0.2], np.float32)
    right_array = left_array + np.asarray([1e-7, 0.0], np.float32)
    parity.validate_contract(left, right, {name: left_array}, {name: right_array})
    with pytest.raises(parity.ContractMismatchError):
        parity.validate_contract(left, right)


def test_template_unit_vector_rounding_tolerance_is_explicit() -> None:
    name = "features.TemplateUnitVector"
    left = _metadata()
    right = _metadata()
    left["prepared_tensors"] = {name: {"shape": [3], "semantic_sha256": "left"}}
    right["prepared_tensors"] = {name: {"shape": [3], "semantic_sha256": "right"}}
    left_array = np.asarray([0.0, 0.5, 1.0], np.float32)
    right_array = left_array + np.asarray([1e-7, 0.0, -1e-7], np.float32)

    parity.validate_contract(left, right, {name: left_array}, {name: right_array})

    with pytest.raises(parity.ContractMismatchError):
        parity.validate_contract(
            left,
            right,
            {name: left_array},
            {name: left_array + np.asarray([1e-5, 0.0, 0.0], np.float32)},
        )


def test_sampler_random_tape_is_deterministic_and_rotation_valid() -> None:
    sampler = _load_sampler_module()
    left = sampler._tape(seed=7, steps=2, samples=3, atoms=5)
    right = sampler._tape(seed=7, steps=2, samples=3, atoms=5)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])
    rotations = left["rotations"].reshape(-1, 3, 3)
    identity = np.broadcast_to(np.eye(3), rotations.shape)
    np.testing.assert_allclose(
        rotations @ np.swapaxes(rotations, -1, -2), identity, atol=2e-6
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=2e-6)


def test_sampler_static_array_drift_reports_numeric_and_discrete_values() -> None:
    sampler = _load_sampler_module()
    numeric = sampler._array_drift(
        np.asarray([1.0, 2.0], np.float32),
        np.asarray([1.0, 2.5], np.float32),
    )
    assert numeric["rmse"] == pytest.approx(np.sqrt(0.125))
    assert numeric["max_abs"] == pytest.approx(0.5)
    discrete = sampler._array_drift(
        np.asarray([1, 2], np.int32), np.asarray([1, 3], np.int64)
    )
    assert discrete["mismatch_count"] == 1


def test_sampler_dispatches_to_requested_torch_bucket() -> None:
    sampler = _load_sampler_module()

    class FakeModule:
        def __init__(self) -> None:
            self.received = None

        def forward_384(self, *values):
            self.received = values
            return ("forward-384",)

    module = FakeModule()
    result = sampler._torch_bucket_forward(module, 384, ("inputs",))

    assert result == ("forward-384",)
    assert module.received == ("inputs",)
    with pytest.raises(ValueError, match="forward_512"):
        sampler._torch_bucket_forward(module, 512, ())


def test_sampler_coordinate_metrics_name_component_and_atom_rmsd_explicitly() -> None:
    sampler = _load_sampler_module()
    reference = np.asarray([[[1, 2, 3], [4, 5, 6]]], np.float32)
    actual = reference + np.asarray([[[3, 4, 0], [0, 0, 12]]], np.float32)
    valid = np.asarray([[True, True]])

    report = sampler._coordinate_drift(reference, actual, valid)

    expected_component = np.sqrt(169.0 / 6.0)
    assert report["coordinate_rmse"] == pytest.approx(expected_component)
    assert report["coordinate_component_rmse"] == pytest.approx(expected_component)
    assert report["all_atom_raw_rmsd"] == pytest.approx(np.sqrt(169.0 / 2.0))
    assert report["max_atom_displacement"] == pytest.approx(12.0)


def test_sampler_component_and_atom_rmsd_thresholds_are_distinct() -> None:
    sampler = _load_sampler_module()
    report = {
        "coordinate_rmse": 0.8,
        "all_atom_raw_rmsd": float(np.sqrt(3.0) * 0.8),
    }

    assert (
        sampler._threshold_failures(
            report,
            max_coordinate_rmse=0.9,
            max_all_atom_raw_rmsd=None,
        )
        == []
    )
    assert sampler._threshold_failures(
        report,
        max_coordinate_rmse=None,
        max_all_atom_raw_rmsd=1.0,
    ) == ["all-atom raw RMSD threshold exceeded"]


def test_branch_effect_reports_magnitude_direction_and_error() -> None:
    branch = _load_branch_module()
    baseline = np.asarray([1.0, 2.0, 3.0], np.float32)
    torch_active = baseline + np.asarray([1.0, -2.0, 2.0], np.float32)
    jax_active = baseline + np.asarray([1.0, -2.0, 2.0], np.float32)

    report = branch._effect_metrics(torch_active, baseline, jax_active, baseline)

    assert report["torch_effect_rms"] == pytest.approx(np.sqrt(3.0))
    assert report["jax_effect_rms"] == pytest.approx(np.sqrt(3.0))
    assert report["effect_cosine"] == pytest.approx(1.0)
    assert report["effect_relative_l2_error"] == pytest.approx(0.0)


def test_real_sampler_context_exposes_msa_and_template_branches() -> None:
    sampler = _load_sampler_module()
    parameters = inspect.signature(sampler._real_protein_inputs).parameters

    assert {
        "msa_directory",
        "template_hits",
        "template_cif_directory",
        "kalign_executable",
    } <= set(parameters)


def test_production_token_embedder_uses_torch_compatibility_precision() -> None:
    from foldjax.models.chai import inference

    assert inference._TOKEN_EMBEDDER_COMPATIBILITY_BF16 is False
