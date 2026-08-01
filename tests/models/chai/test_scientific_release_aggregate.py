from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parent
        / "scripts"
        / "aggregate_scientific_release.py"
    )
    spec = importlib.util.spec_from_file_location("aggregate_scientific_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = _load_module()


def _report(seed: int) -> dict[str, object]:
    return {
        "contract": "component/noise-matched",
        "timesteps": 200,
        "samples": 1,
        "seed": seed,
        "model_size": 384,
        "context": "framework-native-prepared-and-trunk",
        "fasta": "fixture.fasta",
        "branches": {"msa": False, "template": False, "restraint": True},
        "valid_atom_count": 2004,
        "static_input_sha256": "jax-static",
        "torch_static_input_sha256": "torch-static",
        "reference_augmentation_tape_sha256": None,
        "use_esm_embeddings": False,
        "trunk_recycles": 1,
        "coordinate_component_rmse": 0.1,
        "all_atom_raw_rmsd": 0.25 + seed * 0.01,
        "max_atom_displacement": 0.8,
        "correlation": 0.999,
        "shared_torch_static": {"all_atom_raw_rmsd": 1e-4},
        "static_input_drift": {
            "token_single_initial_repr": {"valid_region": {"nrmse": 0.001}},
            "token_pair_initial_repr": {"valid_region": {"nrmse": 0.001}},
            "token_single_trunk_repr": {"valid_region": {"nrmse": 0.01}},
            "token_pair_trunk_repr": {"valid_region": {"nrmse": 0.01}},
            "atom_single_mask": {"mismatch_count": 0},
            "atom_block_pair_mask": {
                "mismatch_count": 0,
                "chemically_valid_mismatch_count": 0,
            },
            "token_single_mask": {"mismatch_count": 0},
            "atom_token_indices": {"mismatch_count": 0},
        },
    }


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "fixture": {
            "max_all_atom_raw_rmsd": 1.0,
            "max_atom_displacement": 5.0,
            "min_correlation": 0.98,
            "max_shared_static_raw_rmsd": 0.002,
            "max_single_initial_valid_nrmse": 0.01,
            "max_pair_initial_valid_nrmse": 0.005,
            "max_single_trunk_valid_nrmse": 0.03,
            "max_pair_trunk_valid_nrmse": 0.03,
        }
    }


def _provenance() -> dict[str, object]:
    return {
        "dirty": False,
        "chai_jax_commit": "chai-commit",
        "upstream_commit": "upstream-commit",
        "bundle_manifest_sha256": "bundle-hash",
        "conformer_sha256": "conformer-hash",
        "source_weight_sha256": {"diffusion_module": "weight-hash"},
    }


def test_aggregate_rejects_duplicate_fixture_seed() -> None:
    runs = [("fixture", _report(0)), ("fixture", _report(0))]

    with pytest.raises(ValueError, match="duplicate report.*fixture.*seed 0"):
        aggregate.aggregate_reports(
            runs,
            expected_fixtures=["fixture"],
            expected_seeds=[0, 1],
            thresholds=_thresholds(),
        )


def test_aggregate_rejects_missing_seed() -> None:
    with pytest.raises(ValueError, match="missing reports.*fixture:seed=1"):
        aggregate.aggregate_reports(
            [("fixture", _report(0))],
            expected_fixtures=["fixture"],
            expected_seeds=[0, 1],
            thresholds=_thresholds(),
        )


def test_aggregate_records_each_seed_and_never_uses_mean_to_rescue() -> None:
    bad = _report(1)
    bad["all_atom_raw_rmsd"] = 1.2

    result = aggregate.aggregate_reports(
        [("fixture", _report(0)), ("fixture", bad)],
        expected_fixtures=["fixture"],
        expected_seeds=[0, 1],
        thresholds=_thresholds(),
        provenance=_provenance(),
    )

    fixture = result["fixtures"]["fixture"]
    assert [run["seed"] for run in fixture["runs"]] == [0, 1]
    assert fixture["aggregate"]["max_all_atom_raw_rmsd"] == pytest.approx(1.2)
    assert fixture["pass"] is False
    assert result["status"] == "red"


def test_aggregate_requires_exact_release_configuration() -> None:
    wrong = _report(0)
    wrong["timesteps"] = 199

    with pytest.raises(ValueError, match="recycles=1, timesteps=200, samples=1"):
        aggregate.aggregate_reports(
            [("fixture", wrong)],
            expected_fixtures=["fixture"],
            expected_seeds=[0],
            thresholds=_thresholds(),
        )


def test_aggregate_rejects_fixture_identity_drift_between_seeds() -> None:
    changed = _report(1)
    changed["static_input_sha256"] = "different-static"

    with pytest.raises(ValueError, match="fixture identity differs.*seed 1"):
        aggregate.aggregate_reports(
            [("fixture", _report(0)), ("fixture", changed)],
            expected_fixtures=["fixture"],
            expected_seeds=[0, 1],
            thresholds=_thresholds(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"dirty": True}, "dirty=false"),
        ({"chai_jax_commit": ""}, "chai_jax_commit"),
        ({"source_weight_sha256": {}}, "source_weight_sha256"),
    ],
)
def test_aggregate_rejects_dirty_or_incomplete_provenance(
    mutation: dict[str, object], message: str
) -> None:
    provenance = _provenance()
    provenance.update(mutation)

    with pytest.raises(ValueError, match=message):
        aggregate.aggregate_reports(
            [("fixture", _report(0))],
            expected_fixtures=["fixture"],
            expected_seeds=[0],
            thresholds=_thresholds(),
            provenance=provenance,
        )
