import json

import numpy as np
import pytest

from bench import esmfold2_tape as tape
from bench import esmfold2_tape_report as report


def _features():
    return dict(
        atom_attention_mask=np.ones((1, 6), bool),
        token_attention_mask=np.ones((1, 2), bool),
        atom_to_token=np.array([[0, 0, 0, 1, 1, 1]]),
        entity_id=np.array([[0, 0]]),
        mol_type=np.array([[0, 0]]),
        asym_id=np.array([[0, 1]]),
    )


def test_entity_instances_not_merged_and_no_entity_refit():
    left = np.random.default_rng(1).normal(size=(5, 6, 3))
    right = left.copy()
    right[:, 3:] += 2
    result = report.compare_coordinates(left, right, _features())
    assert len(result["entity_rmsd"]) == 2
    assert min(result["entity_max_rmsd"].values()) > 0.1


@pytest.mark.parametrize("bad", ["mask", "mapping", "identity"])
def test_rejects_ambiguous_atom_identity(bad):
    features = _features()
    if bad == "mask":
        features["atom_attention_mask"] = np.full((1, 6), 2)
    if bad == "mapping":
        features["atom_to_token"][0, 0] = 2
    if bad == "identity":
        features["asym_id"] = features["asym_id"].astype(float)
    with pytest.raises(ValueError):
        report.entity_inputs(features)


def test_bound_partial_raw_heads_and_tamper(tmp_path):
    native, candidate = tmp_path / "native", tmp_path / "candidate"
    native.mkdir()
    candidate.mkdir()
    features = tmp_path / "input.npz"
    tape._save_npz(features, _features())
    tape._save_npz(native / "features.npz", _features())
    tape._save_npz(native / "tape.npz", {"dummy": np.ones(2)})
    output = {
        name: np.ones(5, np.float32)
        for name in ("plddt", "complex_plddt", "ptm", "iptm")
    }
    output["sample_atom_coords"] = np.random.default_rng(1).normal(size=(5, 6, 3))
    output["pae_logits"] = np.ones((5, 2, 2, 3), np.float32)
    a = dict(
        binding=dict(
            source={"native": "hash"}, runner="hash", checkpoint={"weights": "hash"}
        ),
        input_sha256=tape._sha256(features),
        tape_sha256=tape._sha256(native / "tape.npz"),
        output_schema=tape._save_outputs(native, "upstream", output, native=False),
    )
    (native / "metadata.json").write_text(json.dumps(a))
    output.pop("pae_logits")
    b = {
        **a,
        "binding": {
            **a["binding"],
            "reference_manifest_sha256": tape._sha256(native / "metadata.json"),
        },
        "output_schema": tape._save_outputs(candidate, "jax", output, native=False),
    }
    (candidate / "metadata.json").write_text(json.dumps(b))
    result = report.build_report(native, candidate, features)
    assert result["missing_candidate_raw_heads"] == ["pae_logits"]
    assert not result["raw_head_retention_complete"]
    assert result["full_model_admission"] is None
    (native / "tape.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tape identity"):
        report.build_report(native, candidate, features)


@pytest.mark.parametrize("interchange", [False, True])
def test_lm_arm_statistics_and_exact_interchange_binding(tmp_path, interchange):
    native, candidate = tmp_path / "native", tmp_path / "candidate"
    native.mkdir()
    candidate.mkdir()
    values = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    a = {
        "lm_arm": "native_independent_lm",
        "lm_schema": tape._save_lm(native, "upstream", values, native=False),
    }
    b = {
        "lm_arm": "native_lm_interchange_downstream_core_only"
        if interchange
        else "jax_independent_lm",
        "lm_schema": tape._save_lm(
            candidate, "jax", values if interchange else values + 1, native=False
        ),
    }
    result = report.compare_lm(native, candidate, a, b)
    assert result["downstream_only"] is interchange
    assert result["independent_lm_comparison"] is not interchange
    assert result["statistics"]["leaves"]["lm_hidden_states"]["max_abs"] == (
        0 if interchange else 1
    )
    b["lm_arm"] = "native_lm_interchange_downstream_core_only"
    b["lm_schema"] = tape._save_lm(candidate, "jax", values + 2, native=False)
    with pytest.raises(ValueError, match="does not match"):
        report.compare_lm(native, candidate, a, b)


def test_lm_hash_and_dtype_claims_fail_closed(tmp_path):
    values = np.full((1, 2, 3, 4), 1.001, np.float32)
    document = {"lm_schema": tape._save_lm(tmp_path, "upstream", values, native=False)}
    document["lm_schema"]["original_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="lossless"):
        report._lm_artifact(tmp_path, document)
    (tmp_path / "upstream_lm.npz").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="identity"):
        report._lm_artifact(tmp_path, document)


def test_lm_arm_missing_artifact_is_not_legacy_success(tmp_path):
    with pytest.raises(ValueError, match="lacks"):
        report.compare_lm(tmp_path, tmp_path, {"lm_arm": "native_independent_lm"}, {})
