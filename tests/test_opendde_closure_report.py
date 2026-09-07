import numpy as np
import pytest

from bench.af3_closure import compare_confidence
from bench.opendde_closure_report import (
    ALIASES,
    FULL,
    RAW,
    SHAPE_FLAG,
    SUMMARY,
    canonical_foldjax,
    canonical_native,
)


def fixture_outputs():
    features = {
        "has_frame": np.array([1, 0], np.int64),
        "asym_id": np.array([0, 1], np.int64),
        "atom_to_token_idx": np.array([0, 0, 1], np.int64),
        "is_ligand": np.array([0, 0, 1], np.int64),
        "is_protein": np.array([1, 1, 0], np.int64),
        "is_dna": np.zeros(3, np.int64),
        "is_rna": np.zeros(3, np.int64),
    }
    raw = {key: np.ones((5,), np.float32) for key in RAW}
    raw["coordinate"] = np.arange(45, dtype=np.float32).reshape(5, 3, 3)
    raw["contact_probs"] = np.ones((2, 2), np.float32)
    raw["plddt"] = np.ones((5, 3, 50), np.float32)
    raw["resolved"] = np.ones((5, 3, 2), np.float32)
    raw["pae"] = np.ones((5, 2, 2, 64), np.float32)
    raw["pde"] = np.ones((5, 2, 2, 64), np.float32)
    raw["shape_comp_token_pred"] = np.ones((5, 2), np.float32)
    raw[SHAPE_FLAG] = np.array([0], np.int64)
    raw["shape_comp_token_mask"] = np.ones((5, 2), bool)
    scored = {key: raw[key].copy() for key in RAW | {"coordinate"}}
    scored[SHAPE_FLAG] = np.array(False)
    scored["distogram_logits"] = np.ones((2, 2, 96), np.float32)
    for key in SUMMARY:
        shape = (
            (2, 2)
            if key.startswith("chain_pair_")
            else (2,)
            if key.startswith("chain_")
            else ()
        )
        value = (
            np.array(10, np.int64)
            if key == "num_recycles"
            else np.array(False)
            if key == "has_clash"
            else np.full(shape, 0.5, np.float32)
        )
        for index in range(5):
            raw[f"summary_confidence.{index}.{key}"] = value.copy()
        scored[ALIASES.get(key, key)] = (
            np.array(10, np.int32) if key == "num_recycles" else np.stack([value] * 5)
        )
    scored["ranking_score"] = scored["summary_ranking_score"].copy()
    full = {
        "atom_plddt": np.ones((3,), np.float32),
        "token_pair_pae": np.ones((2, 2), np.float32),
        "token_pair_pde": np.ones((2, 2), np.float32),
        "contact_probs": raw["contact_probs"],
        "token_has_frame": features["has_frame"],
        "token_asym_id": features["asym_id"],
        "atom_to_token_idx": features["atom_to_token_idx"],
        "atom_is_polymer": 1 - features["is_ligand"],
    }
    assert set(full) | {"atom_coordinate"} == FULL
    for index in range(5):
        for key, value in full.items():
            raw[f"full_data.{index}.{key}"] = value.copy()
        raw[f"full_data.{index}.atom_coordinate"] = raw["coordinate"][index].copy()
    for key in ("atom_plddt", "token_pair_pae", "token_pair_pde"):
        scored[key] = np.stack([full[key]] * 5)
    return raw, scored, features


def test_all_native_returned_confidence_leaves_have_explicit_mapping():
    raw, scored, features = fixture_outputs()
    _, native = canonical_native(raw)
    _, foldjax = canonical_foldjax(scored, features)
    report = compare_confidence(native, foldjax)
    assert report["passed"]
    assert all(leaf["bitwise_equal"] for leaf in report["leaves"].values())


@pytest.mark.parametrize("arm", ["native", "foldjax"])
@pytest.mark.parametrize("damage", ["extra", "missing", "flag", "cycles", "duplicate"])
def test_schema_and_representation_changes_fail_closed(arm, damage):
    raw, scored, features = fixture_outputs()
    values = raw if arm == "native" else scored
    if damage == "extra":
        values["new_output"] = np.array(1)
    elif damage == "missing":
        del values["pae"]
    elif damage == "flag":
        values[SHAPE_FLAG] = np.array([2], np.int64)
    elif damage == "cycles":
        values[
            "summary_confidence.0.num_recycles" if arm == "native" else "num_recycles"
        ] = np.array(9, np.int32)
    else:
        values[
            "full_data.0.atom_coordinate" if arm == "native" else "ranking_score"
        ] += 1
    with pytest.raises(ValueError):
        canonical_native(raw) if arm == "native" else canonical_foldjax(
            scored, features
        )


def test_full_pae_failure_is_not_hidden_by_passing_summary():
    raw, scored, features = fixture_outputs()
    scored["token_pair_pae"][3, 0, 1] += 0.02
    _, native = canonical_native(raw)
    _, foldjax = canonical_foldjax(scored, features)
    report = compare_confidence(native, foldjax)
    assert not report["passed"]
    assert not report["leaves"]["full.token_pair_pae"]["passed"]
    assert report["leaves"]["summary.plddt"]["passed"]


@pytest.mark.parametrize("arm", ["native", "foldjax"])
def test_identically_collapsed_raw_head_axes_are_rejected(arm):
    raw, scored, features = fixture_outputs()
    for key in ("plddt", "pae", "pde", "resolved"):
        raw[key] = scored[key] = np.zeros(5, np.float32)
    with pytest.raises(ValueError, match="shape/dtype"):
        canonical_native(raw) if arm == "native" else canonical_foldjax(
            scored, features
        )


def test_report_uses_actual_foldjax_polymer_consumer_masks():
    _, scored, features = fixture_outputs()
    features["is_protein"][:] = 0
    with pytest.raises(ValueError, match="polymer masks disagree"):
        canonical_foldjax(scored, features)
