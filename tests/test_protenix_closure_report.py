from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench import protenix_closure_report as report


def _input_fixture(tmp_path):
    native = {name: np.zeros(2, np.float32) for name in report.COMMON_FIELDS}
    for name in ("asym_id", "entity_id", "token_index"):
        native[name] = np.array([0, 1], np.int64)
    for name in ("residue_index", "sym_id"):
        native[name] = np.zeros(2, np.int64)
    native.update(
        {
            "ref_pos": np.zeros((3, 3), np.float32),
            "atom_to_token_idx": np.array([0, 0, 1], np.int64),
            "atom_to_tokatom_idx": np.array([0, 1, 0], np.int64),
            "has_frame": np.ones(2, np.int64),
            "is_ligand": np.array([0, 0, 1], np.int64),
            "msa": np.zeros((2, 2), np.int64),
            "msa_mask": np.ones((2, 2), bool),
            "profile": np.zeros((2, 32), np.float32),
            "restype": np.zeros((2, 32), np.float32),
            "deletion_value": np.zeros((2, 2), np.float32),
            "has_deletion": np.zeros((2, 2), np.float32),
            "v_lm": np.ones((1, 32, 128, 1), bool),
            "pad_info.mask_trunked": np.ones((1, 32, 128), bool),
            "pad_info.q_pad": np.asarray(29, np.int64),
            "pad_info.k_pad_left": np.asarray(48, np.int64),
            "pad_info.k_pad_right": np.asarray(77, np.int64),
            "template_aatype": np.zeros((4, 2), np.int64),
            "template_atom_mask": np.zeros((4, 2, 24), np.int64),
            "is_protein": np.array([1, 1, 0], np.int64),
            "is_dna": np.zeros(3, np.int64),
            "is_rna": np.zeros(3, np.int64),
            "output_atom_chain_id": np.array(["A", "A", "B"]),
            "output_atom_name": np.array(["N", "CA", "C1"]),
            "output_atom_res_name": np.array(["GLY", "GLY", "LIG"]),
            "output_atom_element": np.array(["N", "C", "C"]),
            "output_atom_res_id": np.ones(3, np.int64),
        }
    )
    for name in report.FLOAT32_INTEGER_FIELDS:
        native[name] = np.zeros(3, np.int64)
    bins = np.array([[32, 65], [65, 32]], np.uint8)
    chain = np.array([[2, 5], [5, 2]], np.uint8)
    same_entity = np.eye(2, dtype=np.uint8)
    native["relp"] = np.concatenate(
        (
            np.eye(66, dtype=np.float32)[bins],
            np.eye(66, dtype=np.float32)[bins],
            same_entity[..., None].astype(np.float32),
            np.eye(6, dtype=np.float32)[chain],
        ),
        axis=-1,
    )
    candidate = {name: native[name].copy() for name in report.COMMON_FIELDS}
    for name in report.FLOAT32_INTEGER_FIELDS | {"v_lm"}:
        candidate[name] = candidate[name].astype(np.float32)
    candidate["template_aatype"] = candidate["template_aatype"].astype(np.int32)
    candidate["template_atom_mask"] = candidate["template_atom_mask"].astype(bool)
    candidate.update(
        {
            "_foldjax_compact_relp": np.asarray(1, np.uint8),
            "_foldjax_relp_residue_bin": bins,
            "_foldjax_relp_token_bin": bins.copy(),
            "_foldjax_relp_chain_bin": chain,
            "_foldjax_relp_same_entity": same_entity,
            "output_atom_polymer_type": np.array(
                ["polypeptide(L)", "polypeptide(L)", "non-polymer"]
            ),
            "token_is_ligand": np.array([False, True]),
            "covalent_atom_indices": np.empty((0, 2), np.int64),
            "covalent_token_indices": np.empty((0, 2), np.int64),
        }
    )
    config = {
        "train_confidence_only": False,
        "sample_diffusion": {"guidance": {"enable": False}},
        "model": {
            "constraint_embedder": {
                name: {"enable": False} for name, _ in report.CONSTRAINT_FIELDS.values()
            }
        },
    }
    (tmp_path / "initial-config.json").write_text(json.dumps(config))
    _write_input(tmp_path, native)
    return native, candidate


def _write_input(root, native):
    np.savez(root / "native-input.npz", **native)
    np.savez(root / "native-derived.npz")
    np.savez(
        root / "native-identity.npz",
        **{name: native[name] for name in report.IDENTITY_FIELDS},
    )


def test_input_gate_accepts_finite_explicit_representations_without_mutating(tmp_path):
    _, candidate = _input_fixture(tmp_path)
    before = {name: value.tobytes() for name, value in candidate.items()}
    result = report.check_inputs(tmp_path, candidate)
    assert result["passed"] is True
    assert len(result["leaves"]) == 40
    assert result["leaves"]["ref_charge"]["representation"] != "identical"
    assert before == {name: value.tobytes() for name, value in candidate.items()}


@pytest.mark.parametrize(
    "field", ["inference_seed", "prot_mystery_num_alignments", "new_model_feature"]
)
def test_native_unknown_fields_are_not_allowed_by_metadata_globs(tmp_path, field):
    native, candidate = _input_fixture(tmp_path)
    native[field] = np.asarray(101, np.int64)
    _write_input(tmp_path, native)
    with pytest.raises(ValueError, match="unmapped native input schema"):
        report.check_inputs(tmp_path, candidate)


def test_common_value_change_fails_even_when_small(tmp_path):
    _, candidate = _input_fixture(tmp_path)
    candidate["profile"][0, 0] = 1e-8
    result = report.check_inputs(tmp_path, candidate)
    assert result["passed"] is False
    assert "shape/value:profile" in result["failures"]


def test_unlisted_integer_dtype_change_is_not_a_general_permission(tmp_path):
    _, candidate = _input_fixture(tmp_path)
    candidate["msa"] = candidate["msa"].astype(np.int32)
    with pytest.raises(ValueError, match="unmapped dtype: msa"):
        report.check_inputs(tmp_path, candidate)


@pytest.mark.parametrize(
    "defect", ["relative", "padding", "polymer", "ligand", "covalent", "nonfinite"]
)
def test_derived_and_identity_mapping_defects_fail(tmp_path, defect):
    native, candidate = _input_fixture(tmp_path)
    if defect == "relative":
        candidate["_foldjax_relp_chain_bin"][0, 1] = 0
    elif defect == "padding":
        native["pad_info.k_pad_right"] = np.asarray(76, np.int64)
        _write_input(tmp_path, native)
    elif defect == "polymer":
        candidate["output_atom_polymer_type"][0] = "non-polymer"
    elif defect == "ligand":
        candidate["token_is_ligand"][0] = True
    elif defect == "covalent":
        candidate["covalent_atom_indices"] = np.array([[0, 2]], np.int64)
    else:
        candidate["profile"][0, 0] = np.nan
    if defect in ("relative", "nonfinite"):
        with pytest.raises(ValueError):
            report.check_inputs(tmp_path, candidate)
    else:
        assert report.check_inputs(tmp_path, candidate)["passed"] is False


def test_unicode_width_only_is_accepted_but_atom_renaming_is_not(tmp_path):
    _, candidate = _input_fixture(tmp_path)
    candidate["output_atom_name"] = candidate["output_atom_name"].astype("U12")
    assert report.check_inputs(tmp_path, candidate)["passed"]
    candidate["output_atom_name"][0] = "renamed"
    assert report.check_inputs(tmp_path, candidate)["passed"] is False


def test_active_native_constraint_cannot_be_exempted_as_metadata(tmp_path):
    native, candidate = _input_fixture(tmp_path)
    native["constraint_feature.pocket"] = np.ones((2, 2, 1), np.float32)
    _write_input(tmp_path, native)
    assert report.check_inputs(tmp_path, candidate)["passed"] is False


def _output_fixture():
    coords = np.arange(45, dtype=np.float32).reshape(5, 3, 3)
    raw = {
        "coordinate": coords,
        "contact_probs": np.zeros((2, 2), np.float32),
        "plddt": np.zeros((5, 3, 50), np.float32),
        "resolved": np.zeros((5, 3, 2), np.float32),
        "pae": np.zeros((5, 2, 2, 64), np.float32),
        "pde": np.zeros((5, 2, 2, 64), np.float32),
    }
    for sample in range(5):
        for name in report.SUMMARY:
            shape = (
                (2, 2)
                if name.startswith("chain_pair_")
                else (2,)
                if name.startswith("chain_")
                else ()
            )
            value = np.zeros(shape, bool if name == "has_clash" else np.float32)
            if name == "num_recycles":
                value = np.asarray(10, np.int64)
            raw[f"summary_confidence.{sample}.{name}"] = value
        for name, value in {
            "atom_coordinate": coords[sample],
            "atom_plddt": np.zeros(3, np.float32),
            "token_pair_pae": np.zeros((2, 2), np.float32),
            "token_pair_pde": np.zeros((2, 2), np.float32),
            "contact_probs": raw["contact_probs"],
            "token_has_frame": np.ones(2, np.int64),
            "token_asym_id": np.array([0, 1], np.int64),
            "atom_to_token_idx": np.array([0, 0, 1], np.int64),
            "atom_is_polymer": np.array([1, 1, 0], np.int64),
        }.items():
            raw[f"full_data.{sample}.{name}"] = value
    distogram = {
        "input": np.zeros((2, 2, 128), np.float32),
        "logits": np.zeros((2, 2, 64), np.float32),
    }
    fj = {name: raw[name].copy() for name in report.RAW | {"coordinate"}}
    fj["distogram_logits"] = distogram["logits"]
    for name in report.SUMMARY:
        fj[report.SUMMARY_ALIASES.get(name, name)] = (
            np.asarray(10, np.int32)
            if name == "num_recycles"
            else np.stack([raw[f"summary_confidence.{i}.{name}"] for i in range(5)])
        )
    fj["ranking_score"] = fj["summary_ranking_score"].copy()
    for name in ("atom_plddt", "token_pair_pae", "token_pair_pde"):
        fj[name] = np.stack([raw[f"full_data.{i}.{name}"] for i in range(5)])
    features = {
        "has_frame": np.ones(2, np.int64),
        "asym_id": np.array([0, 1], np.int64),
        "atom_to_token_idx": np.array([0, 0, 1], np.int64),
        "is_ligand": np.array([0, 0, 1], np.int64),
    }
    return raw, distogram, fj, features


def test_all_native_outputs_are_canonicalized_with_sample_indices_intact():
    raw, distogram, fj, features = _output_fixture()
    left_coords, left = report.canonical_native(raw, distogram)
    right_coords, right, extras = report.canonical_foldjax(fj, features)
    assert len(raw) == 141 and len(left) == len(right) == 31
    assert not extras
    np.testing.assert_array_equal(left_coords, right_coords)
    for name in left:
        assert left[name].dtype == right[name].dtype
        np.testing.assert_array_equal(left[name], right[name])
    assert "summary.chain_pair_pae_mean" in left
    assert "summary.chain_pair_pae_min" in left
    assert "raw.distogram_logits" in left


def test_explicit_foldjax_only_diagnostics_are_retained_not_claimed_native():
    _, _, fj, features = _output_fixture()
    fj["has_vdw_clash"] = np.zeros(5, bool)
    fj["summary_ranking_score_vdw_penalized"] = np.zeros(5, np.float32)
    _, canonical, extras = report.canonical_foldjax(fj, features)
    assert set(extras) == report.FOLDJAX_ONLY_DIAGNOSTICS
    assert all(name not in canonical for name in extras)
    fj["unknown_metric"] = np.zeros(5, np.float32)
    with pytest.raises(ValueError, match="unmapped FoldJAX"):
        report.canonical_foldjax(fj, features)


@pytest.mark.parametrize(
    "defect",
    ["missing", "extra", "sample_count", "coordinate_duplicate", "rank_duplicate"],
)
def test_output_schema_and_duplicate_contracts_fail_closed(defect):
    raw, distogram, fj, features = _output_fixture()
    if defect == "missing":
        del raw["summary_confidence.4.chain_pair_pae_min"]
    elif defect == "extra":
        raw["new_native_head"] = np.zeros(5, np.float32)
    elif defect == "sample_count":
        raw["plddt"] = raw["plddt"][:4]
    elif defect == "coordinate_duplicate":
        raw["full_data.0.atom_coordinate"] = np.zeros((3, 3), np.float32)
    else:
        fj["ranking_score"] += 1
        with pytest.raises(ValueError, match="ranking_score"):
            report.canonical_foldjax(fj, features)
        return
    with pytest.raises(ValueError):
        report.canonical_native(raw, distogram)


def test_native_record_refuses_incomplete_capture_without_reading_arrays(tmp_path):
    (tmp_path / "capture-complete.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="incomplete/unsupported native capture"):
        report.native_record(tmp_path)


def test_calibration_cli_rejects_existing_output_directory(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["report", "calibrate", str(Path("unused")), "--out", str(tmp_path)],
    )
    with pytest.raises(FileExistsError):
        report.main()
