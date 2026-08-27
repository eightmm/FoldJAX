from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.opendde.data.featurize_json import (
    _resolve_seed,
    featurize_opendde_json,
    load_jobs,
)
from foldjax.models.protenix.data.featurize_json import featurize_protein_json


def _use_official_ccd_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rdkit")
    asset_root = Path(__file__).resolve().parents[4] / "protenix" / "common"
    components = asset_root / "components.cif"
    rdkit_cache = asset_root / "components.cif.rdkit_mol.pkl"
    if not components.is_file() or not rdkit_cache.is_file():
        pytest.skip("official components.cif/RDKit CCD assets are unavailable")
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setenv("PROTENIX_CCD_RDKIT_MOL_FILE", str(rdkit_cache))


def _atom_names(features: dict[str, object], atom_indices: np.ndarray) -> list[str]:
    names = np.asarray(features["output_atom_name"])
    return [str(name) for name in names[atom_indices]]


def test_empty_model_seeds_use_release_default_in_direct_api() -> None:
    assert _resolve_seed({"modelSeeds": []}, None) == 101


def test_one_resolved_seed_drives_chemistry_and_reference_augmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foldjax.models.opendde.data import featurize_json as implementation

    observed: dict[str, int] = {}
    original_featurize = implementation.featurize_protein_json
    original_prepare_reference = implementation._prepare_reference_features

    def featurize_spy(*args, **kwargs):
        observed["chemistry"] = kwargs["seed"]
        return original_featurize(*args, **kwargs)

    def reference_spy(*args, **kwargs):
        observed["reference"] = kwargs["seed"]
        return original_prepare_reference(*args, **kwargs)

    monkeypatch.setattr(implementation, "featurize_protein_json", featurize_spy)
    monkeypatch.setattr(implementation, "_prepare_reference_features", reference_spy)

    featurize_opendde_json(
        {"sequences": [{"proteinChain": {"sequence": "A"}}]},
        n_queries=2,
        n_keys=4,
        seed=37,
    )

    assert observed == {"chemistry": 37, "reference": 37}


def _atoms_for_token(
    features: dict[str, object],
    token_idx: int,
    *,
    structural: bool,
) -> np.ndarray:
    key = "atom_to_structural_token_idx" if structural else "atom_to_token_idx"
    return np.flatnonzero(np.asarray(features[key]) == token_idx)


def test_official_tiny_protein_structural_token_invariants() -> None:
    features = featurize_opendde_json(
        {
            "name": "tiny",
            "modelSeeds": [101],
            "sequences": [{"proteinChain": {"sequence": "ACDEFGHIK", "count": 1}}],
        },
        n_queries=2,
        n_keys=4,
    )

    expected_parent = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8]
    expected_role = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2]
    expected_twin = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, -1, 12, 11, 14, 13, 16, 15]

    np.testing.assert_array_equal(features["parent_residue_idx"], expected_parent)
    np.testing.assert_array_equal(features["subtoken_role"], expected_role)
    np.testing.assert_array_equal(features["subtoken_role_id"], expected_role)
    np.testing.assert_array_equal(features["twin_token_idx"], expected_twin)
    np.testing.assert_array_equal(features["residue_token_group_id"], expected_parent)
    np.testing.assert_array_equal(features["structural_token_index"], np.arange(17))

    atom_to_structural = np.asarray(features["atom_to_structural_token_idx"])
    assert np.all((0 <= atom_to_structural) & (atom_to_structural < 17))
    structural_rep = np.asarray(features["structural_pae_rep_atom_mask"])
    np.testing.assert_array_equal(
        np.bincount(atom_to_structural, weights=structural_rep, minlength=17),
        np.ones(17),
    )
    np.testing.assert_array_equal(
        features["structural_distogram_rep_atom_mask"], structural_rep
    )

    backbone_names = {"N", "CA", "C", "O", "OXT"}
    for structural_idx, role in enumerate(expected_role):
        names = set(
            _atom_names(
                features,
                _atoms_for_token(features, structural_idx, structural=True),
            )
        )
        if role == 1:
            assert names <= backbone_names
            assert {"N", "CA", "C"} <= names
        else:
            assert names.isdisjoint(backbone_names)
            assert "CB" in names

    atom_names = np.asarray(features["output_atom_name"])
    residue_map = np.asarray(features["atom_to_token_idx"])
    residue_rep = np.asarray(features["pae_rep_atom_mask"])
    assert np.all(atom_names[residue_rep.astype(bool)] == "CA")
    np.testing.assert_array_equal(
        np.bincount(residue_map, weights=residue_rep, minlength=9), np.ones(9)
    )
    np.testing.assert_array_equal(features["plddt_m_rep_atom_mask"], residue_rep)
    np.testing.assert_array_equal(features["is_protein"], np.ones(len(atom_names)))
    np.testing.assert_array_equal(features["is_ligand"], np.zeros(len(atom_names)))
    np.testing.assert_array_equal(features["is_dna"], np.zeros(len(atom_names)))
    np.testing.assert_array_equal(features["is_rna"], np.zeros(len(atom_names)))

    residue_frames = np.asarray(features["frame_atom_index"])
    structural_frames = np.asarray(features["structural_frame_atom_index"])
    np.testing.assert_array_equal(features["has_frame"], np.ones(9))
    np.testing.assert_array_equal(features["structural_has_frame"], np.ones(17))
    for residue_idx in range(9):
        assert _atom_names(features, residue_frames[residue_idx]) == ["N", "CA", "C"]
    for structural_idx, parent_idx in enumerate(expected_parent):
        np.testing.assert_array_equal(
            structural_frames[structural_idx], residue_frames[parent_idx]
        )

    np.testing.assert_array_equal(
        features["prev_parent_residue_idx"],
        [-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7],
    )
    np.testing.assert_array_equal(
        features["next_parent_residue_idx"],
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, -1, -1],
    )
    np.testing.assert_array_equal(features["structural_polymer_type"], np.ones(17))
    np.testing.assert_array_equal(
        features["structural_seq_pos"], np.asarray(expected_parent) + 1
    )


def test_official_tiny_seeded_reference_and_dummy_msa_match_upstream() -> None:
    features = featurize_opendde_json(
        {
            "name": "tiny",
            "modelSeeds": [101],
            "sequences": [{"proteinChain": {"sequence": "ACDEFGHIK", "count": 1}}],
        },
        seed=101,
        n_queries=2,
        n_keys=4,
    )

    np.testing.assert_array_equal(
        features["msa"],
        [
            [0, 4, 3, 6, 13, 7, 8, 9, 11],
            [0, 4, 3, 6, 13, 7, 8, 9, 11],
        ],
    )
    np.testing.assert_array_equal(features["has_deletion"], np.zeros((2, 9)))
    np.testing.assert_array_equal(features["deletion_value"], np.zeros((2, 9)))
    np.testing.assert_allclose(
        features["ref_pos"][:5],
        np.asarray(
            [
                [0.15713711, 1.5502492, -0.6407273],
                [-0.57528424, 0.38882664, -1.1439961],
                [0.06396435, -0.8278312, -0.5338807],
                [1.0418553, -0.71059155, 0.24777524],
                [-0.5236864, 0.3060227, -2.6444287],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_glycine_remains_one_backbone_structural_token() -> None:
    features = featurize_opendde_json(
        {"sequences": [{"proteinChain": {"sequence": "G"}}]},
        n_queries=2,
        n_keys=4,
    )

    np.testing.assert_array_equal(features["parent_residue_idx"], [0])
    np.testing.assert_array_equal(features["subtoken_role_id"], [1])
    np.testing.assert_array_equal(features["twin_token_idx"], [-1])
    np.testing.assert_array_equal(features["atom_to_structural_token_idx"], 0)
    assert set(np.asarray(features["output_atom_name"])) == {
        "N",
        "CA",
        "C",
        "O",
        "OXT",
    }


@pytest.mark.parametrize(
    ("entity", "roles", "polymer_type", "is_key"),
    [
        ({"dnaSequence": {"sequence": "GA"}}, [3, 4, 3, 4], 2, "is_dna"),
        ({"rnaSequence": {"sequence": "CU"}}, [5, 6, 5, 6], 3, "is_rna"),
    ],
)
def test_standard_nucleic_acids_split_backbone_and_base(
    entity: dict[str, object],
    roles: list[int],
    polymer_type: int,
    is_key: str,
) -> None:
    features = featurize_opendde_json(
        {"sequences": [entity]},
        n_queries=2,
        n_keys=4,
    )

    np.testing.assert_array_equal(features["parent_residue_idx"], [0, 0, 1, 1])
    np.testing.assert_array_equal(features["subtoken_role_id"], roles)
    np.testing.assert_array_equal(features["twin_token_idx"], [1, 0, 3, 2])
    np.testing.assert_array_equal(features["structural_polymer_type"], polymer_type)
    np.testing.assert_array_equal(features[is_key], 1)
    np.testing.assert_array_equal(features["prev_parent_residue_idx"], [-1, -1, 0, 0])
    np.testing.assert_array_equal(features["next_parent_residue_idx"], [1, 1, -1, -1])

    names = np.asarray(features["output_atom_name"])
    residue_rep = np.asarray(features["pae_rep_atom_mask"]).astype(bool)
    assert np.all(names[residue_rep] == "C1'")
    frames = np.asarray(features["structural_frame_atom_index"])
    for pair_start in (0, 2):
        np.testing.assert_array_equal(frames[pair_start], frames[pair_start + 1])
        assert _atom_names(features, frames[pair_start]) == ["C1'", "C3'", "C4'"]


def test_ligand_atoms_remain_atom_level_structural_tokens() -> None:
    features = featurize_opendde_json(
        {"sequences": [{"ligand": {"ligand": "CCD_ATP"}}]},
        n_queries=2,
        n_keys=4,
    )
    n_atom = len(np.asarray(features["ref_pos"]))

    np.testing.assert_array_equal(features["parent_residue_idx"], np.arange(n_atom))
    np.testing.assert_array_equal(features["subtoken_role_id"], np.zeros(n_atom))
    np.testing.assert_array_equal(features["twin_token_idx"], -np.ones(n_atom))
    np.testing.assert_array_equal(
        features["atom_to_structural_token_idx"], np.arange(n_atom)
    )
    np.testing.assert_array_equal(features["atom_to_structural_tokatom_idx"], 0)
    np.testing.assert_array_equal(features["structural_pae_rep_atom_mask"], 1)
    np.testing.assert_array_equal(features["is_ligand"], 1)
    np.testing.assert_array_equal(features["structural_is_polymer"], 0)
    np.testing.assert_array_equal(features["structural_polymer_type"], 0)
    np.testing.assert_array_equal(features["prev_parent_residue_idx"], -1)
    np.testing.assert_array_equal(features["next_parent_residue_idx"], -1)
    np.testing.assert_array_equal(
        np.asarray(features["structural_frame_atom_index"])[:, 1], np.arange(n_atom)
    )


def test_mse_uses_standard_met_structural_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MA",
                    "modifications": [{"ptmType": "CCD_MSE", "ptmPosition": 1}],
                }
            }
        ]
    }
    base_features = featurize_protein_json(job, n_queries=2, n_keys=4)
    features = featurize_opendde_json(
        job,
        n_queries=2,
        n_keys=4,
        augment_reference=False,
    )

    np.testing.assert_array_equal(features["parent_residue_idx"], [0, 0, 1, 1])
    np.testing.assert_array_equal(features["subtoken_role_id"], [1, 2, 1, 2])
    np.testing.assert_array_equal(features["twin_token_idx"], [1, 0, 3, 2])
    np.testing.assert_array_equal(features["modified_res_mask"], 0)
    np.testing.assert_array_equal(features["prev_parent_residue_idx"], [-1, -1, 0, 0])
    np.testing.assert_array_equal(features["next_parent_residue_idx"], [1, 1, -1, -1])
    np.testing.assert_allclose(
        features["ref_pos"][:8], base_features["ref_pos"][:8], atol=1e-6
    )


def test_nonconsecutive_same_chain_backbone_bond_controls_parent_graph() -> None:
    features = featurize_opendde_json(
        {
            "sequences": [{"proteinChain": {"sequence": "AAA"}}],
            "covalent_bonds": [
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "C",
                    "entity2": 1,
                    "copy2": 1,
                    "position2": 3,
                    "atom2": "N",
                }
            ],
        },
        n_queries=2,
        n_keys=4,
        augment_reference=False,
    )

    # Each alanine produces a backbone and side-chain structural token. The
    # explicit parent-0 -> parent-2 backbone edge takes precedence over the
    # adjacent-residue fallback, matching upstream's real bond-graph logic.
    np.testing.assert_array_equal(features["parent_residue_idx"], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(
        features["prev_parent_residue_idx"], [-1, -1, -1, -1, 0, 0]
    )
    np.testing.assert_array_equal(
        features["next_parent_residue_idx"], [2, 2, -1, -1, -1, -1]
    )


def test_sep_uses_atom_structural_tokens_and_breaks_standard_polymer_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    features = featurize_opendde_json(
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "SA",
                        "modifications": [{"ptmType": "CCD_SEP", "ptmPosition": 1}],
                    }
                }
            ]
        },
        n_queries=2,
        n_keys=4,
        augment_reference=False,
    )

    np.testing.assert_array_equal(
        features["parent_residue_idx"], list(range(10)) + [10, 10]
    )
    np.testing.assert_array_equal(features["subtoken_role_id"], [0] * 10 + [1, 2])
    np.testing.assert_array_equal(features["twin_token_idx"], [-1] * 10 + [11, 10])
    np.testing.assert_array_equal(features["modified_res_mask"], [1] * 10 + [0] * 6)
    np.testing.assert_array_equal(
        features["pae_rep_atom_mask"], [1] * 10 + [0, 1, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        features["plddt_m_rep_atom_mask"], [0] * 11 + [1] + [0] * 4
    )
    np.testing.assert_array_equal(features["prev_parent_residue_idx"], -1)
    np.testing.assert_array_equal(features["next_parent_residue_idx"], -1)
    np.testing.assert_array_equal(features["structural_polymer_type"], 1)
    np.testing.assert_array_equal(features["structural_seq_pos"], [1] * 10 + [2, 2])
    np.testing.assert_array_equal(features["is_protein"], 1)
    np.testing.assert_array_equal(features["is_ligand"], 0)
    np.testing.assert_array_equal(
        features["structural_frame_atom_index"],
        [
            [1, 0, 4],
            [0, 1, 4],
            [3, 2, 1],
            [2, 3, 6],
            [5, 4, 1],
            [4, 5, 1],
            [7, 6, 8],
            [6, 7, 8],
            [6, 8, 7],
            [6, 9, 7],
            [10, 11, 12],
            [10, 11, 12],
        ],
    )
    np.testing.assert_array_equal(features["structural_has_frame"], 1)


def test_modified_dna_and_rna_keep_polymer_identity_with_atom_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    dna = featurize_opendde_json(
        {
            "sequences": [
                {
                    "dnaSequence": {
                        "sequence": "AG",
                        "modifications": [
                            {"modificationType": "CCD_6MA", "basePosition": 1}
                        ],
                    }
                }
            ]
        },
        n_queries=2,
        n_keys=4,
        augment_reference=False,
    )
    np.testing.assert_array_equal(dna["parent_residue_idx"], list(range(23)) + [23, 23])
    np.testing.assert_array_equal(dna["subtoken_role_id"], [0] * 23 + [3, 4])
    np.testing.assert_array_equal(dna["twin_token_idx"], [-1] * 23 + [24, 23])
    np.testing.assert_array_equal(dna["modified_res_mask"], [1] * 23 + [0] * 22)
    np.testing.assert_array_equal(dna["structural_polymer_type"], 2)
    np.testing.assert_array_equal(dna["prev_parent_residue_idx"], -1)
    np.testing.assert_array_equal(dna["next_parent_residue_idx"], -1)

    rna = featurize_opendde_json(
        {
            "sequences": [
                {
                    "rnaSequence": {
                        "sequence": "AC",
                        "modifications": [
                            {"modificationType": "CCD_5MC", "basePosition": 2}
                        ],
                    }
                }
            ]
        },
        n_queries=2,
        n_keys=4,
        augment_reference=False,
    )
    np.testing.assert_array_equal(
        rna["parent_residue_idx"], [0, 0] + list(range(1, 22))
    )
    np.testing.assert_array_equal(rna["subtoken_role_id"], [5, 6] + [0] * 21)
    np.testing.assert_array_equal(rna["twin_token_idx"], [1, 0] + [-1] * 21)
    np.testing.assert_array_equal(rna["modified_res_mask"], [0] * 23 + [1] * 21)
    np.testing.assert_array_equal(rna["structural_polymer_type"], 3)
    np.testing.assert_array_equal(rna["prev_parent_residue_idx"], -1)
    np.testing.assert_array_equal(rna["next_parent_residue_idx"], -1)


def test_featurizer_does_not_import_torch_or_upstream_opendde() -> None:
    script = r"""
import builtins
import json

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    root = name.split(".", 1)[0]
    if root in {"torch", "opendde"}:
        raise RuntimeError(f"forbidden import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from foldjax.models.opendde.data.featurize_json import featurize_opendde_json

features = featurize_opendde_json(
    {"sequences": [{"proteinChain": {"sequence": "AG"}}]},
    n_queries=2,
    n_keys=4,
)
print(json.dumps({"n_structural": len(features["parent_residue_idx"])}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"n_structural": 3}


def test_load_jobs_preserves_all_top_level_jobs(tmp_path) -> None:
    path = tmp_path / "jobs.json"
    expected = [
        {"name": "one", "sequences": [{"proteinChain": {"sequence": "A"}}]},
        {"name": "two", "sequences": [{"proteinChain": {"sequence": "G"}}]},
    ]
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert load_jobs(path) == expected


def test_assembly_id_is_accepted_as_benign_job_metadata() -> None:
    features = featurize_opendde_json(
        {
            "name": "assembly-metadata",
            "assembly_id": "1",
            "sequences": [{"proteinChain": {"sequence": "A"}}],
        },
        n_queries=2,
        n_keys=4,
    )

    assert np.asarray(features["restype"]).shape == (1, 32)


def test_official_repo_root_relative_msa_path_falls_back_from_json_parent(
    tmp_path,
) -> None:
    repo_root = tmp_path / "OpenDDE"
    json_parent = repo_root / "examples" / "nested"
    msa_path = repo_root / "examples" / "fixture" / "unpaired.a3m"
    json_parent.mkdir(parents=True)
    msa_path.parent.mkdir(parents=True)
    msa_path.write_text(">query\nA\n>homolog\nG\n", encoding="utf-8")

    features = featurize_opendde_json(
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "A",
                        "unpairedMsaPath": "./examples/fixture/unpaired.a3m",
                    }
                }
            ]
        },
        base_dir=json_parent,
        n_queries=2,
        n_keys=4,
    )

    assert np.asarray(features["msa"]).shape == (2, 1)


def test_json_parent_relative_asset_path_has_priority_over_repo_fallback(
    tmp_path,
) -> None:
    repo_root = tmp_path / "OpenDDE"
    json_parent = repo_root / "inputs"
    preferred = json_parent / "examples" / "fixture" / "unpaired.a3m"
    fallback = repo_root / "examples" / "fixture" / "unpaired.a3m"
    preferred.parent.mkdir(parents=True)
    fallback.mkdir(parents=True)
    preferred.write_text(">query\nA\n", encoding="utf-8")

    features = featurize_opendde_json(
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "A",
                        "unpairedMsaPath": "./examples/fixture/unpaired.a3m",
                    }
                }
            ]
        },
        base_dir=json_parent,
        n_queries=2,
        n_keys=4,
    )

    np.testing.assert_array_equal(features["msa"], [[0], [0]])


def test_pinned_official_example_resolves_legacy_msa_from_repository_root() -> None:
    input_path = (
        Path(__file__).resolve().parents[4] / "OpenDDE" / "examples" / "example.json"
    )
    if not input_path.is_file():
        pytest.skip("pinned sibling OpenDDE checkout is unavailable")

    job = load_jobs(input_path)[0]
    features = featurize_opendde_json(
        job,
        base_dir=input_path.parent,
        n_queries=2,
        n_keys=4,
        max_msa_depth=4,
        seed=101,
    )

    assert 1 < np.asarray(features["msa"]).shape[0] <= 4
    assert (
        np.asarray(features["msa"]).shape[1] == np.asarray(features["restype"]).shape[0]
    )
