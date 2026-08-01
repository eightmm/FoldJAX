"""Parity and contract tests for deterministic inference feature generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.data.features import FEATURE_NAMES, generate_features
from foldjax.models.chai.data.msa import NO_PAIRING_KEY


def _inputs() -> dict[str, np.ndarray]:
    batch, tokens, templates = 1, 4, 2
    query = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    key = query.copy()
    atom_ref_mask = np.asarray([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=np.bool_)
    query_mask = atom_ref_mask[:, query]
    key_mask = atom_ref_mask[:, key]
    block_mask = query_mask[..., :, None] & key_mask[..., None, :]

    atom_coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [9.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )[None]
    template_distances = np.zeros((batch, templates, tokens, tokens), dtype=np.float32)
    template_distances[:, :, 0, 1] = 4.0
    template_distances[:, :, 1, 0] = 4.0
    msa_tokens = np.asarray(
        [[[0, 1, 2, 32], [0, 1, 3, 32], [0, 4, 3, 32]]], dtype=np.uint8
    )
    msa_mask = np.asarray([[[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]]], dtype=np.bool_)
    pairkey = np.asarray(
        [
            [
                [17, 17, 17, NO_PAIRING_KEY],
                [17, 17, 99, NO_PAIRING_KEY],
                [23, 23, NO_PAIRING_KEY, NO_PAIRING_KEY],
            ]
        ],
        dtype=np.int32,
    )
    deletion = np.asarray([[[0, 1, 4, 0], [0, 0, 2, 0], [3, 0, 0, 0]]], dtype=np.uint8)
    return {
        "token_residue_index": np.asarray([[0, 1, 0, 0]], dtype=np.int32),
        "token_asym_id": np.asarray([[1, 1, 2, 0]], dtype=np.int32),
        "token_index": np.arange(tokens, dtype=np.int32)[None],
        "token_entity_id": np.asarray([[1, 1, 2, 0]], dtype=np.int32),
        "token_sym_id": np.asarray([[1, 1, 1, 0]], dtype=np.int32),
        "token_entity_type": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        "token_residue_type": np.asarray([[0, 1, 2, 32]], dtype=np.int32),
        "token_exists_mask": np.asarray([[1, 1, 1, 0]], dtype=np.bool_),
        "token_b_factor_or_plddt": np.zeros((batch, tokens), dtype=np.float32),
        "is_distillation": np.zeros((batch, 1), dtype=np.bool_),
        "esm_embeddings": np.zeros((batch, tokens, 2560), dtype=np.float32),
        "atom_ref_pos": atom_coords.copy(),
        "atom_ref_space_uid": np.asarray([[0, 0, 1, 1, 2, 2, -1, -1]], dtype=np.int32),
        "atom_ref_charge": np.asarray([[0, 1, -1, 0, 0, 0, 0, 0]], dtype=np.int32),
        "atom_ref_mask": atom_ref_mask,
        "atom_ref_element": np.asarray([[6, 7, 8, 16, 6, 6, 0, 0]], dtype=np.int32),
        "atom_ref_name_chars": np.asarray(
            [
                [
                    [2, 1, 0, 0],
                    [3, 1, 0, 0],
                    [4, 1, 0, 0],
                    [5, 1, 0, 0],
                    [2, 2, 0, 0],
                    [3, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ],
            dtype=np.int32,
        ),
        "block_atom_pair_q_idces": query,
        "block_atom_pair_kv_idces": key,
        "block_atom_pair_mask": block_mask,
        "atom_gt_coords": atom_coords,
        "atom_exists_mask": atom_ref_mask.copy(),
        "atom_token_index": np.asarray([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=np.int32),
        "token_centre_atom_index": np.asarray([[0, 2, 4, 6]], dtype=np.int32),
        "subchain_id": np.zeros((batch, tokens, 4), dtype=np.uint8),
        "token_residue_name": np.zeros((batch, tokens, 8), dtype=np.uint8),
        "template_backbone_frame_mask": np.asarray(
            [[[1, 1, 0, 0], [1, 0, 0, 0]]], dtype=np.bool_
        ),
        "template_pseudo_beta_mask": np.asarray(
            [[[1, 1, 1, 0], [1, 0, 0, 0]]], dtype=np.bool_
        ),
        "template_distances": template_distances,
        "template_unit_vector": np.ones(
            (batch, templates, tokens, tokens, 3), dtype=np.float32
        ),
        "template_restype": np.asarray(
            [[[0, 1, 2, 31], [3, 31, 31, 31]]], dtype=np.int32
        ),
        "msa_tokens": msa_tokens,
        "msa_mask": msa_mask,
        "msa_deletion_matrix": deletion,
        "msa_pairkey": pairkey,
        "msa_sequence_source": np.asarray(
            [[[5, 5, 5, 4], [0, 0, 0, 4], [1, 1, 4, 4]]], dtype=np.uint8
        ),
        "main_msa_tokens": msa_tokens.copy(),
        "main_msa_mask": msa_mask.copy(),
        "main_msa_deletion_matrix": deletion.copy(),
    }


def test_feature_schema_matches_model_contract() -> None:
    features = generate_features(_inputs())
    assert tuple(features) == FEATURE_NAMES
    assert len(features) == 32
    expected_shapes = {
        "RelativeSequenceSeparation": (1, 4, 4),
        "RelativeTokenSeparation": (1, 4, 4),
        "RelativeEntity": (1, 4, 4),
        "RelativeChain": (1, 4, 4),
        "ResidueType": (1, 4),
        "ESMEmbeddings": (1, 4, 2560),
        "BlockedAtomPairDistogram": (1, 2, 4, 4),
        "InverseSquaredBlockedAtomPairDistances": (1, 2, 4, 4, 2),
        "AtomRefPos": (1, 8, 3),
        "AtomRefCharge": (1, 8),
        "AtomRefMask": (1, 8),
        "AtomRefElement": (1, 8),
        "AtomNameOneHot": (1, 8, 4),
        "TemplateMask": (1, 2, 4, 4, 2),
        "TemplateUnitVector": (1, 2, 4, 4, 3),
        "TemplateResType": (1, 2, 4, 1),
        "TemplateDistogram": (1, 2, 4, 4),
        "TokenDistanceRestraint": (1, 4, 4, 1),
        "DockingConstraintGenerator": (1, 4, 4),
        "TokenPairPocketRestraint": (1, 4, 4, 1),
        "MSAProfile": (1, 4, 33),
        "MSADeletionMean": (1, 4),
        "IsDistillation": (1, 4),
        "TokenBFactor": (1, 4),
        "TokenPLDDT": (1, 4),
        "ChainIsCropped": (1, 4),
        "MissingChainContact": (1, 4),
        "MSAOneHot": (1, 3, 4),
        "MSAHasDeletion": (1, 3, 4),
        "MSADeletionValue": (1, 3, 4),
        "IsPairedMSA": (1, 3, 4),
        "MSADataSource": (1, 3, 4),
    }
    assert {name: value.shape for name, value in features.items()} == expected_shapes


def test_masks_defaults_and_inputs_are_deterministic() -> None:
    inputs = _inputs()
    source_before = inputs["msa_sequence_source"].copy()
    first = generate_features(inputs)
    second = generate_features(inputs)
    for name in FEATURE_NAMES:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(inputs["msa_sequence_source"], source_before)

    assert np.all(first["TokenDistanceRestraint"] == -1.0)
    assert np.all(first["TokenPairPocketRestraint"] == -1.0)
    assert np.all(first["DockingConstraintGenerator"] == 5)
    assert np.all(first["TokenBFactor"] == 2)
    assert np.all(first["TokenPLDDT"] == 3)
    assert first["RelativeSequenceSeparation"][0, 0, 2] == 66
    assert first["RelativeTokenSeparation"][0, 0, 1] == 66
    assert first["MSADataSource"][0, 0, 0] == 4
    assert first["MSADataSource"][0, 0, 3] == 4
    assert np.all(first["MissingChainContact"] == 0.0)


def test_missing_chain_contact_marks_every_disconnected_chain() -> None:
    inputs = _inputs()
    inputs["atom_gt_coords"][0, 4:6, 0] += 100.0
    feature = generate_features(inputs)["MissingChainContact"]
    np.testing.assert_array_equal(feature, [[1.0, 1.0, 1.0, 0.0]])


@pytest.mark.parametrize(
    "name", ["contact_constraints", "docking_constraints", "pocket_constraints"]
)
def test_malformed_manual_restraints_fail_explicitly(name: str) -> None:
    inputs = _inputs()
    inputs[name] = [[{"distance_threshold": 8.0}]]
    with pytest.raises(ValueError, match="constraint"):
        generate_features(inputs)


def test_null_restraint_containers_are_accepted() -> None:
    inputs = _inputs()
    inputs.update(
        contact_constraints=[[None]],
        docking_constraints=[None],
        pocket_constraints=np.asarray([[None]], dtype=object),
    )
    assert len(generate_features(inputs)) == 32


@pytest.mark.official_parity
def test_official_feature_factory_parity(
    tmp_path: Path, upstream_chai_dir: Path, upstream_chai_python: Path
) -> None:
    input_path = tmp_path / "inputs.npz"
    output_path = tmp_path / "official.npz"
    inputs = _inputs()
    np.savez(input_path, **inputs)
    script = r"""
import sys
import numpy as np
import torch
from chai_lab.chai1 import feature_factory

loaded = np.load(sys.argv[1], allow_pickle=False)
inputs = {name: torch.from_numpy(loaded[name]) for name in loaded.files}
features = feature_factory.generate({"inputs": inputs})
keep_last = {
    "ESMEmbeddings", "InverseSquaredBlockedAtomPairDistances", "AtomRefPos",
    "AtomNameOneHot", "TemplateMask", "TemplateUnitVector", "TemplateResType",
    "TokenDistanceRestraint", "TokenPairPocketRestraint", "MSAProfile",
}
arrays = {}
for name, value in features.items():
    array = value.detach().cpu().numpy()
    if name not in keep_last:
        array = np.squeeze(array, axis=-1)
    arrays[name] = array
np.savez(sys.argv[2], **arrays)
"""
    subprocess.run(
        [str(upstream_chai_python), "-c", script, str(input_path), str(output_path)],
        cwd=upstream_chai_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = np.load(output_path)
    actual = generate_features(inputs)
    for name in FEATURE_NAMES:
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        if np.issubdtype(expected[name].dtype, np.floating):
            np.testing.assert_allclose(
                actual[name], expected[name], atol=2e-6, rtol=2e-6
            )
        else:
            np.testing.assert_array_equal(actual[name], expected[name])
