from __future__ import annotations

import json

import gemmi
import numpy as np
import pytest

from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.data.output import write_protenix_outputs


def _toy_features() -> dict[str, np.ndarray]:
    names = np.zeros((3, 4, 64), dtype=np.float32)
    for atom_i, name in enumerate(("N", "CA", "C")):
        for char_i, char in enumerate(name.ljust(4)):
            names[atom_i, char_i, ord(char) - 32] = 1.0
    elements = np.zeros((3, 128), dtype=np.float32)
    elements[:, [6, 5, 5]] = 1.0
    restype = np.zeros((1, 32), dtype=np.float32)
    restype[0, 0] = 1.0
    return {
        "atom_to_token_idx": np.zeros(3, dtype=np.int64),
        "ref_atom_name_chars": names,
        "ref_element": elements,
        "ref_mask": np.ones(3, dtype=np.float32),
        "restype": restype,
        "asym_id": np.zeros(1, dtype=np.int64),
        "residue_index": np.ones(1, dtype=np.int64),
    }


def test_writer_stably_ranks_samples_and_writes_plddt_b_factors(tmp_path) -> None:
    output = {
        "coordinate": np.array(
            [
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                [[0, 1, 0], [1, 1, 0], [2, 1, 0]],
                [[0, 2, 0], [1, 2, 0], [2, 2, 0]],
            ],
            dtype=np.float32,
        ),
        "atom_plddt": np.array(
            [[0.1, 0.2, 0.3], [0.8, 0.9, 1.0], [0.4, 0.5, 0.6]],
            dtype=np.float32,
        ),
        # Samples 1 and 2 tie; stable ranking keeps sample 1 first.
        "summary_ranking_score": np.array([0.2, 0.9, 0.9], dtype=np.float32),
        # This auxiliary score must not override canonical upstream ranking.
        "summary_ranking_score_vdw_penalized": np.array(
            [1.0, -100.0, -100.0], dtype=np.float32
        ),
        "summary_plddt": np.array([20.0, 90.0, 50.0], dtype=np.float32),
        "summary_ptm": np.array([0.2, 0.8, 0.7], dtype=np.float32),
        "has_clash": np.array([False, False, True]),
    }

    paths = write_protenix_outputs(
        tmp_path,
        job_name="toy job",
        seed=7,
        output=output,
        features=_toy_features(),
        include_raw=True,
    )

    prediction_dir = tmp_path / "toy_job" / "seed_7" / "predictions"
    assert prediction_dir / "toy_job_sample_0.cif" in paths
    assert prediction_dir / "toy_job_sample_1.cif" in paths
    assert prediction_dir / "toy_job_sample_2.cif" in paths
    assert (prediction_dir / "raw_output.npz").is_file()

    # Rank zero is original sample 1, including its coordinates and pLDDT * 100.
    structure = gemmi.read_structure(str(prediction_dir / "toy_job_sample_0.cif"))
    atoms = [cra.atom for cra in structure[0].all()]
    assert [atom.b_iso for atom in atoms] == pytest.approx([80.0, 90.0, 100.0])
    assert atoms[0].pos.y == pytest.approx(1.0)

    confidence = json.loads(
        (prediction_dir / "toy_job_summary_confidence_sample_0.json").read_text()
    )
    assert confidence == {
        "has_clash": False,
        "plddt": 90.0,
        "ptm": pytest.approx(0.8),
        "ranking_score": pytest.approx(0.9),
        "ranking_score_vdw_penalized": pytest.approx(-100.0),
    }


def test_writer_serializes_complete_per_sample_confidence(tmp_path) -> None:
    output = {
        "coordinate": np.zeros((2, 3, 3), dtype=np.float32),
        "atom_plddt": np.full((2, 3), 0.75, dtype=np.float32),
        "summary_ranking_score": np.array([0.8, 0.7], dtype=np.float32),
        "chain_ptm": np.array([[0.6], [0.5]], dtype=np.float32),
        "chain_iptm": np.array([[0.4], [0.3]], dtype=np.float32),
        "chain_pair_iptm": np.array([[[0.0]], [[0.0]]], dtype=np.float32),
        "chain_pair_iptm_global": np.array([[[0.1]], [[0.2]]], dtype=np.float32),
        "chain_plddt": np.array([[0.9], [0.8]], dtype=np.float32),
        "chain_pair_plddt": np.array([[[0.0]], [[0.0]]], dtype=np.float32),
        "chain_gpde": np.array([[1.5], [2.5]], dtype=np.float32),
        "chain_pair_gpde": np.array([[[0.0]], [[0.0]]], dtype=np.float32),
        "chain_pair_pae_mean": np.array([[[3.0]], [[4.0]]], dtype=np.float32),
        "chain_pair_pae_min": np.array([[[np.nan]], [[2.0]]], dtype=np.float32),
        "disorder": np.array([0.0, 0.1], dtype=np.float32),
        "num_recycles": np.array(10, dtype=np.int32),
        "has_clash": np.array([False, True]),
        "has_vdw_clash": np.array([True, False]),
        # Full-data tensors must remain out of the summary JSON.
        "token_pair_pae": np.zeros((2, 1, 1), dtype=np.float32),
        # OpenDDE opts this field in explicitly; the shared default remains
        # Protenix's own allowlist.
        "shape_comp_global_pred": np.asarray([0.2, 0.3], dtype=np.float32),
    }

    write_protenix_outputs(
        tmp_path, job_name="complete", seed=1, output=output, features=_toy_features()
    )

    confidence = json.loads(
        (
            tmp_path
            / "complete"
            / "seed_1"
            / "predictions"
            / "complete_summary_confidence_sample_0.json"
        ).read_text()
    )
    assert confidence["chain_ptm"] == pytest.approx([0.6])
    np.testing.assert_allclose(confidence["chain_pair_iptm_global"], [[0.1]])
    np.testing.assert_allclose(confidence["chain_pair_pae_mean"], [[3.0]])
    assert confidence["chain_pair_pae_min"] == [[None]]
    assert confidence["disorder"] == pytest.approx(0.0)
    assert confidence["num_recycles"] == 10
    assert confidence["has_clash"] is False
    assert confidence["has_vdw_clash"] is True
    assert "token_pair_pae" not in confidence
    assert "shape_comp_global_pred" not in confidence


def test_writer_does_not_substitute_plddt_for_canonical_ranking(tmp_path) -> None:
    output = {
        "coordinate": np.array(
            [
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                [[0, 1, 0], [1, 1, 0], [2, 1, 0]],
            ],
            dtype=np.float32,
        ),
        "atom_plddt": np.full((2, 3), 0.5, dtype=np.float32),
        "summary_plddt": np.array([10.0, 90.0], dtype=np.float32),
    }

    write_protenix_outputs(
        tmp_path, job_name="unranked", seed=1, output=output, features=_toy_features()
    )

    structure = gemmi.read_structure(
        str(tmp_path / "unranked" / "seed_1" / "predictions" / "unranked_sample_0.cif")
    )
    assert next(iter(structure[0].all())).atom.pos.y == pytest.approx(0.0)


def test_writer_uses_explicit_atom_metadata_contract(tmp_path) -> None:
    features = _toy_features()
    features.update(
        {
            "output_atom_name": np.array(["X1", "X2", "X3"]),
            "output_atom_element": np.array(["C", "N", "O"]),
            "output_atom_res_name": np.array(["LIG", "LIG", "LIG"]),
            "output_atom_chain_id": np.array(["Z", "Z", "Z"]),
            "output_atom_res_id": np.array([4, 4, 4]),
        }
    )
    output = {
        "coordinate": np.zeros((1, 3, 3), dtype=np.float32),
        "atom_plddt": np.full((1, 3), 0.75, dtype=np.float32),
        "summary_ranking_score": np.array([1.0]),
    }

    write_protenix_outputs(
        tmp_path, job_name="ligand", seed=1, output=output, features=features
    )

    structure = gemmi.read_structure(
        str(tmp_path / "ligand" / "seed_1" / "predictions" / "ligand_sample_0.cif")
    )
    residue = structure[0]["Z"][0]
    assert residue.name == "LIG"
    assert residue.seqid.num == 4
    assert [atom.name for atom in residue] == ["X1", "X2", "X3"]


def test_writer_requires_confidence_for_original_output(tmp_path) -> None:
    with pytest.raises(ValueError, match="atom_plddt"):
        write_protenix_outputs(
            tmp_path,
            job_name="toy",
            seed=1,
            output={"coordinate": np.zeros((1, 3, 3), dtype=np.float32)},
            features=_toy_features(),
        )


def _bonded_protein_ion_features() -> dict[str, np.ndarray]:
    return featurize_protein_json(
        {
            "name": "bonded",
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "C",
                        "count": 1,
                        "id": ["P"],
                    }
                },
                {"ion": {"ion": "MG", "count": 1, "id": ["M"]}},
            ],
            "covalent_bonds": [
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "MG",
                }
            ],
        },
        n_queries=2,
        n_keys=2,
    )


def _single_sample_output(n_atom: int) -> dict[str, np.ndarray]:
    return {
        "coordinate": np.zeros((1, n_atom, 3), dtype=np.float32),
        "atom_plddt": np.full((1, n_atom), 0.75, dtype=np.float32),
        "summary_ranking_score": np.ones((1,), dtype=np.float32),
    }


def test_writer_preserves_entities_and_emits_explicit_covalent_struct_conn(
    tmp_path,
) -> None:
    features = _bonded_protein_ion_features()
    n_atom = len(features["atom_to_token_idx"])

    [cif_path, _summary_path] = write_protenix_outputs(
        tmp_path,
        job_name="bonded",
        seed=1,
        output=_single_sample_output(n_atom),
        features=features,
    )

    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    entity = block.get_mmcif_category("_entity.")
    assert entity["id"] == ["1", "2"]
    assert entity["type"] == ["polymer", "non-polymer"]
    assert block.get_mmcif_category("_entity_poly.")["entity_id"] == ["1"]
    assert block.get_mmcif_category("_struct_asym.") == {
        "id": ["P", "M"],
        "entity_id": ["1", "2"],
    }

    atom_site = block.get_mmcif_category("_atom_site.")
    assert set(atom_site["label_entity_id"][:-1]) == {"1"}
    assert atom_site["label_entity_id"][-1] == "2"
    assert atom_site["group_PDB"][-1] == "HETATM"
    # Standards-correct non-polymers have no label sequence number.  This is an
    # intentional correction of the pinned writer's ligand-as-polymer quirk.
    assert atom_site["label_seq_id"][-1] is False

    connection = block.get_mmcif_category("_struct_conn.")
    assert connection["conn_type_id"] == ["covale"]
    assert connection["pdbx_value_order"] == ["sing"]
    assert connection["ptnr1_label_asym_id"] == ["P"]
    assert connection["ptnr1_label_comp_id"] == ["CYS"]
    assert connection["ptnr1_label_seq_id"] == ["1"]
    assert connection["ptnr1_label_atom_id"] == ["SG"]
    assert connection["ptnr2_label_asym_id"] == ["M"]
    assert connection["ptnr2_label_comp_id"] == ["MG"]
    assert connection["ptnr2_label_seq_id"] == [False]
    assert connection["ptnr2_label_atom_id"] == ["MG"]


def test_writer_keeps_entity_identity_across_copies_and_identical_entries(
    tmp_path,
) -> None:
    features = featurize_protein_json(
        {
            "name": "copies",
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "C",
                        "count": 2,
                        "id": ["A", "B"],
                    }
                },
                {
                    "proteinChain": {
                        "sequence": "C",
                        "count": 1,
                        "id": ["C"],
                    }
                },
            ],
        },
        n_queries=2,
        n_keys=2,
    )
    n_atom = len(features["atom_to_token_idx"])

    [cif_path, _summary_path] = write_protenix_outputs(
        tmp_path,
        job_name="copies",
        seed=1,
        output=_single_sample_output(n_atom),
        features=features,
    )

    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    assert block.get_mmcif_category("_struct_asym.") == {
        "id": ["A", "B", "C"],
        "entity_id": ["1", "1", "2"],
    }
    atom_site = block.get_mmcif_category("_atom_site.")
    chain_to_entities: dict[str, set[str]] = {}
    for chain_id, entity_id in zip(
        atom_site["label_asym_id"], atom_site["label_entity_id"]
    ):
        chain_to_entities.setdefault(chain_id, set()).add(entity_id)
    assert chain_to_entities == {"A": {"1"}, "B": {"1"}, "C": {"2"}}


def test_writer_suppresses_explicit_intra_residue_bond(tmp_path) -> None:
    features = _bonded_protein_ion_features()
    features["covalent_atom_indices"] = np.asarray([[0, 1]], dtype=np.int64)
    n_atom = len(features["atom_to_token_idx"])

    [cif_path, _summary_path] = write_protenix_outputs(
        tmp_path,
        job_name="intra-residue",
        seed=1,
        output=_single_sample_output(n_atom),
        features=features,
    )

    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    assert block.get_mmcif_category("_struct_conn.") == {}


def test_writer_rejects_incomplete_or_out_of_range_covalent_metadata(tmp_path) -> None:
    features = _bonded_protein_ion_features()
    n_atom = len(features["atom_to_token_idx"])
    output = _single_sample_output(n_atom)

    incomplete = dict(features)
    incomplete.pop("atom_entity_id")
    with pytest.raises(ValueError, match="covalent output metadata.*atom_entity_id"):
        write_protenix_outputs(
            tmp_path,
            job_name="incomplete",
            seed=1,
            output=output,
            features=incomplete,
        )

    out_of_range = dict(features)
    out_of_range["covalent_atom_indices"] = np.asarray([[0, n_atom]])
    with pytest.raises(ValueError, match="out-of-range atom index"):
        write_protenix_outputs(
            tmp_path,
            job_name="out-of-range",
            seed=1,
            output=output,
            features=out_of_range,
        )


def _metadata_features(names, chains, res_ids):
    """Explicit writer metadata for ``len(names)`` atoms, nothing else read."""
    count = len(names)
    return {
        "output_atom_name": np.array(names),
        "output_atom_element": np.array([name[0] for name in names]),
        "output_atom_res_name": np.array(["ALA"] * count),
        "output_atom_chain_id": np.array(chains),
        "output_atom_res_id": np.array(res_ids),
    }


def _angle(vertex, left, right):
    a, b = left - vertex, right - vertex
    cosine = np.sum(a * b, -1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def test_cterminal_oxygens_are_rebuilt_as_upstream_does() -> None:
    """Upstream 4c355be `runner/inference.py:261-362`, applied at `:600`.

    The oxygen nearer C stays in the O slot and OXT becomes its reflection
    across the C->CA axis: equal C-O and C-OXT bond lengths, equal CA-C-O and
    CA-C-OXT angles, and O, OXT, C and CA coplanar. Nothing else moves.
    """
    from foldjax.models.protenix.data.output import fix_cterminal_carboxyl_oxygens

    names = [
        # chain A: two residues, the second the C-terminus
        "N", "CA", "C", "O", "N", "CA", "C", "O", "OXT",
        # chain B: a terminus with no OXT (left alone)
        "N", "CA", "C", "O",
        # chain C: a ligand
        "C1", "O1",
        # chain D: one residue whose O drifted beyond OXT
        "N", "CA", "C", "O", "OXT",
    ]
    chains = ["A"] * 9 + ["B"] * 4 + ["C"] * 2 + ["D"] * 5
    res_ids = [1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    features = _metadata_features(names, chains, res_ids)
    rng = np.random.default_rng(3)
    coordinates = rng.normal(scale=3.0, size=(3, len(names), 3)).astype(np.float32)
    # Chain D's O far away and OXT close: the O slot must take OXT's position.
    coordinates[:, 18] = coordinates[:, 17] + 25.0
    coordinates[:, 19] = coordinates[:, 17] + np.float32([1.2, 0.3, 0.1])
    original = coordinates.copy()

    fixed = fix_cterminal_carboxyl_oxygens(coordinates, features)

    np.testing.assert_array_equal(coordinates, original)  # not mutated
    assert fixed.dtype == np.float32 and fixed.shape == coordinates.shape
    changed = {7, 8, 18, 19}
    untouched = [index for index in range(len(names)) if index not in changed]
    np.testing.assert_array_equal(
        fixed[:, untouched].view(np.uint32), original[:, untouched].view(np.uint32)
    )
    for c, ca, o, oxt in ((6, 5, 7, 8), (17, 16, 18, 19)):
        carbon, alpha = fixed[:, c], fixed[:, ca]
        oxygen, terminal = fixed[:, o], fixed[:, oxt]
        np.testing.assert_allclose(
            np.linalg.norm(oxygen - carbon, axis=-1),
            np.linalg.norm(terminal - carbon, axis=-1),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            _angle(carbon, alpha, oxygen), _angle(carbon, alpha, terminal), atol=1e-2
        )
        volume = np.einsum(
            "sk,sk->s", np.cross(alpha - carbon, oxygen - carbon), terminal - carbon
        )
        np.testing.assert_allclose(volume, 0.0, atol=1e-3)
    # Chain A keeps whichever of its two oxygens was nearer C in the O slot.
    near_o = np.linalg.norm(original[:, 7] - original[:, 6], axis=-1) <= (
        np.linalg.norm(original[:, 8] - original[:, 6], axis=-1)
    )
    np.testing.assert_array_equal(
        fixed[:, 7], np.where(near_o[:, None], original[:, 7], original[:, 8])
    )
    # Chain D's O slot now holds the old OXT, the nearer oxygen.
    np.testing.assert_array_equal(fixed[:, 18], original[:, 19])


def test_cterminal_rebuild_keeps_o_on_a_tie_and_skips_chains_without_oxt() -> None:
    from foldjax.models.protenix.data.output import fix_cterminal_carboxyl_oxygens

    features = _metadata_features(["N", "CA", "C", "O", "OXT"], ["A"] * 5, [1] * 5)
    # C at the origin, CA on +x; O and OXT equidistant from C (a tie).
    coordinates = np.array(
        [[[9, 9, 9], [1.5, 0, 0], [0, 0, 0], [-0.6, 1.1, 0], [-0.6, 0, 1.1]]],
        dtype=np.float32,
    )
    fixed = fix_cterminal_carboxyl_oxygens(coordinates, features)
    np.testing.assert_array_equal(fixed[0, 3], coordinates[0, 3])
    np.testing.assert_allclose(fixed[0, 4], [-0.6, -1.1, 0.0], atol=1e-6)

    no_oxt = _metadata_features(["N", "CA", "C", "O"], ["A"] * 4, [1] * 4)
    plain = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    np.testing.assert_array_equal(fix_cterminal_carboxyl_oxygens(plain, no_oxt), plain)


def test_generated_protein_chains_carry_the_quartet_the_rebuild_needs() -> None:
    """The featurizer gives each protein C-terminus its OXT, so the fix fires."""
    from foldjax.models.protenix.data.output import fix_cterminal_carboxyl_oxygens

    features = featurize_protein_json(
        {"sequences": [{"proteinChain": {"sequence": "AG", "count": 2}}]},
        n_queries=2,
        n_keys=4,
    )
    names = np.asarray(features["output_atom_name"]).astype(str)
    chains = np.asarray(features["output_atom_chain_id"]).astype(str)
    assert (names == "OXT").sum() == 2
    rng = np.random.default_rng(0)
    coordinates = rng.normal(size=(2, names.size, 3)).astype(np.float32)
    fixed = fix_cterminal_carboxyl_oxygens(coordinates, features)
    moved = np.flatnonzero(np.any(fixed != coordinates, axis=(0, 2)))
    assert set(names[moved]) <= {"O", "OXT"}
    for chain in np.unique(chains):
        oxt = np.flatnonzero((names == "OXT") & (chains == chain))
        assert oxt.size == 1 and oxt[0] in moved
