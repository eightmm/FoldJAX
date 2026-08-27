from __future__ import annotations

import json

import numpy as np
import pytest

from foldjax.models.protenix.data.featurize_json import (
    _msa_profile,
    featurize_protein_json,
    load_first_job,
    main,
    parse_a3m_profile,
    parse_a3m_rows,
)
from foldjax.models.protenix.data.static_io import load_static_feature_npz


def test_msa_profile_bounds_one_hot_chunks_without_changing_bits(monkeypatch) -> None:
    rng = np.random.default_rng(13)
    msa = rng.integers(0, 32, size=(37, 19), dtype=np.int64)
    expected = ((msa[..., None] == np.arange(32)).sum(axis=0) / msa.shape[0]).astype(
        np.float32
    )

    import foldjax.models.protenix.data.featurize_json as featurize_json

    budget = 19 * 32 * 5
    monkeypatch.setattr(featurize_json, "_MSA_PROFILE_TEMP_BUDGET", budget)
    real_sum = np.sum
    temporary_sizes: list[int] = []

    def tracked_sum(value, *args, **kwargs):
        temporary_sizes.append(value.nbytes)
        return real_sum(value, *args, **kwargs)

    monkeypatch.setattr(featurize_json.np, "sum", tracked_sum)
    actual = _msa_profile(msa)

    assert np.array_equal(actual, expected)
    assert len(temporary_sizes) > 1
    assert max(temporary_sizes) <= budget


def test_featurize_sequence_only_protein_json() -> None:
    job = {
        "name": "toy",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGX",
                    "count": 2,
                    "id": ["A", "B"],
                },
            },
        ],
    }

    features = featurize_protein_json(job, n_queries=2, n_keys=4)

    assert features["restype"].shape == (6, 32)
    np.testing.assert_array_equal(features["restype"].argmax(axis=-1), [0, 7, 20] * 2)
    np.testing.assert_array_equal(features["profile"], features["restype"])
    np.testing.assert_array_equal(features["deletion_mean"], np.zeros(6))
    np.testing.assert_array_equal(features["residue_index"], [1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(features["asym_id"], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(features["entity_id"], [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(features["sym_id"], [0, 0, 0, 1, 1, 1])
    assert features["token_bonds"].shape == (6, 6)
    assert features["atom_to_token_idx"].shape == (32,)
    assert features["ref_pos"].shape == (32, 3)
    assert features["ref_element"].shape == (32, 128)
    assert features["ref_atom_name_chars"].shape == (32, 4, 64)
    assert features["d_lm"].shape[-1] == 3
    assert features["v_lm"].shape[-1] == 1
    assert features["pad_info"]["mask_trunked"].shape == features["v_lm"].shape[:-1]
    assert features["has_frame"].tolist() == [1] * 6
    assert features["distogram_rep_atom_mask"].sum() == 6


def test_local_atom_geometry_masks_cross_residue_reference_pairs() -> None:
    features = featurize_protein_json(
        {"sequences": [{"proteinChain": {"sequence": "AG"}}]},
        n_queries=32,
        n_keys=128,
    )

    atom_to_token = features["atom_to_token_idx"]
    first_residue_atom = int(np.flatnonzero(atom_to_token == 0)[0])
    second_residue_atom = int(np.flatnonzero(atom_to_token == 1)[0])
    key_offset = (128 - 32) // 2

    assert features["pad_info"]["mask_trunked"][
        0, first_residue_atom, key_offset + second_residue_atom
    ]
    assert features["v_lm"][
        0, first_residue_atom, key_offset + second_residue_atom, 0
    ] == 0.0
    assert features["v_lm"][
        0, first_residue_atom, key_offset + first_residue_atom, 0
    ] == 1.0


@pytest.mark.parametrize(
    ("sequence_entry", "expected_ref_spaces"),
    [
        ({"proteinChain": {"sequence": "AG"}}, 2),
        ({"dnaSequence": {"sequence": "GA"}}, 2),
        ({"rnaSequence": {"sequence": "GA"}}, 2),
        ({"ligand": {"ligand": "CCD_ATP"}}, 1),
        ({"ion": {"ion": "MG"}}, 1),
    ],
)
def test_reference_positions_are_centered_per_reference_space(
    sequence_entry: dict[str, dict[str, str]],
    expected_ref_spaces: int,
) -> None:
    features = featurize_protein_json({"sequences": [sequence_entry]})

    ref_space_uid = features["ref_space_uid"]
    assert np.unique(ref_space_uid).size == expected_ref_spaces
    for uid in np.unique(ref_space_uid):
        np.testing.assert_allclose(
            features["ref_pos"][ref_space_uid == uid].mean(axis=0),
            np.zeros(3, dtype=np.float32),
            atol=1e-6,
        )


def test_parse_a3m_profile_maps_insertions_and_ambiguous_codes() -> None:
    profile, deletion_mean = parse_a3m_profile(
        "ACD",
        ">query\nACD\n>hit1\nAc-D\n>hit2\nAZJ\n",
    )

    assert profile.shape == (3, 32)
    np.testing.assert_allclose(deletion_mean, [0.0, 1 / 3, 0.0])
    np.testing.assert_allclose(profile[0, 0], 1.0)
    np.testing.assert_allclose(profile[1, 4], 1 / 3)
    np.testing.assert_allclose(profile[1, 6], 1 / 3)
    np.testing.assert_allclose(profile[1, 31], 1 / 3)
    np.testing.assert_allclose(profile[2, 3], 2 / 3)
    np.testing.assert_allclose(profile[2, 20], 1 / 3)


def test_parse_a3m_rows_matches_protenix_deletion_encoding() -> None:
    msa, deletion_matrix = parse_a3m_rows(
        "AG",
        ">query\nAG\n>hit\nAc-\n",
    )

    np.testing.assert_array_equal(msa, [[0, 7], [0, 31]])
    np.testing.assert_array_equal(deletion_matrix, [[0, 0], [0, 1]])


def test_parse_a3m_profile_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="aligned length"):
        parse_a3m_profile("ACD", ">query\nACD\n>bad\nAC\n")


def test_featurize_json_uses_msa_profile(tmp_path) -> None:
    # Sequence length must exceed 4 or torch skips the unpaired MSA.
    msa_path = tmp_path / "toy.a3m"
    msa_path.write_text(">query\nAGCDE\n>hit\nAGCD-\n")
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGCDE",
                    "unpairedMsaPath": "toy.a3m",
                }
            }
        ]
    }

    features = featurize_protein_json(job, base_dir=tmp_path, n_queries=2, n_keys=4)

    np.testing.assert_allclose(features["profile"][0, 0], 1.0)
    # Last column: 50% Glu(E=6 in query) ... actually E=6, gap=31 from hit.
    np.testing.assert_allclose(features["profile"][4, 6], 0.5)
    np.testing.assert_allclose(features["profile"][4, 31], 0.5)
    np.testing.assert_allclose(features["deletion_mean"], np.zeros(5))


def test_featurize_json_emits_global_msa_rows(tmp_path) -> None:
    # Two distinct protein chains -> torch multimer pairing path.
    # Chain A unpaired MSA: query + one hit with an insertion column.
    msa_path = tmp_path / "a.a3m"
    msa_path.write_text(">query\nAGCDE\n>hit\nAGCDc-\n")
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGCDE",
                    "unpairedMsaPath": "a.a3m",
                }
            },
            {"proteinChain": {"sequence": "KLMNP"}},
        ]
    }

    features = featurize_protein_json(job, base_dir=tmp_path, n_queries=2, n_keys=4)

    # Row 0: paired query block (chain A | chain B). Row 1: chain A unpaired hit
    # padded with gaps over chain B columns (paired block first, then unpaired).
    np.testing.assert_array_equal(
        features["msa"],
        [
            [0, 7, 4, 3, 6, 11, 10, 12, 2, 14],
            [0, 7, 4, 3, 31, 31, 31, 31, 31, 31],
        ],
    )
    expected_has_del = np.zeros((2, 10), dtype=np.float32)
    expected_has_del[1, 4] = 1.0
    np.testing.assert_array_equal(features["has_deletion"], expected_has_del)
    expected_deletion = np.zeros((2, 10), dtype=np.float32)
    expected_deletion[1, 4] = np.arctan(1 / 3) * (2 / np.pi)
    np.testing.assert_allclose(features["deletion_value"], expected_deletion)


def test_featurize_json_multimer_pairing_matches_torch() -> None:
    # Golden arrays produced by torch Protenix FeatureAssemblyLine.assemble
    # on the same paired+unpaired 2-chain input (see msa_featurizer.py).
    seq_a = "AGCDEFHIKL"
    seq_b = "KLMNPQRSTV"
    paired_a = (
        ">q\nAGCDEFHIKL\n>UniRef100_a_HUMAN\nAGCDAFHIKL\n"
        ">UniRef100_b_MOUSE\nAGCDEFHIKA\n"
    )
    paired_b = (
        ">q\nKLMNPQRSTV\n>UniRef100_c_HUMAN\nKLMNAQRSTV\n"
        ">UniRef100_d_MOUSE\nKLMNPQRSTA\n"
    )
    unpaired_a = ">q\nAGCDEFHIKL\n>h1\nAGCDeFHIKL\n>h2\nAGCDEFHIKK\n"
    unpaired_b = ">q\nKLMNPQRSTV\n>h1\nKLMN-QRSTV\n"
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": seq_a,
                    "pairedMsa": paired_a,
                    "unpairedMsa": unpaired_a,
                }
            },
            {
                "proteinChain": {
                    "sequence": seq_b,
                    "pairedMsa": paired_b,
                    "unpairedMsa": unpaired_b,
                }
            },
        ]
    }

    features = featurize_protein_json(job)

    expected_msa = [
        [0, 7, 4, 3, 6, 13, 8, 9, 11, 10, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19],
        [0, 7, 4, 3, 0, 13, 8, 9, 11, 10, 11, 10, 12, 2, 0, 5, 1, 15, 16, 19],
        [0, 7, 4, 3, 6, 13, 8, 9, 11, 0, 11, 10, 12, 2, 14, 5, 1, 15, 16, 0],
        [0, 7, 4, 3, 13, 8, 9, 11, 10, 0, 11, 10, 12, 2, 31, 5, 1, 15, 16, 19],
        [0, 7, 4, 3, 6, 13, 8, 9, 11, 11, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31],
    ]
    np.testing.assert_array_equal(features["msa"], expected_msa)

    expected_has_del = np.zeros((5, 20), dtype=np.float32)
    expected_has_del[3, 4] = 1.0
    np.testing.assert_array_equal(features["has_deletion"], expected_has_del)

    expected_dv = np.zeros((5, 20), dtype=np.float32)
    expected_dv[3, 4] = np.arctan(1 / 3) * (2 / np.pi)
    np.testing.assert_allclose(features["deletion_value"], expected_dv, atol=1e-6)

    expected_dm = np.zeros(20, dtype=np.float32)
    expected_dm[4] = 1 / 3
    np.testing.assert_allclose(features["deletion_mean"], expected_dm, atol=1e-6)


def test_featurize_json_paired_msa_path_loading(tmp_path) -> None:
    # pairedMsaPath/unpairedMsaPath must match inline pairedMsa/unpairedMsa.
    seq_a, seq_b = "AGCDEFHIKL", "KLMNPQRSTV"
    (tmp_path / "pa.a3m").write_text(
        ">q\nAGCDEFHIKL\n>UniRef100_a_HUMAN\nAGCDAFHIKL\n"
    )
    (tmp_path / "pb.a3m").write_text(
        ">q\nKLMNPQRSTV\n>UniRef100_c_HUMAN\nKLMNAQRSTV\n"
    )
    (tmp_path / "ua.a3m").write_text(">q\nAGCDEFHIKL\n>h2\nAGCDEFHIKK\n")
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": seq_a,
                    "pairedMsaPath": "pa.a3m",
                    "unpairedMsaPath": "ua.a3m",
                }
            },
            {"proteinChain": {"sequence": seq_b, "pairedMsaPath": "pb.a3m"}},
        ]
    }
    features = featurize_protein_json(job, base_dir=tmp_path)
    # Paired query row first, then HUMAN-paired row, then chain-A unpaired hit.
    assert features["msa"].shape == (3, 20)
    np.testing.assert_array_equal(features["msa"][0, :4], [0, 7, 4, 3])
    # chain B columns of the unpaired row are gap-padded.
    np.testing.assert_array_equal(features["msa"][2, 10:], [31] * 10)


def test_featurize_json_rejects_unsupported_entities() -> None:
    job = {
        "name": "bad",
        "sequences": [{"peptideNucleicAcid": {"sequence": "ACGT", "count": 1}}],
    }

    with pytest.raises(ValueError, match="unsupported entity kind"):
        featurize_protein_json(job)


def test_featurize_dna_sequence() -> None:
    # GATC: per-residue tokens; OP3 only on the 5'-terminal residue.
    job = {
        "name": "dna",
        "sequences": [{"dnaSequence": {"sequence": "GATC", "count": 1}}],
    }

    features = featurize_protein_json(job)

    assert features["restype"].shape[0] == 4
    # torch STD_RESIDUES: DG=27, DA=26, DT=29, DC=28.
    assert features["restype"].argmax(-1).tolist() == [27, 26, 29, 28]
    # 23(DG)+22(DA)+21(DT)+21(DC) heavy atoms; OP3 dropped on residues 2-4.
    assert features["atom_to_token_idx"].shape[0] == 83
    assert features["ref_pos"].shape[0] == 83
    # one distogram representative atom per token (purine C4, pyrimidine C2).
    assert int(features["distogram_rep_atom_mask"].sum()) == 4
    assert features["residue_index"].tolist() == [1, 2, 3, 4]


def test_featurize_rna_sequence() -> None:
    job = {
        "name": "rna",
        "sequences": [{"rnaSequence": {"sequence": "GAUC", "count": 1}}],
    }

    features = featurize_protein_json(job)

    assert features["restype"].shape[0] == 4
    # torch STD_RESIDUES: G=22, A=21, U=24, C=23.
    assert features["restype"].argmax(-1).tolist() == [22, 21, 24, 23]
    assert features["atom_to_token_idx"].shape[0] == 86
    assert int(features["distogram_rep_atom_mask"].sum()) == 4


def test_featurize_rna_unpaired_msa_inline_updates_msa_profile_and_deletions() -> None:
    job = {
        "sequences": [
            {
                "rnaSequence": {
                    "sequence": "AGCUA",
                    "unpairedMsa": ">query\nAGCUA\n>hit\nAgu-TUA\n",
                }
            }
        ]
    }

    features = featurize_protein_json(job)

    np.testing.assert_array_equal(
        features["msa"],
        [
            [21, 22, 23, 24, 21],
            [21, 22, 23, 24, 21],
            [21, 31, 25, 24, 21],
        ],
    )
    np.testing.assert_allclose(features["profile"][1, [22, 31]], [0.5, 0.5])
    np.testing.assert_allclose(features["deletion_mean"], [0, 1, 0, 0, 0])
    assert features["has_deletion"][2, 1] == 1.0


def test_featurize_rna_unpaired_msa_path_matches_inline(tmp_path) -> None:
    a3m = ">query\nAGCUA\n>hit\nAgu-TUA\n"
    (tmp_path / "rna.a3m").write_text(a3m)
    inline = featurize_protein_json(
        {
            "sequences": [
                {"rnaSequence": {"sequence": "AGCUA", "unpairedMsa": a3m}}
            ]
        }
    )
    from_path = featurize_protein_json(
        {
            "sequences": [
                {
                    "rnaSequence": {
                        "sequence": "AGCUA",
                        "unpairedMsaPath": "rna.a3m",
                    }
                }
            ]
        },
        base_dir=tmp_path,
    )

    for name in ("msa", "has_deletion", "deletion_value", "profile", "deletion_mean"):
        np.testing.assert_array_equal(from_path[name], inline[name])


@pytest.mark.parametrize(
    ("a3m", "error"),
    [
        (">query\nAGCUA\n>bad\nAGC1A\n", "unsupported RNA MSA residue"),
        (">query\nAGCUA\n>bad\nAGCU\n", "aligned length"),
    ],
)
def test_featurize_rna_unpaired_msa_rejects_invalid_rows(
    a3m: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        featurize_protein_json(
            {
                "sequences": [
                    {"rnaSequence": {"sequence": "AGCUA", "unpairedMsa": a3m}}
                ]
            }
        )


def test_featurize_rejects_unsupported_nucleotide_base() -> None:
    with pytest.raises(ValueError, match="unsupported dnaSequence base"):
        featurize_protein_json(
            {"sequences": [{"dnaSequence": {"sequence": "AX", "count": 1}}]}
        )


def test_featurize_accepts_unknown_dna_and_rna_nucleotide() -> None:
    features = featurize_protein_json(
        {"sequences": [
            {"dnaSequence": {"sequence": "AN"}},
            {"rnaSequence": {"sequence": "UN"}},
        ]}
    )
    assert features["restype"].argmax(axis=-1).tolist() == [26, 30, 24, 25]
    assert features["token_ccd_code_chars"][1, :2].tolist() == [ord("D"), ord("N")]
    assert features["token_ccd_code_chars"][3, 0] == ord("N")


def test_runner_metadata_fields_are_accepted_but_not_features() -> None:
    features = featurize_protein_json(
        {
            "name": "metadata",
            "modelSeeds": [7, 11],
            "dialect": "alphafold3",
            "version": 1,
            "sequences": [{"proteinChain": {"sequence": "A"}}],
        }
    )
    for key in ("name", "modelSeeds", "dialect", "version"):
        assert key not in features
    with pytest.raises(ValueError, match="unsupported top-level"):
        featurize_protein_json(
            {"sequences": [{"proteinChain": {"sequence": "A"}}], "modelSeed": [1]}
        )


def test_deprecated_precomputed_msa_directory_is_supported(tmp_path) -> None:
    msa_dir = tmp_path / "legacy"
    msa_dir.mkdir()
    (msa_dir / "pairing.a3m").write_text(">q\nACDEF\n>p\nAC-EF\n")
    (msa_dir / "non_pairing.a3m").write_text(">q\nACDEF\n>u\n-CDEF\n")
    legacy = featurize_protein_json(
        {"sequences": [{"proteinChain": {
            "sequence": "ACDEF",
            "msa": {"precomputed_msa_dir": "legacy", "pairing_db": "uniref100"},
        }}]},
        base_dir=tmp_path,
    )
    modern = featurize_protein_json(
        {"sequences": [{"proteinChain": {
            "sequence": "ACDEF",
            "pairedMsaPath": str(msa_dir / "pairing.a3m"),
            "unpairedMsaPath": str(msa_dir / "non_pairing.a3m"),
        }}]}
    )
    np.testing.assert_array_equal(legacy["msa"], modern["msa"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="legacy MSA directory.*no pairing"):
        featurize_protein_json(
            {"sequences": [{"proteinChain": {
                "sequence": "ACDEF",
                "msa": {"precomputed_msa_dir": "empty"},
            }}]},
            base_dir=tmp_path,
        )


def test_featurize_json_accepts_smiles_ligand() -> None:
    job = {
        "name": "bad",
        "sequences": [{"ligand": {"ligand": "CCC=O", "count": 1}}],
    }

    features = featurize_protein_json(job)
    assert features["restype"].shape == (4, 32)
    assert features["chemical_bond_atom_indices"].shape == (3, 2)


def test_featurize_json_validates_supported_raw_features() -> None:
    with pytest.raises(ValueError, match="covalent_bonds"):
        featurize_protein_json(
            {
                "sequences": [{"proteinChain": {"sequence": "A"}}],
                "covalent_bonds": [{"entity1": "1"}],
            }
        )
    with pytest.raises(ValueError, match="unsupported residue"):
        featurize_protein_json({"sequences": [{"proteinChain": {"sequence": "AJ"}}]})


def test_selenomethionine_reaches_the_model_as_met(ccd_components) -> None:
    """MSE is read out of the released components.cif, so it needs that asset."""
    modified = featurize_protein_json(
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "A",
                        "modifications": [{"ptmType": "CCD_MSE", "ptmPosition": 1}],
                    },
                }
            ]
        }
    )
    # MSE is the exception among modifications: upstream's mse_to_met, citing
    # AlphaFold3 SI chapter 2.1, rewrites selenomethionine to MET (SE -> SD,
    # element S) and clears the hetero flag, so it reaches the model as a plain
    # standard residue rather than a modified one. The original identity is not
    # discarded -- it moves to token_reference_is_mse, which the atom-level
    # features read to keep the selenium provenance.
    assert modified["token_is_modified"].tolist() == [0]
    assert modified["token_reference_is_mse"].tolist() == [1]


def test_load_first_job_and_cli_write_static_npz(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    out_path = tmp_path / "features.npz"
    input_path.write_text(
        json.dumps(
            [
                {
                    "name": "toy",
                    "sequences": [{"proteinChain": {"sequence": "AG", "count": 1}}],
                }
            ]
        )
    )

    assert load_first_job(input_path)["name"] == "toy"
    main(["--input", str(input_path), "--out", str(out_path), "--n-queries", "2"])

    features = load_static_feature_npz(out_path)
    assert features["restype"].shape == (2, 32)
    assert features["pad_info"]["mask_trunked"].shape == features["v_lm"].shape[:-1]


def test_featurize_protein_ligand_ion_complex() -> None:
    job = {
        "name": "complex",
        "sequences": [
            {"proteinChain": {"sequence": "GACE", "count": 1}},
            {"ligand": {"ligand": "CCD_ATP", "count": 1}},
            {"ion": {"ion": "MG", "count": 2}},
        ],
    }

    features = featurize_protein_json(job)

    # protein GACE tokens + ATP 31 atoms + 2 Mg = 4 + 31 + 2 = 37 tokens.
    assert features["restype"].shape[0] == 37
    np.testing.assert_array_equal(features["token_index"], np.arange(37))
    n_atom = features["atom_to_token_idx"].shape[0]
    assert features["ref_pos"].shape[0] == n_atom
    assert features["ref_mask"].shape[0] == n_atom

    # ligand/ion atoms are one token each (tokatom_idx == 0, distogram rep == 1).
    lig_token_start = 4
    lig_atom_mask = features["atom_to_token_idx"] >= lig_token_start
    assert np.all(features["atom_to_tokatom_idx"][lig_atom_mask] == 0)
    assert np.all(features["distogram_rep_atom_mask"][lig_atom_mask] == 1.0)

    # ligand tokens are restype UNK (index 20).
    assert np.all(features["restype"][lig_token_start:].argmax(-1) == 20)

    # Torch RawMsa uses an X/UNK query for ligand tokens, not a gap placeholder.
    atp_token_stop = lig_token_start + 31
    np.testing.assert_array_equal(
        features["msa"][:, lig_token_start:atp_token_stop],
        np.full((features["msa"].shape[0], 31), 20),
    )
    expected_ligand_profile = np.zeros((31, 32), dtype=np.float32)
    expected_ligand_profile[:, 20] = 1.0
    np.testing.assert_array_equal(
        features["profile"][lig_token_start:atp_token_stop],
        expected_ligand_profile,
    )

    # Torch exposes CCD intra-ligand bonds as a symmetric token adjacency.
    token_bonds = features["token_bonds"]
    np.testing.assert_array_equal(token_bonds, token_bonds.T)
    assert np.count_nonzero(
        token_bonds[lig_token_start:atp_token_stop, lig_token_start:atp_token_stop]
    )
    assert not np.any(token_bonds[:lig_token_start])
    assert not np.any(token_bonds[atp_token_stop:])
    assert not np.any(np.diag(token_bonds))

    # three distinct entities and four chains (1 protein, 1 ligand, 2 ions).
    assert features["entity_id"].max() == 2
    assert features["asym_id"].max() == 3
    # two Mg ions share entity id but differ in sym id.
    assert features["sym_id"].max() == 1


def test_single_protein_pairing_and_nucleic_msa_match_torch() -> None:
    # No trailing newline after the paired hit: concatenation must insert one,
    # otherwise the following unpaired header is merged into that sequence.
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGCDE",
                    "pairedMsa": ">q\nAGCDE\n>paired\nAGCD-",
                    "unpairedMsa": ">q\nAGCDE\n>unpaired\n-GCDE\n",
                }
            },
            {"dnaSequence": {"sequence": "GA"}},
            {"rnaSequence": {"sequence": "UC"}},
        ]
    }

    features = featurize_protein_json(job)

    # Torch inference folds pairedMsa into the unpaired stack when there is only
    # one unique protein. Its all-seq query row remains first, followed by the
    # concatenated/deduplicated query, paired hit, and unpaired hit. DNA/RNA
    # query restypes are copied into every global MSA row rather than gap-filled.
    np.testing.assert_array_equal(
        features["msa"],
        [
            [0, 7, 4, 3, 6, 27, 26, 24, 23],
            [0, 7, 4, 3, 6, 27, 26, 24, 23],
            [0, 7, 4, 3, 31, 27, 26, 24, 23],
            [31, 7, 4, 3, 6, 27, 26, 24, 23],
        ],
    )


def test_featurize_ion_residue_index() -> None:
    job = {
        "name": "ion",
        "sequences": [{"ion": {"ion": "MG", "count": 1}}],
    }

    features = featurize_protein_json(job)
    assert features["restype"].shape[0] == 1
    assert features["residue_index"][0] == 1
    assert features["ref_charge"][0] == 2.0


def test_relative_position_features_are_exactly_zero_or_one_int8() -> None:
    from foldjax.models.protenix.data.featurize_json import (
        _relative_position_features,
    )

    rng = np.random.default_rng(0)
    n = 24
    relp = _relative_position_features(
        asym_id=rng.integers(0, 3, n),
        residue_index=rng.integers(0, 40, n),
        entity_id=rng.integers(0, 2, n),
        sym_id=rng.integers(0, 4, n),
        token_index=rng.integers(0, 40, n),
    )

    assert relp.dtype == np.int8
    assert relp.shape == (n, n, 139)
    assert set(np.unique(relp).tolist()) <= {0, 1}
    np.testing.assert_array_equal(relp[..., :66].sum(-1), np.ones((n, n), np.int8))
    np.testing.assert_array_equal(
        relp[..., 66:132].sum(-1), np.ones((n, n), np.int8)
    )
    np.testing.assert_array_equal(relp[..., 133:].sum(-1), np.ones((n, n), np.int8))
