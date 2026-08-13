"""The NumPy featuriser and the PDB writer, with no torch in sight.

`tests/test_esmfold2_features.py` already checks this builder against
upstream's own -- through the thin torch wrapper -- so what is left to state
here is what a torch-free environment can still check: the shapes and dtypes
the model's contract needs, and that the writer round-trips the atoms it was
handed.
"""

from __future__ import annotations

import numpy as np

from foldjax.models.esmfold2.data import chemistry, features, pdb

PEPTIDE = "ACDEFGHIK"


def test_the_feature_dictionary_has_the_shapes_the_model_reads() -> None:
    built = features.build_features([(PEPTIDE, "A", 0, 0)])
    n_tokens = len(PEPTIDE)

    assert built["res_type"].shape == (1, n_tokens)
    assert built["token_bonds"].shape == (1, n_tokens, n_tokens, 1)
    assert built["ref_pos"].shape[2] == 3
    # The atom axis is padded to a multiple of 32, and the padding is masked.
    assert built["ref_pos"].shape[1] % features.ATOM_BLOCK == 0
    assert built["atom_attention_mask"].dtype == np.bool_
    assert built["ref_charge"].dtype == np.int8
    assert built["res_type"].dtype == np.int64
    assert built["ref_pos"].dtype == np.float32
    # Every real atom is masked in, every padded one out.
    valid = int(built["atom_attention_mask"].sum())
    assert valid < built["ref_pos"].shape[1]
    assert np.all(built["atom_attention_mask"][0, :valid])


def test_the_representative_atom_is_cb_where_there_is_one() -> None:
    """The distogram and every distance-based confidence read this atom."""
    built = features.build_features([("AG", "A", 0, 0)])
    names = built["ref_atom_name_chars"][0][built["distogram_atom_idx"][0]]
    decoded = [pdb.decode_atom_name(row) for row in names]
    # Alanine has a CB; glycine does not and falls back to CA.
    assert decoded == ["CB", "CA"]


def test_two_chains_keep_their_identity() -> None:
    built = features.build_features([(PEPTIDE, "A", 0, 0), (PEPTIDE, "B", 0, 1)])
    n = len(PEPTIDE)
    assert np.array_equal(np.unique(built["entity_id"][0]), [1])
    assert np.array_equal(np.unique(built["asym_id"][0]), [0, 1])
    assert built["residue_index"][0, n] == 0


def test_an_alignment_becomes_rows_over_its_own_columns(tmp_path) -> None:
    a3m = tmp_path / "chain.a3m"
    a3m.write_text(f">query\n{PEPTIDE}\n>hit\nAADEFGHIK\n")
    built = features.build_features(
        [(PEPTIDE, "A", 0, 0), (PEPTIDE, "B", 1, 0)], {0: a3m}
    )
    n = len(PEPTIDE)
    assert built["msa"].shape == (1, 2, 2 * n)
    # The hit covers chain A's columns and is a gap over chain B's.
    assert np.all(built["msa"][0, 1, n:] == chemistry.MSA_GAP_TOKEN_ID)
    assert not np.all(built["msa"][0, 1, :n] == chemistry.MSA_GAP_TOKEN_ID)


def test_the_pdb_writer_emits_every_folded_atom() -> None:
    built = features.build_features([(PEPTIDE, "A", 0, 0), (PEPTIDE, "B", 0, 1)])
    n_atoms = built["ref_pos"].shape[1]
    coords = np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3) / 10
    plddt = np.full(n_atoms, 0.87, dtype=np.float32)

    text = pdb.to_pdb(coords, built, plddt)
    atoms = [line for line in text.splitlines() if line.startswith("ATOM")]
    assert len(atoms) == int(built["atom_attention_mask"].sum())
    # Both chains are named, which upstream's OpenFold writer cannot do.
    assert {line[21] for line in atoms} == {"A", "B"}
    # pLDDT reaches the b-factor column on the 0-100 scale viewers assume.
    assert atoms[0][60:66].strip() == "87.00"
    assert text.rstrip().endswith("END")


def test_several_samples_become_numbered_models() -> None:
    built = features.build_features([(PEPTIDE, "A", 0, 0)])
    n_atoms = built["ref_pos"].shape[1]
    coords = np.zeros((3, n_atoms, 3), dtype=np.float32)
    text = pdb.to_pdb_models(coords, built)
    assert text.count("MODEL ") == 3
    assert text.count("ENDMDL") == 3
    assert text.count("\nEND\n") == 1
