"""The NumPy featuriser and the PDB writer, with no torch in sight.

`tests/test_esmfold2_features.py` already checks this builder against
upstream's own -- through the thin torch wrapper -- so what is left to state
here is what a torch-free environment can still check: the shapes and dtypes
the model's contract needs, and that the writer round-trips the atoms it was
handed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def test_streamed_a3m_parser_matches_the_historical_literal_split(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = (
        "ignored preface\r\n"
        ">query\r\nACDE\r\n"
        ">hit-one\r\nAaC-DE\r\n"
        ">hit-two\r\nWYf-E\r\n"
        ">bad-length\r\nAC\r\n"
        ">\r\n"
    )
    a3m = tmp_path / "chain.a3m"
    a3m.write_text(content, encoding="utf-8")
    normalized = a3m.read_text(encoding="utf-8")

    def historical() -> list[tuple[str, list[int]]]:
        rows: list[tuple[str, list[int]]] = []
        for block in normalized.split(">")[1:]:
            if not block.strip():
                continue
            body = "".join(block.split("\n")[1:])
            aligned: list[str] = []
            counts: list[int] = []
            pending = 0
            for character in body:
                if character.islower():
                    pending += 1
                    continue
                aligned.append(character.upper() if character != "-" else "-")
                counts.append(pending)
                pending = 0
            if len(aligned) != 4:
                continue
            sequence = "".join(aligned)
            if sequence == "ACDE" and not any(counts):
                continue
            rows.append((sequence, counts))
        return rows

    monkeypatch.setattr(features, "_A3M_READ_CHUNK_CHARS", 3)

    assert features.read_a3m(a3m, query="ACDE") == historical()


def test_msa_depth_does_not_materialise_the_complete_alignment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a3m = tmp_path / "chain.a3m"
    a3m.write_text(
        f">query\n{PEPTIDE}\n"
        + "".join(f">hit-{index}\nAADEFGHIK\n" for index in range(10_000)),
        encoding="utf-8",
    )
    original_read_text = Path.read_text

    def reject_full_alignment_read(path: Path, *args, **kwargs) -> str:
        if path == a3m:
            raise AssertionError("bounded MSA featurization must stream the alignment")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_full_alignment_read)

    built = features.build_features(
        [(PEPTIDE, "A", 0, 0)],
        {0: a3m},
        msa_depth=2,
    )

    assert built["msa"].shape == (1, 2, len(PEPTIDE))


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


def test_the_mmcif_route_parses_and_keeps_the_b_factors() -> None:
    """What the backend actually writes -- `foldjax.output` names files `.cif`."""
    gemmi = pytest.importorskip("gemmi")
    built = features.build_features([(PEPTIDE, "A", 0, 0), (PEPTIDE, "B", 0, 1)])
    n_atoms = built["ref_pos"].shape[1]
    coords = np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3) / 10
    plddt = np.full(n_atoms, 0.87, dtype=np.float32)

    text = pdb.to_mmcif(coords, built, plddt, name="job_sample_0")
    structure = gemmi.read_structure_string(text)
    atoms = [atom for chain in structure[0] for res in chain for atom in res]
    assert len(atoms) == int(built["atom_attention_mask"].sum())
    assert {chain.name for chain in structure[0]} == {"A", "B"}
    assert atoms[0].b_iso == pytest.approx(87.0, abs=0.01)


def test_build_msa_scales_deletions_counts_dots_keeps_homologs_and_pairs_taxa(
    tmp_path,
) -> None:
    a3m_a = tmp_path / "a.a3m"
    a3m_a.write_text(
        ">query\nACDE\n"
        ">hit_dot_and_lower key=9606\nA.aC-E\n"
        ">duplicate_query_homolog\nACDE\n"
        ">unpaired_a key=562\nWCDE\n",
        encoding="utf-8",
    )
    a3m_b = tmp_path / "b.a3m"
    a3m_b.write_text(
        ">query\nFGHI\n"
        ">paired_b key=9606\nF..GHI\n"
        ">unpaired_b key=7227\nYGHI\n",
        encoding="utf-8",
    )

    parsed_a = features.read_a3m(a3m_a, query="ACDE")
    assert parsed_a == [
        ("AC-E", [0, 2, 0, 0]),
        ("ACDE", [0, 0, 0, 0]),
        ("WCDE", [0, 0, 0, 0]),
    ]

    built = features.build_features(
        [("ACDE", "A", 0, 0), ("FGHI", "B", 1, 0)],
        {0: a3m_a, 1: a3m_b},
    )
    expected_scale_2 = float(np.arctan(2.0 / 3.0) * (np.pi / 2.0))
    assert built["msa"].shape == (1, 4, 8)
    # Row 1 pairs key=9606 from chain A ("AC-E") and chain B ("FGHI") together.
    assert built["msa"][0, 1, 2] == features.chemistry.MSA_GAP_TOKEN_ID
    assert np.all(built["msa"][0, 1, 4:] == built["res_type"][0, 4:])
    assert built["has_deletion"][0, 1, 1]
    assert built["has_deletion"][0, 1, 5]
    assert built["deletion_value"][0, 1, 1] == pytest.approx(
        expected_scale_2, abs=1e-6
    )
    assert built["deletion_value"][0, 1, 5] == pytest.approx(
        expected_scale_2, abs=1e-6
    )
    # Duplicate query homolog from chain A is retained on row 2 alongside unpaired B.
    np.testing.assert_array_equal(built["msa"][0, 2, :4], built["res_type"][0, :4])

    # Edge cases: empty A3M, msa_depth=0, msa_depth=1, exact limit msa_depth=2.
    empty_a3m = tmp_path / "empty.a3m"
    empty_a3m.write_text("", encoding="utf-8")
    empty_built = features.build_features([("ACDE", "A", 0, 0)], {0: empty_a3m})
    assert empty_built["msa"].shape == (1, 1, 4)
    assert not np.any(empty_built["has_deletion"])

    for zero_or_one in (0, 1):
        capped = features.build_features(
            [("ACDE", "A", 0, 0)], {0: a3m_a}, msa_depth=zero_or_one
        )
        assert capped["msa"].shape == (1, 1, 4)
        assert not np.any(capped["has_deletion"])

    at_limit = features.build_features([("ACDE", "A", 0, 0)], {0: a3m_a}, msa_depth=2)
    assert at_limit["msa"].shape == (1, 2, 4)
    assert at_limit["deletion_value"][0, 1, 1] == pytest.approx(
        expected_scale_2, abs=1e-6
    )

    # Single-chain repeated taxonomy IDs across the cap boundary: pairing ranks
    # multi-member taxa (key=2: AW, WW) ahead of singleton hits (key=1: WC), so
    # capping at depth=2 must match capping the complete assembled MSA (AW, not WC).
    single_taxa_a3m = tmp_path / "single_taxa.a3m"
    single_taxa_a3m.write_text(
        ">query\nAC\n"
        ">singleton key=1\nWC\n"
        ">pair_first key=2\nAW\n"
        ">pair_second key=2\nWW\n",
        encoding="utf-8",
    )
    uncapped_taxa = features.build_features(
        [("AC", "A", 0, 0)], {0: single_taxa_a3m}, msa_depth=None
    )
    capped_taxa = features.build_features(
        [("AC", "A", 0, 0)], {0: single_taxa_a3m}, msa_depth=2
    )
    np.testing.assert_array_equal(capped_taxa["msa"], uncapped_taxa["msa"][:, :2])
    expected_aw = np.asarray(
        [
            features.chemistry.PROTEIN_RESIDUE_TO_RES_TYPE["ALA"],
            features.chemistry.PROTEIN_RESIDUE_TO_RES_TYPE["TRP"],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(capped_taxa["msa"][0, 1], expected_aw)

    for bad_depth in (-1, True, "2"):
        with pytest.raises(ValueError, match="msa_depth"):
            features.build_features(
                [("ACDE", "A", 0, 0)], {0: a3m_a}, msa_depth=bad_depth  # type: ignore[arg-type]
            )
