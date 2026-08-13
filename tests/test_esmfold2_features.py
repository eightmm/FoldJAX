"""The general featurizer against upstream's own, and past where it stops.

`prepare_protein_features` is upstream's builder for the case it covers -- one
protein chain, no alignment -- so for that case the two must agree tensor for
tensor. Everything past it (several chains, symmetry, an MSA) has no upstream
reference, so those tests state the rule each field follows instead.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.esmfold2")

from transformers.models.esmfold2.protein_utils import (  # noqa: E402
    MSA_GAP_TOKEN_ID,
    prepare_protein_features,
)

from foldjax.backends._esmfold2_features import build_features  # noqa: E402

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def test_one_chain_matches_upstreams_own_builder() -> None:
    """Bit-for-bit, every key: the shared case cannot drift."""
    expected = prepare_protein_features(UBIQUITIN)
    got = build_features([(UBIQUITIN, "A", 0, 0)])

    assert set(got) == set(expected)
    for key, reference in expected.items():
        assert got[key].shape == reference.shape, key
        assert got[key].dtype == reference.dtype, key
        assert torch.equal(got[key], reference), key


def test_a_homodimer_shares_an_entity_and_counts_copies() -> None:
    """One sequence, two chains: same entity_id, different asym_id and sym_id.

    That distinction is the whole of what tells the model a homodimer from a
    heterodimer of identical length, so it is asserted rather than assumed.
    """
    got = build_features([(UBIQUITIN, "A", 0, 0), (UBIQUITIN, "B", 0, 1)])
    n = len(UBIQUITIN)

    assert got["res_type"].shape == (1, 2 * n)
    assert torch.equal(got["entity_id"][0].unique(), torch.tensor([1]))
    assert torch.equal(got["asym_id"][0].unique(), torch.tensor([0, 1]))
    assert torch.equal(got["sym_id"][0, :n].unique(), torch.tensor([0]))
    assert torch.equal(got["sym_id"][0, n:].unique(), torch.tensor([1]))
    # Residue numbering restarts per chain; it is a position in a polymer.
    assert got["residue_index"][0, n].item() == 0


def test_two_entities_are_numbered_apart() -> None:
    got = build_features([(UBIQUITIN, "A", 0, 0), (UBIQUITIN[:20], "B", 1, 0)])
    assert torch.equal(got["entity_id"][0].unique(), torch.tensor([1, 2]))


def test_an_alignment_lands_in_its_own_chains_columns(tmp_path) -> None:
    """Unpaired assembly: a row belongs to one chain and gaps the others.

    Pairing rows across chains would claim the alignments were searched
    together; a per-chain a3m cannot support that claim.
    """
    short = UBIQUITIN[:10]
    a3m = tmp_path / "chain.a3m"
    a3m.write_text(f">query\n{short}\n>hit\n{'A' * 10}\n")

    got = build_features(
        [(short, "A", 0, 0), (short, "B", 1, 0)], {0: a3m}
    )
    msa = got["msa"][0]

    assert msa.shape[0] == 2, "query row plus the one hit"
    assert torch.equal(msa[0], got["res_type"][0]), "row 0 is the query itself"
    assert (msa[1, 10:] == MSA_GAP_TOKEN_ID).all(), "the other chain is gapped"
    assert (msa[1, :10] != MSA_GAP_TOKEN_ID).all()


def test_insertions_are_counted_against_the_column_that_follows(tmp_path) -> None:
    """Lowercase a3m columns are deletions, and they are what the model reads."""
    short = UBIQUITIN[:10]
    a3m = tmp_path / "chain.a3m"
    a3m.write_text(f">query\n{short}\n>hit\nAAAAAaaAAAAA\n")

    got = build_features([(short, "A", 0, 0)], {0: a3m})

    assert got["deletion_value"][0, 1, 5].item() == 2.0
    assert bool(got["has_deletion"][0, 1, 5])
    assert got["deletion_value"][0, 1, 4].item() == 0.0


def test_a_row_of_the_wrong_length_is_dropped(tmp_path) -> None:
    """It is not an alignment to this sequence, and padding it would invent one."""
    short = UBIQUITIN[:10]
    a3m = tmp_path / "chain.a3m"
    a3m.write_text(f">query\n{short}\n>ragged\nAAA\n>good\n{'C' * 10}\n")

    msa = build_features([(short, "A", 0, 0)], {0: a3m})["msa"][0]
    assert msa.shape[0] == 2
