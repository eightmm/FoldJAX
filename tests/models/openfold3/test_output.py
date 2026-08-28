"""Structure and score output, checked on real depositions.

Decoding atom identity out of the features is where this can go quietly wrong: an
off-by-one in the element table or the wrong character offset produces a file that
opens fine and describes the wrong molecule. So the decoded names, elements and
residues are compared against the deposition itself, and the embedded tables are
compared against their authorities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.openfold3.output import (
    ELEMENT_SYMBOLS,
    RESIDUE_NAMES,
    UNKNOWN_ELEMENT_BIN,
    _has_clash_blocked,
    atom_metadata,
    confidence_summary,
    write_prediction_outputs,
)

from .real_targets import featurized, requires_real_targets

pytestmark = pytest.mark.torch_parity


def test_residue_vocabulary_matches_upstream(openfold3_source: Path) -> None:
    """``restype`` is a one-hot over this exact list, in this exact order."""
    from openfold3.core.data.resources.residues import STANDARD_RESIDUES_WITH_GAP_3

    assert list(RESIDUE_NAMES) == list(STANDARD_RESIDUES_WITH_GAP_3)


def test_element_table_matches_the_periodic_table(openfold3_source: Path) -> None:
    """Bin *k* of ``ref_element`` means atomic number *k + 1*.

    Upstream builds it as ``GetAtomicNumber(symbol) - 1``, so an off-by-one here
    would relabel every atom in every output file.
    """
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    assert len(ELEMENT_SYMBOLS) == UNKNOWN_ELEMENT_BIN
    for index, symbol in enumerate(ELEMENT_SYMBOLS):
        assert table.GetElementSymbol(index + 1) == symbol, index


@pytest.mark.parametrize("pdb_id", ["1UBQ", "1BNA", "1EHZ"])
@requires_real_targets
def test_decoded_atoms_describe_the_deposition(pdb_id: str) -> None:
    """Protein, DNA and RNA, because each uses a different residue vocabulary
    range and a different set of atom names."""
    from tests._foldbench import load_target

    features = featurized(pdb_id, include_ligands=False)
    metadata = atom_metadata(features)
    target = load_target(pdb_id)

    assert metadata.name.size == int(np.asarray(features["atom_mask"]).sum())
    # Every element must have decoded to something real.
    assert "X" not in set(metadata.element.tolist())
    assert set(metadata.element.tolist()) <= set(ELEMENT_SYMBOLS)
    # Residue names must come from the vocabulary and match the molecule type.
    assert set(metadata.residue_name.tolist()) <= set(RESIDUE_NAMES)
    if target.molecule_types == ("protein",):
        assert "CA" in set(metadata.name.tolist())
    if target.molecule_types == ("dna",):
        assert {"DA", "DC", "DG", "DT"} & set(metadata.residue_name.tolist())
    if target.molecule_types == ("rna",):
        assert {"A", "C", "G", "U"} & set(metadata.residue_name.tolist())
    # One chain label per deposited chain.
    expected_chains = sum(len(p.chain_ids) for p in target.polymers)
    assert len(set(metadata.chain_id.tolist())) == expected_chains


@requires_real_targets
def test_residue_numbering_follows_the_features() -> None:
    """The features already carry the deposition's 1-based numbering.

    Adding one "to convert to mmCIF" shifted every residue; a hand-built batch that
    numbered from zero is what made that look right.
    """
    features = featurized("1UBQ", include_ligands=False)
    metadata = atom_metadata(features)
    residue_index = np.asarray(features["residue_index"])[0]
    atom_to_token = np.asarray(features["atom_to_token_index"])[0]
    keep = np.asarray(features["atom_mask"])[0] > 0
    np.testing.assert_array_equal(
        metadata.residue_id, residue_index[atom_to_token[keep]]
    )
    assert sorted(set(int(x) for x in metadata.residue_id)) == list(
        range(1, len(residue_index) + 1)
    )


@requires_real_targets
def test_chains_are_labelled_from_a() -> None:
    """``asym_id`` is 1-based, so labelling by the id itself would start at B."""
    metadata = atom_metadata(featurized("1UBQ", include_ligands=False))
    assert set(metadata.chain_id.tolist()) == {"A"}

    duplex = atom_metadata(featurized("1BNA", include_ligands=False))
    assert set(duplex.chain_id.tolist()) == {"A", "B"}


@requires_real_targets
def test_padding_is_excluded() -> None:
    """A prediction covers padded atoms; the file must not."""
    from foldjax.models.openfold3.data import pad_features

    features = featurized("1UBQ", include_ligands=False)
    real = int(np.asarray(features["atom_mask"]).sum())
    padded = pad_features(
        features,
        n_token=features["token_mask"].shape[-1] + 10,
        n_atom=features["atom_mask"].shape[-1] + 100,
    )
    assert atom_metadata(padded).name.size == real


def _prediction(n_atom: int, n_samples: int = 3, with_iptm: bool = True):
    from foldjax.models.openfold3.inference import Prediction

    generator = np.random.default_rng(0)
    return Prediction(
        coordinates=generator.normal(size=(n_samples, n_atom, 3)) * 10.0,
        plddt=generator.uniform(0.4, 0.95, size=(n_samples, n_atom)),
        ptm=np.linspace(0.5, 0.8, n_samples),
        iptm=np.linspace(0.3, 0.9, n_samples) if with_iptm else None,
        chain_pair_iptm=None,
        pae_logits=np.zeros((n_samples, 4, 4, 8)),
        pde_logits=np.zeros((n_samples, 4, 4, 8)),
        distogram_logits=np.zeros((1, 4, 4, 8)),
    )


@requires_real_targets
def test_written_structure_round_trips(tmp_path: Path) -> None:
    """The file must parse back with the same atoms in the same places."""
    gemmi = pytest.importorskip("gemmi")

    features = featurized("1UBQ", include_ligands=False)
    metadata = atom_metadata(features)
    prediction = _prediction(features["atom_mask"].shape[-1])

    with pytest.warns(RuntimeWarning, match="no exact output metadata"):
        written = write_prediction_outputs(prediction, features, tmp_path, name="ubq")
    assert len(written["structures"]) == 3
    assert written["num_atoms"] == metadata.name.size

    structure = gemmi.read_structure(str(written["structures"][0]))
    atoms = [
        (atom.name, atom.element.name, residue.name, residue.seqid.num, chain.name)
        for model in structure
        for chain in model
        for residue in chain
        for atom in residue
    ]
    assert len(atoms) == metadata.name.size
    names, elements, residues, numbers, chains = zip(*atoms, strict=True)
    assert list(names) == metadata.name.tolist()
    assert list(residues) == metadata.residue_name.tolist()
    assert list(numbers) == metadata.residue_id.tolist()
    assert list(chains) == metadata.chain_id.tolist()

    # Coordinates survive, to file precision.
    positions = np.asarray(
        [
            [atom.pos.x, atom.pos.y, atom.pos.z]
            for model in structure
            for chain in model
            for residue in chain
            for atom in residue
        ]
    )
    np.testing.assert_allclose(
        positions,
        np.asarray(prediction.coordinates)[0][metadata.keep],
        atol=1e-3,
    )


@requires_real_targets
def test_b_factors_carry_plddt(tmp_path: Path) -> None:
    gemmi = pytest.importorskip("gemmi")
    features = featurized("1UBQ", include_ligands=False)
    metadata = atom_metadata(features)
    prediction = _prediction(features["atom_mask"].shape[-1])
    with pytest.warns(RuntimeWarning, match="no exact output metadata"):
        written = write_prediction_outputs(prediction, features, tmp_path, name="ubq")

    structure = gemmi.read_structure(str(written["structures"][0]))
    b_factors = np.asarray(
        [
            atom.b_iso
            for model in structure
            for chain in model
            for residue in chain
            for atom in residue
        ]
    )
    np.testing.assert_allclose(
        b_factors,
        np.asarray(prediction.plddt)[0][metadata.keep] * 100.0,
        atol=1e-2,
    )


def _ranking_features(n_atom: int, *, has_protein: bool) -> dict[str, np.ndarray]:
    split = n_atom // 2
    return {
        "atom_mask": np.ones((1, n_atom), dtype=np.float32),
        "atom_to_token_index": np.arange(n_atom, dtype=np.int32)[None],
        "token_mask": np.ones((1, n_atom), dtype=np.float32),
        "asym_id": np.asarray(
            [[0] * split + [1] * (n_atom - split)], dtype=np.int32
        ),
        "is_protein": np.full((1, n_atom), has_protein, dtype=np.int32),
        "is_rna": np.zeros((1, n_atom), dtype=np.int32),
        "is_dna": np.full((1, n_atom), not has_protein, dtype=np.int32),
    }


def test_nonprotein_scores_use_the_exact_upstream_ranking_formula() -> None:
    prediction = _prediction(64, n_samples=3, with_iptm=True)
    summary = confidence_summary(
        prediction, _ranking_features(64, has_protein=False)
    )
    assert summary["num_samples"] == 3
    scores = [entry["sample_ranking_score"] for entry in summary["samples"]]
    for entry, score in zip(summary["samples"], scores, strict=True):
        assert entry["has_clash"] == 0.0
        assert score == pytest.approx(0.8 * entry["iptm"] + 0.2 * entry["ptm"])
    assert summary["ranked_samples"] == [
        index for index, _ in sorted(enumerate(scores), key=lambda p: -p[1])
    ]
    assert summary["samples"][0]["mean_plddt"] > 1.0


def test_protein_score_is_clearly_partial_and_never_promoted_to_ranking() -> None:
    prediction = _prediction(64, n_samples=3, with_iptm=True)
    summary = confidence_summary(prediction, _ranking_features(64, has_protein=True))
    assert "ranked_samples" not in summary
    for entry in summary["samples"]:
        assert "sample_ranking_score" not in entry
        assert entry["sample_ranking_score_no_disorder"] == pytest.approx(
            0.8 * entry["iptm"]
            + 0.2 * entry["ptm"]
            - 100.0 * entry["has_clash"]
        )


def test_scores_do_not_fall_back_to_plddt_without_exact_ranking_inputs() -> None:
    prediction = _prediction(64, n_samples=3, with_iptm=False)
    summary = confidence_summary(prediction)
    assert "ranked_samples" not in summary
    assert all("sample_ranking_score" not in s for s in summary["samples"])


def test_nonfinite_coordinates_prevent_an_exact_ranking_claim() -> None:
    prediction = _prediction(8, n_samples=2, with_iptm=True)
    coordinates = np.asarray(prediction.coordinates).copy()
    coordinates[1, 0, 0] = np.nan
    summary = confidence_summary(
        prediction._replace(coordinates=coordinates),
        _ranking_features(8, has_protein=False),
    )
    assert summary["samples"][1]["has_clash"] is None
    assert "sample_ranking_score" not in summary["samples"][1]
    assert "ranked_samples" not in summary


def test_blocked_clash_veto_matches_the_upstream_score_formula() -> None:
    prediction = _prediction(8, n_samples=2, with_iptm=True)
    first = np.arange(4, dtype=np.float32)[:, None] * np.asarray([[3.0, 0.0, 0.0]])
    clashing = np.concatenate((first, first + 0.1), axis=0)
    clean = np.concatenate((first, first + 100.0), axis=0)
    prediction = prediction._replace(coordinates=np.stack((clashing, clean)))

    summary = confidence_summary(
        prediction, _ranking_features(8, has_protein=False)
    )

    assert [entry["has_clash"] for entry in summary["samples"]] == [1.0, 0.0]
    assert summary["samples"][0]["sample_ranking_score"] == pytest.approx(
        0.8 * summary["samples"][0]["iptm"]
        + 0.2 * summary["samples"][0]["ptm"]
        - 100.0
    )
    assert summary["ranked_samples"][0] == 1


def test_blocked_clash_matches_the_jax_metric_across_distance_tiles() -> None:
    import jax.numpy as jnp

    from foldjax.models.openfold3.models.clash import compute_has_clash

    first = np.arange(4, dtype=np.float32)[:, None] * np.asarray([[3.0, 0.0, 0.0]])
    positions = np.stack(
        (
            np.concatenate((first, first + 0.1)),
            np.concatenate((first, first + 100.0)),
        )
    )
    asym_id = np.asarray([0] * 4 + [1] * 4, dtype=np.int32)
    atom_mask = np.ones(8, dtype=bool)
    is_polymer = np.ones(8, dtype=bool)

    expected = compute_has_clash(
        jnp.asarray(asym_id),
        jnp.asarray(positions),
        jnp.asarray(atom_mask),
        jnp.asarray(is_polymer),
        n_chain=2,
    )
    actual = _has_clash_blocked(
        positions,
        atom_mask,
        asym_id,
        is_polymer,
        block_size=2,
    )

    np.testing.assert_array_equal(actual, np.asarray(expected))


def test_missing_features_are_named() -> None:
    with pytest.raises(KeyError, match="ref_element"):
        atom_metadata({"ref_atom_name_chars": np.zeros((2, 4, 64))})


@requires_real_targets
def test_a_wrong_encoding_width_is_refused() -> None:
    """Decoding a 128-bin element vector against a 119-bin table would silently
    relabel atoms."""
    features = dict(featurized("1UBQ", include_ligands=False))
    features["ref_element"] = np.zeros((1, features["ref_element"].shape[1], 128))
    with pytest.raises(ValueError, match="119 bins"):
        atom_metadata(features)


def _fake_prediction(n_token: int, n_bin: int, num_samples: int = 5):
    """A prediction whose pair logits are the only large thing in it."""
    from foldjax.models.openfold3.inference import Prediction

    return Prediction(
        coordinates=np.zeros((num_samples, n_token * 8, 3), dtype=np.float32),
        plddt=np.zeros((num_samples, n_token * 8), dtype=np.float32),
        ptm=np.zeros((num_samples,), dtype=np.float32),
        iptm=None,
        chain_pair_iptm=None,
        pae_logits=np.zeros((num_samples, n_token, n_token, n_bin), dtype=np.float32),
        pde_logits=np.zeros((num_samples, n_token, n_token, n_bin), dtype=np.float32),
        distogram_logits=np.zeros((1, n_token, n_token, n_bin), dtype=np.float32),
        experimentally_resolved_logits=None,
    )


def test_write_arrays_keeps_everything_when_it_fits(tmp_path) -> None:
    from foldjax.models.openfold3.output import write_arrays

    prediction = _fake_prediction(n_token=8, n_bin=4)
    path, omitted = write_arrays(prediction, tmp_path / "raw.npz")
    assert omitted == ()
    stored = np.load(path)
    assert "pae_logits" in stored and "distogram_logits" in stored


def test_write_arrays_drops_pair_logits_over_the_budget(tmp_path) -> None:
    """The logits are quadratic in tokens with a bin axis on top.

    At 2076 tokens PAE and PDE are 5.14 GiB each, so writing them turns a
    prediction into a file nobody asked for. What must never be dropped is anything
    the reported scores come from.
    """
    from foldjax.models.openfold3.output import write_arrays

    prediction = _fake_prediction(n_token=64, n_bin=64)
    # Small enough to force dropping, large enough to keep the rest.
    path, omitted = write_arrays(prediction, tmp_path / "raw.npz", max_bytes=1 << 20)
    assert set(omitted) == {"pae_logits", "pde_logits", "distogram_logits"}
    stored = np.load(path)
    assert set(stored.files) == {"coordinates", "plddt", "ptm"}


def test_write_arrays_drops_largest_first(tmp_path) -> None:
    """Only as much is dropped as the budget requires, biggest first.

    Dropping in declaration order would throw away the distogram -- a fifth the
    size of PAE, because it has no sample axis -- while leaving PAE in place.
    """
    from foldjax.models.openfold3.output import write_arrays

    prediction = _fake_prediction(n_token=64, n_bin=64, num_samples=5)
    pae_bytes = prediction.pae_logits.nbytes
    everything = sum(
        value.nbytes for value in prediction._asdict().values() if value is not None
    )
    # A budget that only one of the two big ones has to go to satisfy.
    path, omitted = write_arrays(
        prediction, tmp_path / "raw.npz", max_bytes=everything - pae_bytes
    )
    assert len(omitted) == 1
    stored = np.load(path)
    assert "distogram_logits" in stored.files


def test_write_arrays_can_be_told_to_write_everything(tmp_path) -> None:
    from foldjax.models.openfold3.output import write_arrays

    prediction = _fake_prediction(n_token=16, n_bin=8)
    path, omitted = write_arrays(prediction, tmp_path / "raw.npz", max_bytes=None)
    assert omitted == ()
    assert "pae_logits" in np.load(path).files
