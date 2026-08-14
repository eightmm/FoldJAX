"""Featurization from a query specification.

This is the layer that was missing: the model reads 34 features and nothing in the
package produced them, so the port could not fold anything from a sequence.

Featurization delegates to upstream's ``InferenceDataset``, so these tests are not
checking chemistry -- upstream owns that. They check the contract this package adds:
that every feature the model reads is produced, that ``max_atom_per_token_mask``
(which the dataset does not emit) is right, that padding preserves the masks, and
that the result actually drives ``predict``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.openfold3.data import (
    MODEL_FEATURES,
    featurize_query,
    pad_features,
    save_features,
)

pytestmark = pytest.mark.torch_parity

UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


def _spec(**chain_extra) -> dict:
    return {
        "queries": {
            "ubq": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": UBIQUITIN,
                        **chain_extra,
                    }
                ]
            }
        }
    }


@pytest.fixture(scope="module")
def features(openfold3_source: Path) -> dict:
    return featurize_query(_spec())


def test_every_feature_the_model_reads_is_produced(features: dict) -> None:
    missing = [name for name in MODEL_FEATURES if name not in features]
    assert not missing, f"predict reads these but featurization omits them: {missing}"


def test_shapes_are_consistent(features: dict) -> None:
    n_token = features["token_mask"].shape[-1]
    n_atom = features["atom_mask"].shape[-1]
    assert n_token == len(UBIQUITIN)
    assert features["restype"].shape == (1, n_token, 32)
    assert features["ref_pos"].shape == (1, n_atom, 3)
    assert features["token_bonds"].shape == (1, n_token, n_token)
    assert features["template_distogram"].shape[-3:] == (n_token, n_token, 39)
    # Atom-to-token indices must address real tokens.
    assert int(features["atom_to_token_index"].max()) == n_token - 1
    # And the per-token atom counts must add up to the atom count.
    assert int(features["num_atoms_per_token"].sum()) == n_atom


def test_max_atom_per_token_mask_counts_the_real_atoms(features: dict) -> None:
    """This one is derived here, not by upstream, so it needs its own check.

    It is a padded ``N_token x max_atoms_per_token`` layout, so its total must equal
    the number of real atoms, and its length must be the padded size rather than
    the atom count.
    """
    mask = features["max_atom_per_token_mask"]
    n_token = features["token_mask"].shape[-1]
    assert mask.shape == (1, n_token * 23)
    assert float(mask.sum()) == float(features["atom_mask"].sum())

    # Slot k of token t is set only when the token has more than k atoms.
    counts = features["num_atoms_per_token"][0]
    blocks = mask[0].reshape(n_token, 23)
    np.testing.assert_array_equal(blocks.sum(axis=-1), counts.astype(mask.dtype))


def test_msa_files_reach_the_features(openfold3_source: Path, tmp_path: Path) -> None:
    """Without alignments the MSA is the query alone, which is a much weaker input.

    The query specification carries per-chain alignment paths, so this checks that
    path is wired rather than silently ignored.
    """
    # The stem must be a database name upstream recognizes; see
    # test_an_unrecognized_alignment_filename_is_refused.
    a3m = tmp_path / "colabfold_main.a3m"
    a3m.write_text(
        f">query\n{UBIQUITIN}\n"
        f">hit1\n{UBIQUITIN[:-1]}A\n"
        f">hit2\n{'A' * len(UBIQUITIN)}\n"
    )
    without = featurize_query(_spec())
    with_msa = featurize_query(_spec(main_msa_file_paths=[str(a3m)]))

    assert without["msa"].shape[1] == 1, "expected a single-sequence MSA baseline"
    assert with_msa["msa"].shape[1] > 1, "alignment file did not reach the features"
    assert with_msa["msa_mask"].shape[1] == with_msa["msa"].shape[1]


def test_padding_preserves_the_masks(features: dict) -> None:
    n_token = features["token_mask"].shape[-1]
    n_atom = features["atom_mask"].shape[-1]
    padded = pad_features(features, n_token=n_token + 20, n_atom=n_atom + 199)

    assert padded["token_mask"].shape == (1, n_token + 20)
    assert padded["atom_mask"].shape == (1, n_atom + 199)
    assert padded["token_bonds"].shape == (1, n_token + 20, n_token + 20)
    assert padded["ref_pos"].shape == (1, n_atom + 199, 3)
    # Padding must be masked out, not counted.
    assert float(padded["token_mask"].sum()) == float(features["token_mask"].sum())
    assert float(padded["atom_mask"].sum()) == float(features["atom_mask"].sum())
    # Rebuilt for the padded token count, and still counting only real atoms.
    assert padded["max_atom_per_token_mask"].shape == (1, (n_token + 20) * 23)
    assert float(padded["max_atom_per_token_mask"].sum()) == float(
        features["atom_mask"].sum()
    )


def test_padding_down_is_refused(features: dict) -> None:
    with pytest.raises(ValueError, match="cannot pad down"):
        pad_features(features, n_token=4, n_atom=4)


def test_padding_is_a_no_op_at_the_same_size(features: dict) -> None:
    same = pad_features(
        features,
        n_token=features["token_mask"].shape[-1],
        n_atom=features["atom_mask"].shape[-1],
    )
    for name in MODEL_FEATURES:
        np.testing.assert_array_equal(same[name], features[name], err_msg=name)


def test_saved_features_load_without_torch(features: dict, tmp_path: Path) -> None:
    """The point of saving is that inference needs none of the data stack."""
    path = save_features(features, tmp_path / "ubq.npz")
    loaded = np.load(path, allow_pickle=False)
    for name in MODEL_FEATURES:
        np.testing.assert_array_equal(loaded[name], features[name], err_msg=name)


def test_a_multi_query_spec_requires_choosing(openfold3_source: Path) -> None:
    spec = _spec()
    spec["queries"]["second"] = spec["queries"]["ubq"]
    with pytest.raises(KeyError, match="pass query_id"):
        featurize_query(spec)
    with pytest.raises(KeyError, match="not in the specification"):
        featurize_query(spec, query_id="absent")


def test_selected_query_rejects_native_covalent_bonds(
    openfold3_source: Path,
) -> None:
    spec = _spec()
    spec["queries"]["ubq"]["covalent_bonds"] = [
        [["A", 1, 1], ["A", 2, 2]]
    ]

    with pytest.raises(ValueError, match="covalent_bonds.*does not apply"):
        featurize_query(spec)


def test_unselected_query_msa_filename_is_not_validated(
    openfold3_source: Path, tmp_path: Path
) -> None:
    stray = tmp_path / "ignored.a3m"
    stray.write_text(f">query\n{UBIQUITIN}\n")
    spec = _spec()
    spec["queries"]["unused"] = _spec(
        main_msa_file_paths=[str(stray)]
    )["queries"]["ubq"]

    selected = featurize_query(spec, query_id="ubq")

    assert selected["token_mask"].shape[-1] == len(UBIQUITIN)


def test_an_unrecognized_alignment_filename_is_refused(
    openfold3_source: Path, tmp_path: Path
) -> None:
    """Upstream skips unknown stems silently and then dies in the parser.

    Converting that into an error here is the difference between "your file was
    named wrong" and an IndexError inside a file the caller has never seen.
    """
    stray = tmp_path / "my_alignment.a3m"
    stray.write_text(f">query\n{UBIQUITIN}\n")
    with pytest.raises(ValueError, match="would be ignored"):
        featurize_query(_spec(main_msa_file_paths=[str(stray)]))


def test_a_recognized_stem_in_a_subdirectory_is_accepted(
    openfold3_source: Path, tmp_path: Path
) -> None:
    """The check is on the stem, not the parent directory."""
    nested = tmp_path / "aln"
    nested.mkdir()
    path = nested / "uniref90_hits.a3m"
    path.write_text(f">query\n{UBIQUITIN}\n>hit\n{'A' * len(UBIQUITIN)}\n")
    features = featurize_query(_spec(main_msa_file_paths=[str(path)]))
    assert features["msa"].shape[1] > 1


def test_msa_directory_requires_a_supported_accepted_alignment(
    openfold3_source: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "alignments"
    directory.mkdir()
    (directory / "colabfold_main.txt").write_text("not an alignment\n")
    (directory / "unknown.a3m").write_text(f">query\n{UBIQUITIN}\n")

    with pytest.raises(ValueError, match="no supported .a3m/.sto"):
        featurize_query(_spec(main_msa_file_paths=[str(directory)]))


def test_msa_directory_accepts_a_supported_alignment_among_other_files(
    openfold3_source: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "alignments"
    directory.mkdir()
    (directory / "notes.txt").write_text("ignored\n")
    (directory / "colabfold_main.a3m").write_text(
        f">query\n{UBIQUITIN}\n>hit\n{'A' * len(UBIQUITIN)}\n"
    )

    features = featurize_query(_spec(main_msa_file_paths=[str(directory)]))

    assert features["msa"].shape[1] > 1


def test_features_drive_predict_end_to_end(openfold3_source: Path, randomized) -> None:
    """Sequence in, coordinates out.

    The featurizer is only useful if its output is what the model actually
    consumes, so this runs the whole path on one reduced-depth model rather than
    asserting shapes and stopping. Depth is reduced because this is a wiring check;
    numerical agreement at released depths is gated elsewhere.
    """
    import jax
    import jax.numpy as jnp
    import torch
    from openfold3.projects.of3_all_atom.model import OpenFold3

    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table
    from foldjax.models.openfold3.bridge.torch_mapping import map_inference_params
    from foldjax.models.openfold3.inference import predict, released_config

    from .test_composite_parity_trunk import COMPOSITE_SCALE, _reduced_config

    features = featurize_query(_spec())
    n_token = features["token_mask"].shape[-1]
    n_atom = features["atom_mask"].shape[-1]

    torch.manual_seed(0)
    model = randomized(OpenFold3(_reduced_config()), scale=COMPOSITE_SCALE)
    params = map_inference_params(dict(model.state_dict()))

    config = released_config(
        n_token=n_token, n_atom=n_atom, num_samples=1, no_rollout_steps=2
    )
    prediction = predict(
        jax.random.key(0),
        {name: jnp.asarray(value) for name, value in features.items()},
        params,
        config,
        representative_atom_table(),
    )

    assert prediction.coordinates.shape == (1, n_atom, 3)
    assert bool(np.isfinite(np.asarray(prediction.coordinates)).all())
    # Real coordinates, not a collapsed point.
    assert float(np.asarray(prediction.coordinates).std()) > 1.0
    assert prediction.plddt.shape == (1, n_atom)
    assert bool(np.isfinite(np.asarray(prediction.plddt)).all())


def _msa_features(tmp_path: Path):
    a3m = tmp_path / "colabfold_main.a3m"
    a3m.write_text(
        f">query\n{UBIQUITIN}\n"
        f">hit1\n{UBIQUITIN[:-1]}A\n"
        f">hit2\n{'A' * len(UBIQUITIN)}\n"
    )
    return featurize_query(_spec(main_msa_file_paths=[str(a3m)]))


def test_subsampling_cuts_rows_and_leaves_the_profile(
    openfold3_source: Path, tmp_path: Path
) -> None:
    """The cut must land on the alignment axis and nowhere else.

    `profile` and `deletion_mean` summarise the whole alignment per token, so
    slicing them would silently change what the model reads about the alignment it
    was not shown. Upstream leaves them alone for the same reason -- it subsamples
    inside the network, long after those two are computed.
    """
    from foldjax.models.openfold3.data import MSA_ROW_FEATURES, subsample_msa_rows

    features = _msa_features(tmp_path)
    rows = features["msa"].shape[1]
    assert rows > 1, "a single-row alignment cannot show subsampling working"
    depth = rows - 1
    capped = subsample_msa_rows(features, depth)

    for name in MSA_ROW_FEATURES:
        assert capped[name].shape[1] == depth, name
    for name in ("profile", "deletion_mean", "token_mask", "restype"):
        np.testing.assert_array_equal(capped[name], features[name])

    for depth in (rows, rows + 10, None):
        kept = subsample_msa_rows(features, depth)
        for name in MSA_ROW_FEATURES:
            np.testing.assert_array_equal(kept[name], features[name])


def test_subsampling_keeps_valid_rows_before_all_masked_filler(
    openfold3_source: Path, tmp_path: Path
) -> None:
    """Upstream's rule ranks rows by validity, not by position.

    A prefix cut is only the same thing when every row is valid. Alignments are
    padded to a shared depth across chains, so all-masked rows appear in the middle
    of the stack, and a prefix cut would spend the budget on rows that contribute
    nothing while dropping real hits.
    """
    from foldjax.models.openfold3.data import subsample_msa_rows

    features = dict(_msa_features(tmp_path))
    rows = features["msa"].shape[1]
    assert rows >= 3, "need at least three rows to place filler between two hits"
    # Blank out row 0, so a prefix cut and upstream's rule disagree observably.
    mask = np.array(features["msa_mask"])
    mask[:, 0] = 0.0
    features["msa_mask"] = mask
    marker = np.array(features["deletion_value"])
    marker[:, :, 0] = np.arange(rows, dtype=marker.dtype)[None, :]
    features["deletion_value"] = marker

    kept = subsample_msa_rows(features, rows - 1)
    # Row 0 is the invalid one, so it is the one dropped -- not the last row, which
    # is what a prefix cut would have kept it over.
    np.testing.assert_array_equal(
        kept["deletion_value"][0, :, 0], np.arange(1, rows, dtype=marker.dtype)
    )
    assert float(kept["msa_mask"].sum()) == float(mask.sum())


def test_subsampling_agrees_with_the_model_side_selection(
    openfold3_source: Path, tmp_path: Path
) -> None:
    """The host-side rule and the port's jnp ``subsample_msa`` must select alike.

    Two implementations of one rule is the arrangement that rots. This pins them
    together; the host one exists only so the full alignment never reaches the
    device.
    """
    from foldjax.models.openfold3.data import MSA_ROW_FEATURES, subsample_msa_rows
    from foldjax.models.openfold3.models.input_embedders import subsample_msa

    features = dict(_msa_features(tmp_path))
    rows = features["msa"].shape[1]
    mask = np.array(features["msa_mask"])
    mask[:, 0] = 0.0
    features["msa_mask"] = mask

    depth = rows - 1
    host = subsample_msa_rows(features, depth)
    unbatched = {name: features[name][0] for name in MSA_ROW_FEATURES}
    model = subsample_msa(unbatched, depth)
    for name in MSA_ROW_FEATURES:
        np.testing.assert_array_equal(
            np.asarray(host[name][0]), np.asarray(model[name]), err_msg=name
        )
