from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.cli.predict import _write
from foldjax.models.opendde.postprocess import (
    SHAPE_COMPLEMENTARITY_SCORE_KEYS,
    compute_contact_prob,
    opendde_confidence_scores,
)
from foldjax.models.protenix.data.output import _sample_summary


def _writer_features() -> dict[str, np.ndarray]:
    names = np.zeros((3, 4, 64), dtype=np.float32)
    for atom_i, name in enumerate(("N", "CA", "C")):
        for char_i, char in enumerate(name.ljust(4)):
            names[atom_i, char_i, ord(char) - 32] = 1.0
    elements = np.zeros((3, 128), dtype=np.float32)
    elements[:, [6, 5, 5]] = 1.0
    restype = np.zeros((2, 32), dtype=np.float32)
    restype[:, 0] = 1.0
    return {
        "atom_to_token_idx": np.asarray([0, 1, 1], dtype=np.int64),
        "ref_atom_name_chars": names,
        "ref_element": elements,
        "ref_mask": np.ones(3, dtype=np.float32),
        "restype": restype,
        "asym_id": np.asarray([0, 1], dtype=np.int64),
        "residue_index": np.ones(2, dtype=np.int64),
    }


def test_contact_probability_uses_opendde_inclusive_bin_tops() -> None:
    logits = jnp.zeros((1, 1, 96), dtype=jnp.float32)

    actual = compute_contact_prob(logits)

    np.testing.assert_allclose(actual, np.asarray([[24.0 / 96.0]]), atol=1e-7)


def test_contact_threshold_includes_bin_with_top_exactly_eight_angstrom() -> None:
    logits = jnp.full((2, 96), -100.0, dtype=jnp.float32)
    logits = logits.at[0, 23].set(100.0)
    logits = logits.at[1, 24].set(100.0)

    actual = compute_contact_prob(logits)

    np.testing.assert_allclose(actual, np.asarray([1.0, 0.0]), atol=1e-7)


def test_opendde_confidence_scores_replace_contact_dependent_summaries() -> None:
    num_samples, n_token, n_atom = 1, 2, 3
    output = {
        "coordinate": jnp.zeros((num_samples, n_atom, 3), dtype=jnp.float32),
        "plddt": jnp.zeros((num_samples, n_atom, 50), dtype=jnp.float32),
        "pae": jnp.zeros((num_samples, n_token, n_token, 64), dtype=jnp.float32),
        "pde": jnp.zeros((num_samples, n_token, n_token, 64), dtype=jnp.float32),
        "distogram_logits": jnp.zeros((n_token, n_token, 96), dtype=jnp.float32),
    }
    features = {
        "has_frame": jnp.ones((n_token,), dtype=bool),
        "asym_id": jnp.asarray([0, 0], dtype=jnp.int32),
        "atom_to_token_idx": jnp.asarray([0, 1, 1], dtype=jnp.int32),
        "is_protein": jnp.ones((n_atom,), dtype=jnp.int32),
    }

    actual = opendde_confidence_scores(output, features, num_recycles=10)

    np.testing.assert_allclose(actual["contact_probs"], 0.25, atol=1e-7)
    np.testing.assert_allclose(actual["summary_gpde"], 16.0, atol=1e-6)
    assert actual["atom_plddt"].shape == (num_samples, n_atom)
    assert actual["summary_ranking_score"].shape == (num_samples,)
    assert int(actual["num_recycles"]) == 10
    assert set(_sample_summary(actual, 0, num_samples)) == {
        "plddt",
        "gpde",
        "ptm",
        "iptm",
        "chain_gpde",
        "chain_pair_gpde",
        "chain_ptm",
        "chain_iptm",
        "chain_pair_iptm",
        "chain_pair_iptm_global",
        "chain_plddt",
        "chain_pair_plddt",
        "has_clash",
        "disorder",
        "ranking_score",
        "num_recycles",
    }


def test_opendde_writer_reports_shape_complementarity_without_raw_arrays(
    tmp_path,
) -> None:
    output = {
        "coordinate": np.zeros((2, 3, 3), dtype=np.float32),
        "atom_plddt": np.full((2, 3), 0.75, dtype=np.float32),
        # Rank zero is original sample one, so this also checks sample indexing.
        "summary_ranking_score": np.asarray([0.1, 0.9], dtype=np.float32),
        "shape_comp_token_pred": np.asarray(
            [[0.1, 0.2], [0.7, 0.8]], dtype=np.float32
        ),
        "shape_comp_token_mask": np.asarray(
            [[True, False], [False, True]], dtype=bool
        ),
        "shape_comp_global_pred": np.asarray([0.15, 0.75], dtype=np.float32),
        "shape_comp_pair_mean_pred": np.asarray([0.25, 0.85], dtype=np.float32),
        "shape_comp_pair_topk_mean_pred": np.asarray(
            [0.35, 0.95], dtype=np.float32
        ),
        "shape_comp_valid_pair_frac_pred": np.asarray(
            [0.5, 0.75], dtype=np.float32
        ),
        "shape_comp_uses_structural_tokens": np.asarray(True),
        # A large internal confidence array must not hitch a ride into JSON.
        "token_pair_pae": np.zeros((2, 2, 2), dtype=np.float32),
    }

    paths = _write(
        tmp_path,
        job_name="shape",
        seed=3,
        output=output,
        features=_writer_features(),
        include_raw=False,
        include_trunk=False,
    )

    confidence_path = (
        tmp_path
        / "shape"
        / "seed_3"
        / "predictions"
        / "shape_summary_confidence_sample_0.json"
    )
    assert confidence_path in paths
    confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
    assert SHAPE_COMPLEMENTARITY_SCORE_KEYS <= confidence.keys()
    np.testing.assert_allclose(confidence["shape_comp_token_pred"], [0.7, 0.8])
    assert confidence["shape_comp_token_mask"] == [False, True]
    assert confidence["shape_comp_global_pred"] == pytest.approx(0.75)
    assert confidence["shape_comp_pair_mean_pred"] == pytest.approx(0.85)
    assert confidence["shape_comp_pair_topk_mean_pred"] == pytest.approx(0.95)
    assert confidence["shape_comp_valid_pair_frac_pred"] == pytest.approx(0.75)
    assert confidence["shape_comp_uses_structural_tokens"] is True
    assert "token_pair_pae" not in confidence
    assert not confidence_path.with_name("raw_output.npz").exists()


def test_opendde_confidence_excludes_ligands_from_af3_clash_and_vdw_summary() -> None:
    num_samples, n_token, n_atom = 1, 2, 2
    output = {
        "coordinate": jnp.asarray(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]],
            dtype=jnp.float32,
        ),
        "plddt": jnp.zeros((num_samples, n_atom, 50), dtype=jnp.float32),
        "pae": jnp.zeros((num_samples, n_token, n_token, 64), dtype=jnp.float32),
        "pde": jnp.zeros((num_samples, n_token, n_token, 64), dtype=jnp.float32),
        "distogram_logits": jnp.zeros((n_token, n_token, 96), dtype=jnp.float32),
    }
    features = {
        "has_frame": jnp.ones((n_token,), dtype=bool),
        "asym_id": jnp.asarray([0, 1], dtype=jnp.int32),
        "atom_to_token_idx": jnp.asarray([0, 1], dtype=jnp.int32),
        "is_protein": jnp.asarray([1, 0], dtype=jnp.int32),
        "ref_element": jnp.zeros((n_atom, 128), dtype=jnp.float32),
    }

    actual = opendde_confidence_scores(output, features, num_recycles=1)

    assert not bool(actual["has_clash"][0])
    assert "has_vdw_clash" not in actual
    assert "summary_ranking_score_vdw_penalized" not in actual
