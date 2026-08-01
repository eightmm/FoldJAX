from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.opendde.postprocess import (
    compute_contact_prob,
    opendde_confidence_scores,
)
from foldjax.models.protenix.data.output import _sample_summary


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
    n_sample, n_token, n_atom = 1, 2, 3
    output = {
        "coordinate": jnp.zeros((n_sample, n_atom, 3), dtype=jnp.float32),
        "plddt": jnp.zeros((n_sample, n_atom, 50), dtype=jnp.float32),
        "pae": jnp.zeros((n_sample, n_token, n_token, 64), dtype=jnp.float32),
        "pde": jnp.zeros((n_sample, n_token, n_token, 64), dtype=jnp.float32),
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
    assert actual["atom_plddt"].shape == (n_sample, n_atom)
    assert actual["summary_ranking_score"].shape == (n_sample,)
    assert int(actual["num_recycles"]) == 10
    assert set(_sample_summary(actual, 0, n_sample)) == {
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


def test_opendde_confidence_excludes_ligands_from_af3_clash_and_vdw_summary() -> None:
    n_sample, n_token, n_atom = 1, 2, 2
    output = {
        "coordinate": jnp.asarray(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]],
            dtype=jnp.float32,
        ),
        "plddt": jnp.zeros((n_sample, n_atom, 50), dtype=jnp.float32),
        "pae": jnp.zeros((n_sample, n_token, n_token, 64), dtype=jnp.float32),
        "pde": jnp.zeros((n_sample, n_token, n_token, 64), dtype=jnp.float32),
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
