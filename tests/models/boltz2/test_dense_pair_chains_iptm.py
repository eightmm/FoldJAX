"""Upstream's `chains_ptm`/`pair_chains_iptm` without a program per chain-label set.

The static-label dictionary path (`confidence_chain_ids`) is already checked
against eager evaluation in `test_confidence_chain_ids`; the dense path must
give the same numbers for every label, zero for labels the input lacks, and
one executable for every input whose labels fit the capacity bucket.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.models.predict as predict_module
from foldjax.models.boltz2 import api
from foldjax.models.boltz2.models.heads import confidence
from tests.models.boltz2.test_confidence_chain_ids import CHAIN_IDS, _inputs


@pytest.mark.parametrize("sequential", [False, True])
def test_dense_matrix_matches_the_label_dictionary(sequential) -> None:
    logits, coords, feats = _inputs()
    capacity = confidence.pair_chains_capacity(feats["asym_id"])
    assert capacity == 128
    expected = confidence._compute_ptms(
        logits, coords, feats, 5, recompute_nonpolymer_frames=False
    )

    def run(logits, coords):
        def evaluate(logits, coords, multiplicity):
            return confidence._compute_ptms(
                logits,
                coords,
                feats,
                multiplicity,
                pair_chains_capacity=capacity,
                recompute_nonpolymer_frames=False,
            )

        if sequential:
            result = jax.lax.map(
                lambda value: evaluate(value[0][None], value[1][None], 1),
                (logits, coords),
            )
            return jax.tree.map(lambda value: value[:, 0], result)
        return evaluate(logits, coords, 5)

    actual = jax.jit(run)(logits, coords)
    dense = np.asarray(actual[-1])
    assert dense.shape == (5, capacity, capacity)
    # The scalar scores do not move.
    for result, reference in zip(actual[:-1], expected[:-1], strict=True):
        np.testing.assert_allclose(result, reference, rtol=2e-6, atol=2e-7)
    for first in CHAIN_IDS:
        for second in CHAIN_IDS:
            np.testing.assert_allclose(
                dense[:, first, second],
                expected[-1][first][second],
                rtol=2e-6,
                atol=2e-7,
                err_msg=f"[{first}][{second}]",
            )
    absent = np.setdiff1d(np.arange(capacity), CHAIN_IDS)
    assert not dense[:, absent].any() and not dense[:, :, absent].any()


def test_capacity_is_a_power_of_two_bucket() -> None:
    assert confidence.pair_chains_capacity(np.arange(5)) == 32
    assert confidence.pair_chains_capacity(np.asarray([[0, 31]])) == 32
    assert confidence.pair_chains_capacity(np.asarray([0, 32])) == 64
    assert confidence.pair_chains_capacity(np.asarray([99])) == 128


def test_one_program_serves_every_label_set_in_the_bucket() -> None:
    traces = []
    rng = np.random.default_rng(3)
    tm = jnp.asarray(rng.uniform(size=(2, 6, 6)), jnp.float32)
    ones = jnp.ones((2, 6), jnp.float32)

    @jax.jit
    def dense(asym_id):
        traces.append(asym_id.shape)
        return confidence._dense_pair_chains_iptm(tm, ones, ones, asym_id, capacity=32)

    two = dense(jnp.asarray([[0, 0, 0, 1, 1, 1]] * 2))
    four = dense(jnp.asarray([[0, 1, 2, 3, 3, 3]] * 2))
    assert len(traces) == 1
    assert two.shape == four.shape == (2, 32, 32)
    # Entry [c, d] is the best row of chain d averaged over chain c's columns.
    tm_host = np.asarray(tm)
    # (Upstream's denominator carries a 1e-5 floor: 3 / (3 + 1e-5).)
    np.testing.assert_allclose(
        two[:, 1, 0], tm_host[:, :3, 3:].mean(axis=-1).max(axis=-1), rtol=1e-5
    )


def test_present_labels_and_chains_ptm_are_cut_from_the_bucket() -> None:
    dense = np.zeros((2, 32, 32), np.float32)
    dense[:, 3, 3] = [0.5, 0.6]
    dense[:, 3, 7] = [0.1, 0.2]
    feats = {
        "asym_id": np.asarray([[3, 3, 7, 7, 0]]),
        "token_pad_mask": np.asarray([[1, 1, 1, 1, 0]]),
    }
    selected = api._select_present_chains({"pair_chains_iptm": dense}, feats)
    assert selected["pair_chains_iptm"].shape == (2, 2, 2)
    np.testing.assert_array_equal(selected["chains_ptm"], dense[:, [3, 7], [3, 7]])
    np.testing.assert_array_equal(selected["pair_chains_iptm"][:, 0, 1], dense[:, 3, 7])
    # The dictionary form a direct caller asked for is left alone.
    untouched = {"pair_chains_iptm": {3: {3: np.ones(2)}}}
    assert api._select_present_chains(untouched, feats) == untouched


def test_the_wrapper_threads_the_capacity_and_keeps_coordinates(monkeypatch) -> None:
    trunk = {
        "s_inputs": jnp.zeros((1, 1, 1)),
        "s": jnp.zeros((1, 1, 1)),
        "z": jnp.zeros((1, 1, 1, 1)),
    }
    coords = jnp.arange(9, dtype=jnp.float32).reshape(3, 1, 3)
    seen = []

    def fake_confidence(*args, x_pred, pair_chains_capacity, **kwargs):
        seen.append(pair_chains_capacity)
        samples = x_pred.shape[0]
        width = pair_chains_capacity or 1
        return {
            "plddt": x_pred[:, 0, 0],
            "complex_plddt": x_pred[:, 0, 0],
            "ptm": jnp.zeros(samples),
            "iptm": jnp.zeros(samples),
            "pair_chains_iptm": jnp.broadcast_to(
                x_pred[:, 0, :1, None], (samples, width, width)
            ),
        }

    monkeypatch.setattr(
        predict_module, "boltz2_trunk_forward", lambda *args, **kwargs: trunk
    )
    monkeypatch.setattr(
        predict_module,
        "boltz2_sample_forward",
        lambda *args, **kwargs: {"sample_atom_coords": coords},
    )
    monkeypatch.setattr(
        predict_module,
        "distogram_forward",
        lambda *args, **kwargs: jnp.zeros((1, 1, 1, 1, 1)),
    )
    monkeypatch.setattr(predict_module, "confidence_module_forward", fake_confidence)

    def run(**changes):
        return predict_module.boltz2_predict(
            {"trunk": {}, "confidence": {}},
            {},
            jax.random.PRNGKey(0),
            multiplicity=3,
            run_distogram=False,
            confidence_sequentially=True,
            return_pair_chains_iptm=False,
            **changes,
        )

    without = run()
    dense = run(pair_chains_capacity=32)
    assert seen == [None, 32]
    assert dense["pair_chains_iptm"].shape == (3, 32, 32)
    np.testing.assert_array_equal(
        np.asarray(dense["sample_atom_coords"]),
        np.asarray(without["sample_atom_coords"]),
    )
