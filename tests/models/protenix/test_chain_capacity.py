"""Capacity masks preserve the former exact-count confidence program."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.heads.confidence import (
    calculate_chain_based_ptm,
    confidence_scores_from_logits,
)


def _score_inputs(case: str, dtype=jnp.float32):
    rng = np.random.default_rng(42)
    active = 1 if case == "monomer" else 2 if case == "dimer" else 3
    chains = {
        1: [0, 0, 0, 0, 0, 0],
        2: [0, 0, 0, 1, 1, 1],
        3: [0, 0, 0, 1, 1, 2],
    }[active]
    if case == "sparse":
        chains = [0, 0, 0, 2, 2, 2]
    elif case == "truncated":
        active = 1
    asym_id = jnp.asarray([*chains, 99, 99], dtype=jnp.int32)
    frames = jnp.asarray([True] * 6 + [False] * 2)
    if case == "no_frames":
        frames = jnp.zeros_like(frames)
    elif case == "partial_frames":
        frames = frames.at[3:].set(False)
    token_ligand = jnp.asarray([True, False, False, False, False, True, False, False])
    atom_to_token = jnp.asarray([0, 0, 1, 2, 3, 4, 4, 5, 6, 7], dtype=jnp.int32)
    pae = rng.normal(size=(2, 8, 8, 8)).astype(np.float32)
    if case == "nan":
        pae[:, 0, 1, 0] = np.nan
    elif case == "inf":
        pae[:, 0, 1, :] = -np.inf
    features = {
        "plddt_logits": jnp.asarray(rng.normal(size=(2, 10, 5)), dtype=dtype),
        "pae_logits": jnp.asarray(pae, dtype=dtype),
        "pde_logits": jnp.asarray(rng.normal(size=(2, 8, 8, 8)), dtype=dtype),
        "distogram_logits": jnp.asarray(rng.normal(size=(8, 8, 8)), dtype=dtype),
        "token_has_frame": frames,
        "token_asym_id": asym_id,
        "atom_to_token_idx": atom_to_token,
        "atom_coordinate": jnp.zeros((2, 10, 3), dtype=jnp.float32),
        "atom_is_polymer": ~token_ligand[atom_to_token],
        "elements_one_hot": jax.nn.one_hot(jnp.full((10,), 6), 119),
        "mol_id": jnp.arange(10),
        "token_mask": jnp.asarray([True] * 6 + [False] * 2),
        "atom_mask": jnp.asarray([True] * 8 + [False] * 2),
        "token_is_ligand": token_ligand,
        "num_recycles": 2,
    }
    if case == "unspecified_polymer":
        features.pop("atom_is_polymer")
    elif case == "compact":
        features["return_confidence_details"] = False
    return active, features


def _crop(key, value, count):
    if key.startswith("chain_pair_"):
        return value[..., :count, :count]
    if key.startswith("chain_"):
        return value[..., :count]
    return value


@pytest.mark.parametrize(
    "case",
    [
        "monomer",
        "dimer",
        "trimer",
        "sparse",
        "no_frames",
        "partial_frames",
        "nan",
        "inf",
        "truncated",
        "unspecified_polymer",
        "compact",
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_capacity_scores_match_the_exact_count_program(case, dtype):
    count, kwargs = _score_inputs(case, dtype)
    # The no-mask implementation retains the pre-capacity Python-loop program.
    expected = confidence_scores_from_logits(**kwargs, n_chain=count)
    actual = confidence_scores_from_logits(**kwargs, chain_mask=jnp.arange(4) < count)
    assert actual.keys() == expected.keys()
    for key in expected:
        cropped = _crop(key, actual[key], count)
        assert cropped.dtype == expected[key].dtype, key
        np.testing.assert_allclose(
            np.asarray(cropped),
            np.asarray(expected[key]),
            rtol=2e-6,
            atol=1e-7,
            equal_nan=True,
            err_msg=key,
        )


def test_inactive_slots_do_not_dilute_chain_means_or_bespoke_iptm():
    probabilities = jnp.full((1, 6, 6, 8), 1.0 / 8, dtype=jnp.float32)
    kwargs = {
        "has_frame": jnp.ones((6,), dtype=bool),
        "asym_id": jnp.asarray([0, 0, 0, 1, 1, 1]),
        "token_is_ligand": jnp.zeros((6,), dtype=bool),
        "min_bin": 0.0,
        "max_bin": 32.0,
    }
    expected = calculate_chain_based_ptm(probabilities, **kwargs, n_chain=2)
    actual = calculate_chain_based_ptm(
        probabilities, **kwargs, chain_mask=jnp.asarray([True, True, False, False])
    )
    assert np.all(np.asarray(expected["chain_iptm"]) > 0)
    for key in expected:
        np.testing.assert_allclose(
            _crop(key, actual[key], 2), expected[key], rtol=1e-6, atol=1e-9
        )


def test_single_slot_capacity_keeps_the_monomer_output_contract():
    _, kwargs = _score_inputs("monomer")
    expected = confidence_scores_from_logits(**kwargs, n_chain=1)
    actual = confidence_scores_from_logits(**kwargs, chain_mask=jnp.ones((1,), bool))
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=1e-7)


def test_chain_count_changes_reuse_one_capacity_executable():
    _, kwargs = _score_inputs("trimer")
    probabilities = jax.nn.softmax(kwargs["pae_logits"], axis=-1)

    def score(asym_id, chain_mask):
        return calculate_chain_based_ptm(
            probabilities,
            kwargs["token_has_frame"],
            asym_id,
            kwargs["token_is_ligand"],
            token_mask=kwargs["token_mask"],
            min_bin=0.0,
            max_bin=32.0,
            chain_mask=chain_mask,
        )

    compiled = jax.jit(score)
    lowered = []
    for case in ("monomer", "dimer", "trimer"):
        count, features = _score_inputs(case)
        args = features["token_asym_id"], jnp.arange(4) < count
        lowered.append(compiled.lower(*args).as_text())
        jax.block_until_ready(compiled(*args))
    assert lowered[0] == lowered[1] == lowered[2]
    assert compiled._cache_size() == 1


def test_runtime_monomer_skips_quadratic_clash_scans(monkeypatch):
    from foldjax.models.protenix.models.heads import confidence as implementation

    scans = []

    def scan(name, coordinate, *args, **kwargs):
        jax.debug.callback(lambda: scans.append(name))
        return jnp.zeros(coordinate.shape[:-2], dtype=bool)

    monkeypatch.setattr(
        implementation, "calculate_clash", lambda *a, **kw: scan("clash", *a, **kw)
    )
    monkeypatch.setattr(
        implementation, "calculate_vdw_clash", lambda *a, **kw: scan("vdw", *a, **kw)
    )
    _, kwargs = _score_inputs("dimer")
    compiled = jax.jit(
        lambda mask: confidence_scores_from_logits(**kwargs, chain_mask=mask)
    )
    cases = ((1, []), (2, ["clash", "vdw"]), (1, ["clash", "vdw"]))
    for count, expected_scans in cases:
        jax.block_until_ready(compiled(jnp.arange(4) < count))
        jax.effects_barrier()
        assert sorted(scans) == expected_scans
    assert compiled._cache_size() == 1


@pytest.mark.parametrize("mask", [jnp.zeros((0,), bool), jnp.ones((2, 2), bool)])
def test_chain_capacity_requires_one_nonempty_axis(mask):
    _, kwargs = _score_inputs("monomer")
    with pytest.raises(ValueError, match="non-empty chain capacity axis"):
        confidence_scores_from_logits(**kwargs, chain_mask=mask)
