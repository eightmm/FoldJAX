from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model as m
from tests.models.esmfold2.test_diffusion_tape import _fixture
from tests.models.esmfold2.test_lm_dropout_tape import _loop_inputs


def test_dropout_tape_rejects_unproved_prefix_combination():
    with pytest.raises(ValueError, match="prefix"):
        m._dropout(
            jax.random.key(0),
            jnp.ones((2,)),
            0.25,
            keep_mask=jnp.ones((2,), bool),
            preserve_prefix_rng=True,
        )


def test_column_tape_rejects_unproved_prefix_combination():
    with pytest.raises(ValueError, match="prefix"):
        m._msa_column_keep(
            jax.random.key(0),
            jnp.ones((1, 2)),
            0.1,
            keep_tape=jnp.ones((1, 2), bool),
            preserve_prefix_rng=True,
        )


def test_sampler_tape_rejects_unproved_prefix_combination(monkeypatch):
    run, tape = _fixture(monkeypatch)
    with pytest.raises(ValueError, match="prefix"):
        run(jax.random.key(0), preserve_prefix_rng=True, **tape)


def test_loop_tape_rejects_unproved_prefix_combination():
    z, params, settings = _loop_inputs()
    with pytest.raises(ValueError, match="prefix"):
        m.run_loops(
            jax.random.key(0),
            z,
            z,
            z,
            None,
            jnp.ones(z.shape[:-1]),
            params,
            settings=settings,
            total_steps=2,
            preserve_prefix_rng=True,
            lm_dropout_masks=jnp.ones((2, *z.shape), bool),
        )


def test_taped_eager_driver_consumes_same_steps_as_scan(monkeypatch):
    run, tape = _fixture(monkeypatch)
    seen = []

    def observe(carry, step, *, draws, **kwargs):
        del kwargs
        jax.debug.callback(
            lambda s, q: seen.append((np.asarray(s), np.asarray(q))),
            step,
            draws[0],
            ordered=True,
        )
        x, previous, key, representation = carry
        return (x + draws[2], previous, key, representation), None

    # Test driver indexing, not the unrelated SVD/fusion rounding difference
    # between CPU eager and scan evaluation of the denoiser.
    monkeypatch.setattr(m.diffusion, "_step", observe)
    scanned = run(jax.random.key(0), **tape)
    jax.block_until_ready(scanned)
    scanned_steps = list(seen)
    seen.clear()
    eager = run(jax.random.key(0), early_exit_rmsd=0.0, **tape)
    jax.block_until_ready(eager)
    for left, right in zip(scanned, eager, strict=True):
        np.testing.assert_array_equal(left, right)
    assert len(seen) == len(scanned_steps) == 3
    for left, right in zip(scanned_steps, seen, strict=True):
        for a, b in zip(left, right, strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "name",
    [
        "initial_pair_state",
        "lm_dropout_masks",
        "msa_column_keep",
        "msa_row_choices",
        "diffusion_initial_normal",
        "diffusion_rotation_quaternions",
        "diffusion_translations",
        "diffusion_churn_normals",
    ],
)
def test_predict_rejects_any_tape_with_prefix_before_parameters(name):
    features = {
        "token_attention_mask": jnp.ones((1, 2), bool),
        "atom_attention_mask": jnp.ones((1, 2), bool),
    }
    with pytest.raises(ValueError, match="prefix"):
        m.predict(
            jax.random.key(0),
            features,
            {},
            settings=m.ModelSettings(),
            preserve_prefix_rng=True,
            **{name: jnp.zeros((1,))},
        )


@pytest.mark.parametrize("bad", ["shape", "integer", "nan"])
def test_initial_pair_preflight_rejects_invalid_states(bad):
    value = np.ones((1, 2, 2, 4), np.float32)
    if bad == "shape":
        value = value[..., :1]
    elif bad == "integer":
        value = value.astype(np.int32)
    else:
        value[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="initial pair"):
        m.validate_initial_pair_state(value, batch=1, tokens=2, width=4)


def test_initial_pair_and_msa_value_preflight_require_concrete_arrays():
    def pair(value):
        m.validate_initial_pair_state(value, batch=1, tokens=2, width=4)
        return value

    with pytest.raises(ValueError, match="concrete"):
        jax.jit(pair)(jnp.ones((1, 2, 2, 4)))

    def rows(value):
        m.validate_msa_tape(
            None,
            value,
            batch=1,
            tokens=2,
            depth=4,
            loops=1,
            settings=replace(m.ModelSettings(), msa_n_layers=0, max_msa_depth=2),
        )
        return value

    with pytest.raises(ValueError, match="concrete"):
        jax.jit(rows)(jnp.array([[0, 1]], jnp.int32))
