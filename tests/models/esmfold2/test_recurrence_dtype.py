"""Discretize original parameters before casting coefficients to pair dtype."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("coefficient", ["decay", "matrix"])
def test_recurrence_discretizes_original_fp32_before_pair_cast(
    monkeypatch, compiled, coefficient
):
    original = {
        "parcae_log_delta": jnp.array([-0.173, 0.219, -0.417, 0.739], jnp.float32),
        "parcae_log_a": jnp.array([-0.239, 0.723, 0.529, -0.117], jnp.float32),
        "parcae_b_cont": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 13 + 0.1037,
    }
    converted = {key: value.astype(jnp.bfloat16) for key, value in original.items()}
    converted.update(
        {
            "parcae_input_norm.weight": jnp.ones(4),
            "parcae_input_norm.bias": jnp.zeros(4),
        }
    )
    monkeypatch.setattr(model, "layer_norm", lambda value, *a, **kw: value)
    monkeypatch.setattr(model, "folding_trunk", lambda value, *a, **kw: value)
    shape = (1, 2, 2, 4)
    z = (
        jnp.ones(shape, jnp.bfloat16)
        if coefficient == "decay"
        else jnp.zeros(shape, jnp.bfloat16)
    )
    initial = (
        jnp.zeros(shape, jnp.bfloat16)
        if coefficient == "decay"
        else jnp.eye(4, dtype=jnp.bfloat16).reshape(shape)
    )

    def expected(p):
        delta = jax.nn.softplus(p["parcae_log_delta"])
        if coefficient == "decay":
            value = jnp.exp(-delta * jnp.exp(p["parcae_log_a"])).astype(jnp.bfloat16)
            return jnp.broadcast_to(value, shape)
        value = (delta[:, None] * p["parcae_b_cont"]).astype(jnp.bfloat16)
        return value.T.reshape(shape)

    assert not np.array_equal(expected(original), expected(converted))

    def run(p):
        return model.run_loops(
            jax.random.key(0),
            z,
            initial,
            None,
            None,
            jnp.ones(shape[:-1]),
            converted,
            settings=replace(model.ModelSettings(), d_pair=4),
            total_steps=1,
            recurrence_params=p,
        )

    result = (jax.jit(run) if compiled else run)(original)
    np.testing.assert_array_equal(result, expected(original))
