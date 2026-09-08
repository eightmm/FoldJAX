"""Native recycle norm keeps original affine values until the linear cast."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model


@pytest.mark.parametrize("compiled", [False, True])
def test_recycle_norm_original_affine_and_fp32_output(monkeypatch, compiled):
    weight = jnp.array([1.0035, -0.1037], jnp.float32)
    original = {"parcae_input_norm.weight": weight, "parcae_input_norm.bias": weight}
    params = {key: value.astype(jnp.bfloat16) for key, value in original.items()}
    # softplus(log_delta)=1, so B=identity and the normalized injection is visible.
    params.update(
        {
            "parcae_log_delta": jnp.full(2, jnp.log(jnp.expm1(1.0))),
            "parcae_log_a": jnp.zeros(2),
            "parcae_b_cont": jnp.eye(2),
        }
    )
    monkeypatch.setattr(model, "folding_trunk", lambda x, *a, **kw: x)
    sentinel = jnp.array([0.5017, 0.1037], jnp.float32)

    def norm(x, selected, prefix):
        assert prefix == "parcae_input_norm"
        assert x.dtype == jnp.bfloat16
        assert selected["parcae_input_norm.weight"].dtype == jnp.float32
        return jnp.broadcast_to(
            selected["parcae_input_norm.weight"] + sentinel, x.shape
        )

    monkeypatch.setattr(model.trunk_ops, "_autocast_norm", norm)

    def run(affine):
        z = jnp.zeros((1, 2, 2, 2), jnp.bfloat16)
        return model.run_loops(
            jax.random.key(0),
            z,
            z,
            None,
            None,
            jnp.ones(z.shape[:-1]),
            params,
            settings=replace(model.ModelSettings(), d_pair=2),
            total_steps=1,
            injection_norm_params=affine,
        )

    result = (jax.jit(run) if compiled else run)(original)
    expected = (weight + sentinel).astype(jnp.bfloat16)
    np.testing.assert_array_equal(result, jnp.broadcast_to(expected, result.shape))
