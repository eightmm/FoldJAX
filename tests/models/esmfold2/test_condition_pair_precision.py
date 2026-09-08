import jax.numpy as jnp
import pytest

from foldjax.models.esmfold2.models import diffusion, primitives


def test_condition_pair_passes_unrounded_residual_and_original_params(monkeypatch):
    params = {"z_input_norm.weight": jnp.ones(4), "z_input_norm.bias": jnp.zeros(4)}
    calls = []
    monkeypatch.setattr(diffusion, "layer_norm", lambda x, *a: x)
    monkeypatch.setattr(diffusion, "linear", lambda x, *a: x)

    def transition(x, p, prefix, *, linear_dtype):
        assert p is params
        assert x.dtype == jnp.float32
        assert float(x[0, 0, 0, 0]) == float(jnp.float32(1.001))
        assert linear_dtype == jnp.bfloat16
        calls.append(prefix)
        return jnp.zeros_like(x, dtype=linear_dtype)

    monkeypatch.setattr(diffusion, "transition_layer", transition)
    x = jnp.full((1, 1, 1, 2), 1.001, dtype=jnp.float32)
    result = diffusion.condition_pair(x, x, params, trunk_dtype=jnp.bfloat16)
    assert result.dtype == jnp.float32
    assert calls == ["z_transitions.0", "z_transitions.1"]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_transition_autocast_keeps_norm_fp32(monkeypatch, dtype):
    params = {"norm.weight": jnp.ones(2), "norm.bias": jnp.zeros(2)}
    params.update(
        {f"{name}.weight": jnp.eye(2) for name in ("a_proj", "b_proj", "out_proj")}
    )
    calls = []

    def norm(x, weight, bias, **kwargs):
        assert x.dtype == weight.dtype == bias.dtype == jnp.float32
        calls.append("norm")
        return x

    def linear(x, weights, name):
        assert x.dtype == weights[f"{name}.weight"].dtype == jnp.dtype(dtype)
        calls.append(name)
        return x

    monkeypatch.setattr(primitives, "layer_norm", norm)
    monkeypatch.setattr(primitives, "linear", linear)
    result = primitives.transition_layer(jnp.ones((1, 2)), params, linear_dtype=dtype)
    assert result.dtype == jnp.dtype(dtype)
    assert calls == ["norm", "a_proj", "b_proj", "out_proj"]
    assert all(x.dtype == jnp.float32 for x in params.values())
