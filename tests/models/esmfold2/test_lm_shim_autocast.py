"""LM-shim-local autocast routing; native CUDA operator parity is a separate gate."""

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.esmfold2.models import model as m
from tests.models.esmfold2.test_lm_embedding_cache import _parameters


def test_native_shim_keeps_original_fp32_normalization_weights(monkeypatch):
    params = _parameters(9)
    hidden = jnp.asarray(
        np.random.default_rng(4).normal(size=(1, 3, 5, 8)), jnp.float32
    )
    seen = []
    original = m.layer_norm

    def norm(x, weight=None, bias=None, **kwargs):
        seen.append((x, weight, bias))
        return original(x, weight, bias, **kwargs)

    monkeypatch.setattr(m, "layer_norm", norm)
    pair = m.language_model_pair(hidden, params, compute_dtype=jnp.bfloat16)
    assert pair.dtype == jnp.float32
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0][0], hidden)
    np.testing.assert_array_equal(
        seen[0][1], params["language_model.base_z_linear.0.weight"]
    )
    np.testing.assert_array_equal(
        seen[1][1], params["language_model.base_z_mlp.1.weight"]
    )
    assert all(
        x.dtype == jnp.float32 and weight.dtype == jnp.float32 for x, weight, _ in seen
    )


def test_embedding_uses_fp32_softmax_before_bf16_matmul(monkeypatch):
    params = _parameters(7)
    hidden = jnp.asarray(
        np.random.default_rng(8).normal(size=(1, 3, 5, 8)), jnp.float32
    )
    seen = []
    original = jax.nn.softmax

    def softmax(x, *a, **kw):
        seen.append(x)
        return original(x, *a, **kw)

    monkeypatch.setattr(jax.nn, "softmax", softmax)
    result = m.language_model_embedding(hidden, params, compute_dtype=jnp.bfloat16)
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(seen[0], params["language_model.base_z_combine"])
    assert seen[0].dtype == jnp.float32


def test_autocast_linear_applies_bias_before_bf16_output_rounding():
    x = jnp.array([[1.003, 2.007]], jnp.float32)
    params = {
        "weight": jnp.array([[0.999, 1.003]], jnp.float32),
        "bias": jnp.array([0.007], jnp.float32),
    }
    # Empty prefixes are not part of Linear's flat-name contract.
    params = {f"x.{name}": value for name, value in params.items()}
    expected = (
        x.astype(jnp.bfloat16).astype(jnp.float32)
        @ params["x.weight"].astype(jnp.bfloat16).astype(jnp.float32).T
        + params["x.bias"].astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    actual = jax.jit(lambda value: m._lm_autocast_linear(value, params, "x"))(x)
    np.testing.assert_array_equal(actual, expected)


def test_direct_default_contract_and_explicit_native_dtype_are_distinct():
    params = _parameters(2)
    hidden = jnp.asarray(
        np.random.default_rng(2).normal(size=(1, 3, 5, 8)), jnp.float32
    )
    direct = m.language_model_pair(hidden, params)
    explicit = m.language_model_pair(hidden, params, compute_dtype=jnp.float32)
    np.testing.assert_array_equal(direct, explicit)
    native = jax.jit(
        lambda h: m.language_model_pair(h, params, compute_dtype=jnp.bfloat16)
    )(hidden)
    assert native.dtype == jnp.float32
    assert not np.array_equal(native, direct)


def test_trunk_storage_cast_does_not_round_original_lm_shim_parameters():
    params = _parameters(3)
    cast = m._cast(params, m.TRUNK_PREFIXES, jnp.bfloat16)
    for name, value in params.items():
        assert cast[name] is value


def test_explicit_fp32_compute_preserves_prior_promotion_of_bf16_hidden():
    params = _parameters(4)
    hidden = jnp.ones((1, 3, 5, 8), jnp.bfloat16)
    actual = m.language_model_embedding(hidden, params, compute_dtype=jnp.float32)
    expected = m.language_model_embedding(hidden.astype(jnp.float32), params)
    np.testing.assert_array_equal(actual, expected)
