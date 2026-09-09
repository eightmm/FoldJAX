"""Explicit autocast preserves ordinary shared primitive contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.primitives.primitives import (
    AutocastLinearParams,
    LinearParams,
    linear,
)


def test_explicit_autocast_narrows_fp32_residual_and_bias_under_jit():
    x = jnp.array([[1.003, 2.007]], dtype=jnp.float32)
    weight = jnp.array([[1.0, 1.0]], dtype=jnp.bfloat16)
    bias = jnp.array([0.003], dtype=jnp.float32)
    result = jax.jit(linear)(x, AutocastLinearParams(weight, bias))
    assert result.dtype == jnp.bfloat16
    expected = (x.astype(jnp.bfloat16) @ weight.T) + bias.astype(jnp.bfloat16)
    np.testing.assert_array_equal(result, expected)
    ordinary = jax.jit(linear)(x, LinearParams(weight, bias))
    assert ordinary.dtype == jnp.float32
    assert not np.array_equal(result.astype(jnp.float32), ordinary)


def test_fp32_island_keeps_original_operands():
    x = jnp.array([[1.003, 2.007]], dtype=jnp.float32)
    weight = jnp.array([[0.1003, 0.2007]], dtype=jnp.float32)
    result = jax.jit(linear)(x, LinearParams(weight))
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result, np.asarray(x) @ np.asarray(weight).T)
