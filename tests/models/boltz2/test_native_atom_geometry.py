import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.primitives.native_atom_geometry import (
    inverse_squared_distance,
)


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 32, 128, 3), (1, 97, 32, 128, 3)])
def test_cpu_and_unobserved_profiles_preserve_ordinary_formula(shape):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU fallback contract")
    d = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 8192
    expected = 1.0 / (1.0 + jnp.sum(d * d, axis=-1, keepdims=True))
    np.testing.assert_array_equal(inverse_squared_distance(d), expected)


def test_other_dtype_retains_ordinary_formula():
    d = jnp.ones((2, 3), jnp.bfloat16)
    result = inverse_squared_distance(d)
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(result, jnp.full((2, 1), 0.25, jnp.bfloat16))
