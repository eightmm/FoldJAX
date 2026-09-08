import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models.primitives import _cuda_profile_divide


@pytest.mark.parametrize("rows", [0, 1, 7, 33])
def test_profile_division_shape_and_tail(rows):
    result = jax.eval_shape(
        _cuda_profile_divide,
        jax.ShapeDtypeStruct((1, rows, 33), jnp.float32),
        jax.ShapeDtypeStruct((1, rows, 1), jnp.float32),
    )
    assert result.shape == (1, rows, 33)
    assert result.dtype == jnp.float32


def test_profile_division_rejects_low_precision_operands():
    with pytest.raises(ValueError, match="FP32"):
        _cuda_profile_divide(jnp.ones((2, 33), jnp.bfloat16), jnp.ones((2, 1)))


@pytest.mark.skipif(
    os.environ.get("FOLDJAX_TEST_PROFILE_CUDA") != "1",
    reason="queued CUDA operator gate",
)
@pytest.mark.parametrize("rows", [1, 7, 33])
def test_profile_division_cuda_rounds_integer_count_ratios(rows):
    assert jax.default_backend() == "gpu"
    rng = np.random.default_rng(27)
    denominator = rng.integers(1, 4097, size=(1, rows, 1)).astype(np.float32)
    numerator = np.minimum(
        rng.integers(0, 4097, size=(1, rows, 33)), denominator
    ).astype(np.float32)
    expected = (numerator.astype(np.float64) / denominator.astype(np.float64)).astype(
        np.float32
    )
    actual = jax.jit(_cuda_profile_divide)(
        jnp.asarray(numerator), jnp.asarray(denominator)
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)
