"""Native AMP GEMV must retain FP32 accumulation before its BF16 output."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.primitives._common import linear


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_one_column_amp_dot_requests_fp32_accumulator(dtype):
    x = jnp.ones((2, 128), dtype)
    weight = jnp.ones((128, 1), dtype)
    graph = jax.make_jaxpr(linear)(x, weight).jaxpr
    dots = [eq for eq in graph.eqns if eq.primitive.name == "dot_general"]
    assert len(dots) == 1
    assert dots[0].params["preferred_element_type"] == jnp.dtype(jnp.float32)
    assert graph.outvars[0].aval.dtype == jnp.dtype(dtype)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("compiled", [False, True])
def test_one_column_amp_matches_single_round_oracle(dtype, compiled):
    # Power-of-two operands have exact FP32 products/sum: no framework-specific
    # tie or summation-order tolerance is needed for this regression contract.
    rng = np.random.default_rng(411)
    x = jnp.asarray(rng.integers(-40, 41, (2, 5, 128)) / 32, dtype)
    weight = jnp.asarray(rng.integers(-40, 41, (128, 1)) / 32, dtype)
    exact = np.asarray(x, np.float64) @ np.asarray(weight, np.float64)
    expected = jnp.asarray(exact, dtype)
    actual = (jax.jit(linear) if compiled else linear)(x, weight)
    np.testing.assert_array_equal(actual, expected)


def test_fp32_one_column_path_stays_a_single_unchanged_dot():
    graph = jax.make_jaxpr(linear)(jnp.ones((2, 128)), jnp.ones((128, 1))).jaxpr
    assert [eq.primitive.name for eq in graph.eqns] == ["dot_general"]
    assert graph.outvars[0].aval.dtype == jnp.dtype(jnp.float32)
