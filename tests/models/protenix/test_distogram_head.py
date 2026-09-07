from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.bridge.torch_mapping import map_distogram_state_dict
from foldjax.models.protenix.models.heads.head import DistogramParams, distogram_head
from foldjax.models.protenix.models.primitives.primitives import LinearParams


def test_distogram_head_matches_reference_formula() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 3, 5)).astype(np.float32)
    state = {
        "linear.weight": rng.normal(size=(4, 5)).astype(np.float32),
        "linear.bias": rng.normal(size=(4,)).astype(np.float32),
    }
    params = map_distogram_state_dict(state)

    actual = np.asarray(distogram_head(jnp.asarray(z), params))
    projected = np.matmul(z, state["linear.weight"].T) + state["linear.bias"]
    expected = projected + np.swapaxes(projected, -2, -3)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_native_amp_distogram_rounds_after_bias_then_symmetrizes(compiled, bias):
    rng = np.random.default_rng(872)
    z = jnp.asarray(rng.normal(size=(3, 3, 5)), jnp.float32)
    state = {"linear.weight": rng.normal(size=(4, 5)).astype(np.float32)}
    if bias:
        state["linear.bias"] = rng.normal(size=4).astype(np.float32)
    params = DistogramParams(
        LinearParams(
            jnp.asarray(state["linear.weight"]),
            jnp.asarray(state["linear.bias"]) if bias else None,
        )
    )

    def bf32(x):
        return np.asarray(jnp.asarray(x, jnp.bfloat16), np.float32)

    projected = bf32(z) @ bf32(state["linear.weight"]).T
    if bias:
        projected += bf32(state["linear.bias"])
    projected = bf32(projected)
    expected = bf32(projected + projected.swapaxes(-2, -3))

    def run(z, params):
        return distogram_head(z, params, compute_dtype=jnp.bfloat16)

    actual = (jax.jit(run) if compiled else run)(z, params)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)
