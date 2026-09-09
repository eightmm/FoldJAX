import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.protenix_atom_reduction_control import fp32_atom_aggregation
from foldjax.models.protenix.models.diffusion import atom


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_control_uses_fp32_accumulation_and_restores_function(reduction):
    original = atom.aggregate_atom_to_token
    x = jnp.asarray([1.0] + [2**-8] * 256, jnp.bfloat16).reshape(-1, 1)
    index = jnp.zeros(257, jnp.int32)
    expected = original(
        x.astype(jnp.float32), index, n_token=1, reduce=reduction
    ).astype(x.dtype)
    with fp32_atom_aggregation("foldjax"):
        actual = jax.jit(
            lambda value: atom.aggregate_atom_to_token(
                value, index, n_token=1, reduce=reduction
            )
        )(x)
    assert atom.aggregate_atom_to_token is original
    assert actual.dtype == x.dtype
    np.testing.assert_array_equal(actual, expected)


def test_invalid_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        with fp32_atom_aggregation("other"):
            pass
