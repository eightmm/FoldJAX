import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.esmfold2_transition_splitk_probe import serial_splitk, split_k_ranges


def test_measured_three_partition_geometry():
    assert split_k_ranges(1024, 32, 3) == [(0, 352), (352, 704), (704, 1024)]
    assert split_k_ranges(1024, 8, 3) == [(0, 344), (344, 688), (688, 1024)]
    assert split_k_ranges(33, 32, 3) == [(0, 32), (32, 33)]
    with pytest.raises(ValueError):
        split_k_ranges(1024, 0, 3)


@pytest.mark.parametrize("compiled", [False, True])
def test_partial_sum_rounding_is_observable(compiled):
    x = (
        jnp.zeros((1, 96), jnp.bfloat16)
        .at[0, 0]
        .set(256)
        .at[0, 32]
        .set(1)
        .at[0, 64]
        .set(-256)
    )
    weight = jnp.ones((1, 96), jnp.bfloat16)
    run = (
        jax.jit(serial_splitk, compiler_options={"xla_allow_excess_precision": False})
        if compiled
        else serial_splitk
    )
    np.testing.assert_array_equal(run(x, weight), [[0]])
    assert (
        float(jnp.matmul(x.astype(jnp.float32), weight.astype(jnp.float32).T)[0, 0])
        == 1
    )
