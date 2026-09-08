"""Materialized BF16 recurrence terms must not become one beta=1 GEMM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model


def _operands():
    z = jnp.full((1, 2, 2, 2), -1.015625, jnp.bfloat16)
    decay = jnp.ones((1, 1, 1, 2), jnp.bfloat16)
    injected = jnp.full(z.shape, 1.0078125, jnp.bfloat16)
    matrix = jnp.eye(2, dtype=jnp.bfloat16) * jnp.bfloat16(1.0078125)
    return z, decay, injected, matrix


@pytest.mark.parametrize("compiled", [False, True])
def test_bf16_recurrence_materializes_matmul_before_cancellation(compiled):
    def run(*args):
        return model._recurrence_update(*args, native_rounding=True)

    operands = _operands()
    # 1.0078125**2 = 1.015686035...; native BF16 stores 1.015625 first.
    result = (jax.jit(run) if compiled else run)(*operands)
    np.testing.assert_array_equal(result, jnp.zeros_like(result))
    fused = operands[0].astype(jnp.float32) + jnp.matmul(
        operands[2].astype(jnp.float32), operands[3].astype(jnp.float32).T
    )
    assert np.all(np.asarray(fused.astype(jnp.bfloat16)) != 0)


@pytest.mark.parametrize("native", [False, True])
def test_recurrence_rounding_barrier_is_scoped_and_separates_dot_from_sum(native):
    graph = jax.make_jaxpr(
        lambda *a: model._recurrence_update(*a, native_rounding=native)
    )(*_operands()).jaxpr
    names = [eq.primitive.name for eq in graph.eqns]
    assert ("optimization_barrier" in names) == native
    if native:
        assert (
            names.index("dot_general")
            < names.index("optimization_barrier")
            < names.index("add")
        )
        barrier = graph.eqns[names.index("optimization_barrier")]
        assert len(barrier.invars) == len(barrier.outvars) == 2
