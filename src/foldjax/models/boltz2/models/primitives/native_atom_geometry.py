"""Observed native CUDA distance reduction; other profiles retain JAX arithmetic."""

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_mesh
from foldjax.models.boltz2.models.primitives.native_pwa_weights import _cuda_rn_divide


def _ordinary(d):
    return 1.0 / (1.0 + jnp.sum(d * d, axis=-1, keepdims=True))


def _cuda_inverse_squared_distance(d):
    # Native's three-element reduction combines lanes 0/2 before lane 1.
    # Retain each FP32 boundary instead of contracting products into additions.
    squares = jax.lax.optimization_barrier(d * d)
    partial = jax.lax.optimization_barrier(squares[..., 0] + squares[..., 2])
    total = jax.lax.optimization_barrier(partial + squares[..., 1])
    denominator = jax.lax.optimization_barrier(1.0 + total)[..., None]
    return _cuda_rn_divide(jnp.ones_like(denominator), denominator)


def inverse_squared_distance(d):
    if (d.shape != (1, 97, 32, 128, 3) or d.dtype != jnp.float32
            or cp_mesh() is not None):
        return _ordinary(d)
    return jax.lax.platform_dependent(
        d, cuda=_cuda_inverse_squared_distance, default=_ordinary
    )
