"""Measured CUDA PWA reduction with conservative profile/device fallback."""

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_mesh


def spatial_softmax(x):
    from foldjax.models.boltz2.models.primitives.native_pwa_weights import (
        _cuda_rn_divide,
    )

    if x.shape[-2:] != (437, 8) or x.dtype != jnp.float32:
        raise ValueError("spatial softmax requires FP32 437 by 8")
    rows = jnp.swapaxes(x, -1, -2)
    padded = jnp.pad(
        rows, [(0, 0)] * (rows.ndim - 1) + [(0, 75)], constant_values=-jnp.inf
    )
    exponentials = jnp.exp(padded - jnp.max(padded, axis=-1, keepdims=True))
    lanes = exponentials.reshape(*rows.shape[:-1], 4, 128)
    total = jnp.zeros_like(lanes[..., 0, :])
    for i in range(4):
        total = jax.lax.optimization_barrier(total + lanes[..., i, :])
    for offset in (64, 32, 16, 8, 4, 2, 1):
        total = jax.lax.optimization_barrier(
            total[..., :offset] + total[..., offset : 2 * offset]
        )
    output = _cuda_rn_divide(exponentials, total)[..., :437]
    return jnp.swapaxes(output, -1, -2)


def pwa_softmax(x):
    def fallback(x):
        return jax.nn.softmax(x, axis=-2)

    # Only this finite CUDA profile has native-input reduction evidence.
    if x.shape != (1, 437, 437, 8) or x.dtype != jnp.float32 or cp_mesh() is not None:
        return fallback(x)
    return jax.lax.platform_dependent(x, cuda=spatial_softmax, default=fallback)
