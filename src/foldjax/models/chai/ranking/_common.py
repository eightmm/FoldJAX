"""Shared confidence and ranking utilities matching Chai-1 inference."""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLYMER_ENTITY_TYPES = jnp.asarray((0, 1, 2, 4), dtype=jnp.int32)


def midpoint_bin_centers(
    min_bin: float,
    max_bin: float,
    num_bins: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Return the midpoint bins used by ``chai_lab.chai1._bin_centers``."""

    return jnp.linspace(min_bin, max_bin, 2 * num_bins + 1, dtype=dtype)[1::2]


def expectation(logits: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Expected value over the final logits axis."""

    return jnp.sum(jax.nn.softmax(logits, axis=-1) * weights, axis=-1)


def masked_mean(
    mask: jnp.ndarray,
    value: jnp.ndarray,
    *,
    axis: int | tuple[int, ...],
) -> jnp.ndarray:
    """Match Chai's denominator-clamped masked mean."""

    shape = jnp.broadcast_shapes(mask.shape, value.shape)
    mask = jnp.broadcast_to(mask, shape)
    value = jnp.broadcast_to(value, shape)
    numerator = jnp.sum(mask * value, axis=axis)
    denominator = jnp.maximum(jnp.sum(mask, axis=axis), 1)
    return numerator / denominator


def get_chain_masks_and_asyms(
    asym_id: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return masks for globally sorted, non-padding asym IDs.

    This is output postprocessing rather than a compiled model primitive:
    ``jnp.unique`` intentionally gives the same data-dependent chain axis as
    upstream Chai.
    """

    sorted_unique_asyms = jnp.unique(asym_id[mask.astype(bool)])
    chain_masks = asym_id[..., None, :] == sorted_unique_asyms[:, None]
    return chain_masks & mask[..., None, :].astype(bool), sorted_unique_asyms
