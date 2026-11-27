"""Shape-stable random draws for masked serving buckets."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp


def supports_masked_prefix_draw() -> bool:
    """Whether the active JAX PRNG has shape-stable Threefry prefixes."""

    return bool(jax.config.jax_threefry_partitionable) and (
        str(jax.config.jax_default_prng_impl) == "threefry2x32"
    )


def masked_prefix_draw(
    draw: Callable[[jax.Array, tuple[int, ...]], jnp.ndarray],
    key: jax.Array,
    valid_mask: jnp.ndarray,
    *,
    trailing_shape: Sequence[int] = (),
) -> jnp.ndarray:
    """Draw the exact compact stream, scattered into a masked padded shape.

    JAX's counter stream is prefix-stable when only the flattened draw length
    grows, but a right-padded matrix is not a flattened prefix: its row stride
    changes.  Draw a flat semantic row stream, rank only valid rows, and scatter
    that compact prefix back into the padded layout.  Thus the real entries
    receive the same values as ``draw(key, compact_shape)`` without making the
    real cardinality a static compilation argument.

    ``trailing_shape`` stays attached to each semantic row (for example the
    coordinate width 3 or pair channel width 256), keeping gather indices well
    below the element count of very wide pair tensors.
    """

    mask = jnp.asarray(valid_mask, dtype=bool)
    trailing = tuple(int(size) for size in trailing_shape)
    if any(size < 1 for size in trailing):
        raise ValueError("random trailing dimensions must be positive")
    semantic_size = math.prod(mask.shape)
    source = draw(key, (semantic_size, *trailing))
    flat_mask = mask.reshape(-1)
    rank = jnp.cumsum(flat_mask.astype(jnp.int32)) - 1
    gathered = jnp.take(source, jnp.maximum(rank, 0), axis=0)
    keep = flat_mask.reshape((-1, *(1 for _ in trailing)))
    output = jnp.where(keep, gathered, jnp.zeros((), dtype=source.dtype))
    return output.reshape((*mask.shape, *trailing))


def prefix_atom_normal(
    key: jax.Array,
    atom_mask: jnp.ndarray,
    *,
    leading_shape: Sequence[int] | None = None,
    num_samples: int | None = None,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Draw one padded coordinate field from the compact atom prefix."""

    if (leading_shape is None) == (num_samples is None):
        raise ValueError(
            "prefix_atom_normal requires exactly one of leading_shape or num_samples"
        )
    prefix_leading = (
        (int(num_samples),) if leading_shape is None else tuple(leading_shape)
    )
    prefix_mask = jnp.broadcast_to(
        jnp.asarray(atom_mask, dtype=bool),
        (*prefix_leading, atom_mask.shape[0]),
    )
    return masked_prefix_draw(
        lambda normal_key, shape: jax.random.normal(
            normal_key, shape, dtype=dtype
        ),
        key,
        prefix_mask,
        trailing_shape=(3,),
    )


__all__ = [
    "masked_prefix_draw",
    "prefix_atom_normal",
    "supports_masked_prefix_draw",
]
