"""Pure JAX Boltz-2 distogram head."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import linear as _linear

Params = Mapping[str, object]


def distogram_forward(
    params: Params,
    z: jnp.ndarray,
    *,
    num_distograms: int = 1,
    num_bins: int = 64,
) -> jnp.ndarray:
    """JAX port of ``boltz.model.modules.trunkv2.DistogramModule``.

    Symmetrizes the pair embedding then projects to ``num_bins`` logits.
    Returns shape ``(b, n, n, num_distograms, num_bins)``.
    """

    projection = params["distogram"]
    if projection["kernel"].dtype in (jnp.bfloat16, jnp.float16):
        # AMP rounds z+z.T once at the native Linear boundary; projecting
        # each side first introduces a different BF16 sum and bias rounding.
        logits = _linear(
            z + jnp.swapaxes(z, 1, 2), projection["kernel"], projection["bias"]
        )
    else:
        logits = z @ projection["kernel"]
        logits = logits + jnp.swapaxes(logits, 1, 2) + projection["bias"]
    b, n, _, _ = logits.shape
    return logits.reshape(b, n, n, num_distograms, num_bins)
