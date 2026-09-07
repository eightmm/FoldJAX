"""Small output heads ported from Protenix."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.primitives import LinearParams, linear


class DistogramParams(NamedTuple):
    """Parameters for ``protenix.model.modules.head.DistogramHead``."""

    linear: LinearParams


def distogram_head(
    z: jnp.ndarray,
    params: DistogramParams,
    *,
    compute_dtype: jnp.dtype | None = None,
) -> jnp.ndarray:
    """Apply the Protenix distogram head.

    The reference computes ``linear(z) + linear(z).transpose(-2, -3)`` where
    the token-pair axes are the two dimensions before the channel dimension.
    """

    if compute_dtype == jnp.bfloat16:
        # Native F.linear under AMP accumulates the quantized bias before its
        # one BF16 output rounding. A BF16 matmul followed by +bias rounds twice.
        logits = jnp.matmul(
            z.astype(jnp.bfloat16),
            jnp.swapaxes(params.linear.weight.astype(jnp.bfloat16), -1, -2),
            preferred_element_type=jnp.float32,
        )
        if params.linear.bias is not None:
            logits = logits + params.linear.bias.astype(jnp.bfloat16).astype(
                jnp.float32
            )
        logits = logits.astype(jnp.bfloat16)
    else:
        logits = linear(z, params.linear)
    return logits + jnp.swapaxes(logits, -2, -3)
