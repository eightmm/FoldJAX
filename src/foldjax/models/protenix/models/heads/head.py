"""Small output heads ported from Protenix."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.primitives import LinearParams, linear


class DistogramParams(NamedTuple):
    """Parameters for ``protenix.model.modules.head.DistogramHead``."""

    linear: LinearParams


def distogram_head(z: jnp.ndarray, params: DistogramParams) -> jnp.ndarray:
    """Apply the Protenix distogram head.

    The reference computes ``linear(z) + linear(z).transpose(-2, -3)`` where
    the token-pair axes are the two dimensions before the channel dimension.
    """

    logits = linear(z, params.linear)
    return logits + jnp.swapaxes(logits, -2, -3)
