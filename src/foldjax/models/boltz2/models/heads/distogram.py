"""Pure JAX Boltz-2 distogram head."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

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

    logits = z @ params["distogram"]["kernel"]
    logits = logits + jnp.swapaxes(logits, 1, 2) + params["distogram"]["bias"]
    b, n, _, _ = logits.shape
    return logits.reshape(b, n, n, num_distograms, num_bins)
