"""Pure JAX Boltz-2 B-factor head."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

Params = Mapping[str, object]


def bfactor_forward(params: Params, s: jnp.ndarray) -> jnp.ndarray:
    """JAX port of ``boltz.model.modules.trunkv2.BFactorModule``.

    Linear projection of the single (sequence) embedding to a B-factor
    histogram of ``num_bins`` logits.
    """

    return s @ params["bfactor"]["kernel"] + params["bfactor"]["bias"]
