"""Pure JAX conditioning modules for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping
from math import pi

import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.primitives.transition import transition_forward

ConditioningParams = Mapping[str, object]


def single_conditioning_forward(
    params: ConditioningParams,
    times: jnp.ndarray,
    s_trunk: jnp.ndarray,
    s_inputs: jnp.ndarray,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run Boltz SingleConditioning with mapped PyTorch parameters."""

    s = jnp.concatenate((s_trunk, s_inputs), axis=-1)
    norm_single = params["norm_single"]
    single_embed = params["single_embed"]
    s = _linear(
        _layer_norm(s, norm_single["scale"], norm_single["bias"], eps),
        single_embed["kernel"],
        single_embed["bias"],
    )

    fourier_proj = params["fourier_embed"]["proj"]
    fourier_embed = jnp.cos(
        2.0 * pi * _linear(times[:, None], fourier_proj["kernel"], fourier_proj["bias"])
    )
    norm_fourier = params["norm_fourier"]
    normed_fourier = _layer_norm(
        fourier_embed,
        norm_fourier["scale"],
        norm_fourier["bias"],
        eps,
    )
    s = s + _linear(normed_fourier, params["fourier_to_single"]["kernel"])[:, None, :]

    for transition_params in params["transitions"]:
        s = s + transition_forward(transition_params, s, eps=eps)
    return s, normed_fourier


def pairwise_conditioning_forward(
    params: ConditioningParams,
    z_trunk: jnp.ndarray,
    token_rel_pos_feats: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz PairwiseConditioning with mapped PyTorch parameters."""

    z = jnp.concatenate((z_trunk, token_rel_pos_feats), axis=-1)
    init_proj = params["dim_pairwise_init_proj"]
    z = _linear(
        _layer_norm(
            z,
            init_proj["norm"]["scale"],
            init_proj["norm"]["bias"],
            eps,
        ),
        init_proj["linear"]["kernel"],
    )
    for transition_params in params["transitions"]:
        z = z + transition_forward(transition_params, z, eps=eps)
    return z
