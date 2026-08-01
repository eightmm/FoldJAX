"""OpenDDE diffusion conditioning with structural-pair compression."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    TransitionParams,
    layer_norm,
    linear,
    transition,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    FourierParams,
    RelativePositionParams,
    fourier_embedding,
    relative_position_encoding,
)


class DiffusionConditioningParams(NamedTuple):
    """Parameters for OpenDDE's compressed diffusion conditioning."""

    relpe: RelativePositionParams
    layernorm_z_trunk: LayerNormParams
    linear_z_trunk: LinearParams
    layernorm_z: LayerNormParams
    linear_z: LinearParams
    transition_z1: TransitionParams
    transition_z2: TransitionParams
    layernorm_s: LayerNormParams
    linear_s: LinearParams
    fourier: FourierParams
    layernorm_n: LayerNormParams
    linear_n: LinearParams
    transition_s1: TransitionParams
    transition_s2: TransitionParams


def diffusion_conditioning_prepare_cache(
    relp_feature: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionConditioningParams,
) -> jnp.ndarray:
    """Build OpenDDE's reusable 128-channel pair conditioning cache."""

    z_trunk = linear(
        layer_norm(z_trunk, params.layernorm_z_trunk),
        params.linear_z_trunk,
    )
    pair_z = jnp.concatenate(
        [z_trunk, relative_position_encoding(relp_feature, params.relpe)],
        axis=-1,
    )
    pair_z = linear(layer_norm(pair_z, params.layernorm_z), params.linear_z)
    pair_z = pair_z + transition(pair_z, params.transition_z1)
    return pair_z + transition(pair_z, params.transition_z2)


def diffusion_conditioning(
    t_hat_noise_level: jnp.ndarray,
    relp_feature: jnp.ndarray,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionConditioningParams,
    *,
    pair_z: jnp.ndarray | None = None,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply OpenDDE diffusion conditioning in inference mode."""

    if pair_z is None:
        if not use_conditioning:
            s_trunk = jnp.zeros_like(s_trunk)
            z_trunk = jnp.zeros_like(z_trunk)
        pair_z = diffusion_conditioning_prepare_cache(
            relp_feature,
            z_trunk,
            params,
        )

    single_s = jnp.concatenate([s_trunk, s_inputs], axis=-1)
    single_s = linear(
        layer_norm(single_s, params.layernorm_s),
        params.linear_s,
    )
    noise_ratio = jnp.maximum(
        t_hat_noise_level / sigma_data,
        jnp.asarray(1.0e-10, dtype=t_hat_noise_level.dtype),
    )
    noise = fourier_embedding(jnp.log(noise_ratio) / 4.0, params.fourier)
    noise = noise.astype(single_s.dtype)
    noise = linear(layer_norm(noise, params.layernorm_n), params.linear_n)
    single_s = single_s[..., None, :, :] + noise[..., :, None, :]
    single_s = single_s + transition(single_s, params.transition_s1)
    return single_s + transition(single_s, params.transition_s2), pair_z
