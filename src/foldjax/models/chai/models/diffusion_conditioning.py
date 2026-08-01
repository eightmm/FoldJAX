"""Chai diffusion conditioning recovered from ``diffusion_module.pt``.

This is the complete 31-tensor conditioning prefix of the denoiser.  Unlike
the BF16 trunk, the exported diffusion graph receives float32 representations
and keeps these projections and transitions in float32.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from foldjax.models.chai.models.pairformer import (
    PairformerTransitionParams,
    map_pairformer_transition,
    pairformer_transition,
)
from foldjax.models.chai.models.primitives import layer_norm, linear


class NormLinearParams(NamedTuple):
    norm_weight: jnp.ndarray
    norm_bias: jnp.ndarray
    linear_weight: jnp.ndarray


class DiffusionConditioningParams(NamedTuple):
    token_pair_projection: NormLinearParams
    token_single_projection: NormLinearParams
    pair_transition_1: PairformerTransitionParams
    pair_transition_2: PairformerTransitionParams
    single_transition_1: PairformerTransitionParams
    single_transition_2: PairformerTransitionParams
    fourier_weights: jnp.ndarray
    fourier_bias: jnp.ndarray
    fourier_projection: NormLinearParams
    single_norm_weight: jnp.ndarray
    single_norm_bias: jnp.ndarray
    pair_norm_weight: jnp.ndarray
    pair_norm_bias: jnp.ndarray


def map_norm_linear(state: Mapping[str, Any], prefix: str) -> NormLinearParams:
    return NormLinearParams(
        norm_weight=jnp.asarray(state[f"{prefix}.0.weight"]),
        norm_bias=jnp.asarray(state[f"{prefix}.0.bias"]),
        linear_weight=jnp.asarray(state[f"{prefix}.1.weight"]),
    )


def map_diffusion_conditioning(
    state: Mapping[str, Any], prefix: str = "diffusion_conditioning"
) -> DiffusionConditioningParams:
    """Map all 31 official diffusion-conditioning tensors."""
    return DiffusionConditioningParams(
        token_pair_projection=map_norm_linear(state, f"{prefix}.token_pair_proj"),
        token_single_projection=map_norm_linear(state, f"{prefix}.token_in_proj"),
        pair_transition_1=map_pairformer_transition(state, f"{prefix}.pair_trans1"),
        pair_transition_2=map_pairformer_transition(state, f"{prefix}.pair_trans2"),
        single_transition_1=map_pairformer_transition(state, f"{prefix}.single_trans1"),
        single_transition_2=map_pairformer_transition(state, f"{prefix}.single_trans2"),
        fourier_weights=jnp.asarray(state[f"{prefix}.fourier_embedding.weights"]),
        fourier_bias=jnp.asarray(state[f"{prefix}.fourier_embedding.bias"]),
        fourier_projection=map_norm_linear(state, f"{prefix}.fourier_proj"),
        single_norm_weight=jnp.asarray(state[f"{prefix}.single_ln.weight"]),
        single_norm_bias=jnp.asarray(state[f"{prefix}.single_ln.bias"]),
        pair_norm_weight=jnp.asarray(state[f"{prefix}.pair_ln.weight"]),
        pair_norm_bias=jnp.asarray(state[f"{prefix}.pair_ln.bias"]),
    )


def norm_linear(x: jnp.ndarray, params: NormLinearParams) -> jnp.ndarray:
    normalized = layer_norm(x.astype(jnp.float32), params.norm_weight, params.norm_bias)
    return linear(normalized, params.linear_weight)


def diffusion_conditioning(
    token_single_initial_repr: jnp.ndarray,
    token_pair_initial_repr: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    token_pair_trunk_repr: jnp.ndarray,
    noise_sigma: jnp.ndarray,
    params: DiffusionConditioningParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return conditioned ``(single, pair)`` representations.

    ``noise_sigma`` has shape ``[batch, diffusion_samples]``.  The returned
    single representation has shape ``[batch, samples, tokens, 384]`` while
    the pair representation remains ``[batch, tokens, tokens, 256]``.
    """
    if noise_sigma.ndim != 2:
        raise ValueError("noise_sigma must have shape [batch, samples]")

    pair = diffusion_pair_conditioning(
        token_pair_initial_repr, token_pair_trunk_repr, params
    )

    single = diffusion_single_conditioning(
        token_single_initial_repr,
        token_single_trunk_repr,
        noise_sigma,
        params,
    )
    return single, pair


def diffusion_pair_conditioning_projection(
    token_pair_initial_repr: jnp.ndarray,
    token_pair_trunk_repr: jnp.ndarray,
    params: DiffusionConditioningParams,
) -> jnp.ndarray:
    """Project one complete or row-chunked pair input."""

    pair_input = jnp.concatenate(
        [token_pair_trunk_repr, token_pair_initial_repr], axis=-1
    )
    return norm_linear(pair_input, params.token_pair_projection)


def diffusion_pair_conditioning(
    token_pair_initial_repr: jnp.ndarray,
    token_pair_trunk_repr: jnp.ndarray,
    params: DiffusionConditioningParams,
) -> jnp.ndarray:
    """Return the sigma-independent conditioned pair representation."""

    pair = diffusion_pair_conditioning_projection(
        token_pair_initial_repr, token_pair_trunk_repr, params
    )
    pair = pair + pairformer_transition(pair, params.pair_transition_1, lin=linear)
    pair = pair + pairformer_transition(pair, params.pair_transition_2, lin=linear)
    return layer_norm(
        pair.astype(jnp.float32), params.pair_norm_weight, params.pair_norm_bias
    )


def diffusion_single_conditioning(
    token_single_initial_repr: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    noise_sigma: jnp.ndarray,
    params: DiffusionConditioningParams,
) -> jnp.ndarray:
    """Return the sigma-dependent conditioned single representation."""

    single_input = jnp.concatenate(
        [token_single_initial_repr, token_single_trunk_repr], axis=-1
    )
    single = norm_linear(single_input, params.token_single_projection)

    log_sigma = jnp.log(jnp.maximum(noise_sigma, jnp.finfo(jnp.float32).eps)) * 0.25
    frequencies = (
        log_sigma[..., None] * params.fourier_weights + params.fourier_bias
    ) * (2.0 * math.pi)
    noise_embedding = jnp.cos(frequencies)[:, :, None, :]
    noise_embedding = norm_linear(noise_embedding, params.fourier_projection)

    single = single[:, None, :, :] + noise_embedding
    single = single + pairformer_transition(
        single, params.single_transition_1, lin=linear
    )
    single = single + pairformer_transition(
        single, params.single_transition_2, lin=linear
    )
    single = layer_norm(
        single.astype(jnp.float32),
        params.single_norm_weight,
        params.single_norm_bias,
    )
    return single
