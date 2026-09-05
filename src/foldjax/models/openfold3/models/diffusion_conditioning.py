"""Noise and single conditioning for the diffusion module (AF3 Algorithm 21).

Covers both halves of ``DiffusionConditioning``: the trunk and input single
representations are concatenated and projected, a Fourier embedding of the log
noise level is added, and the trunk pair representation is conditioned with
relative-position features.

``FourierEmbedding``'s ``w``/``b`` are *buffers*, not learned parameters — they
are sampled once from a seeded generator at construction. They appear in the
``state_dict`` and must be carried over from the checkpoint rather than
re-sampled, since JAX cannot reproduce torch's generator stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUTransitionParams,
    layer_norm,
    linear,
    swiglu_transition,
)
from foldjax.models.openfold3.models.relpos import relpos_complex


class FourierEmbeddingParams(NamedTuple):
    """Buffers for ``FourierEmbedding``: frequencies and phases, not weights."""

    w: jnp.ndarray
    b: jnp.ndarray


class DiffusionConditioningParams(NamedTuple):
    """Parameters for ``DiffusionConditioning``.

    Both the single and the pair path live here. Every layer norm is scale-only
    (``create_offset=False``).
    """

    layer_norm_s: LayerNormParams
    linear_s: LinearParams
    fourier_emb: FourierEmbeddingParams
    layer_norm_n: LayerNormParams
    linear_n: LinearParams
    transition_s: tuple[SwiGLUTransitionParams, ...]
    layer_norm_z: LayerNormParams
    linear_z: LinearParams
    transition_z: tuple[SwiGLUTransitionParams, ...]


# Kept as an alias: the name described only the single path, which this type
# outgrew when the pair path was added.
SingleConditioningParams = DiffusionConditioningParams


def fourier_embedding(
    x: jnp.ndarray, params: FourierEmbeddingParams
) -> jnp.ndarray:
    """Return ``cos(2*pi*(x*w + b))`` for ``[..., 1]`` input."""
    return jnp.cos(2.0 * jnp.pi * (x * params.w + params.b))


def single_conditioning(
    si_input: jnp.ndarray,
    si_trunk: jnp.ndarray,
    t: jnp.ndarray,
    params: DiffusionConditioningParams,
    *,
    sigma_data: float,
    token_mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Condition the single representation on the trunk and the noise level.

    This is the single path only; :func:`diffusion_conditioning` runs both and is
    what the denoiser needs.

    Args:
        si_input: ``[..., N_token, c_s_input]`` input-embedder representation.
        si_trunk: ``[..., N_token, c_s]`` trunk representation.
        t: ``[...]`` noise level.
        params: mapped parameters.
        sigma_data: EDM data standard deviation.
        token_mask: ``[..., N_token]`` mask for the transitions.
        eps: layer norm epsilon.

    Returns:
        ``[..., N_token, c_s]`` conditioned single representation.
    """
    # Trunk first, then input — the concatenation order sets the weight layout.
    si = jnp.concatenate([si_trunk, si_input], axis=-1)
    si = linear(layer_norm(si, params.layer_norm_s, eps=eps), params.linear_s)

    n = 0.25 * jnp.log(t / sigma_data)
    n = fourier_embedding(n[..., None], params.fourier_emb)
    si = si + linear(layer_norm(n, params.layer_norm_n, eps=eps), params.linear_n)[
        ..., None, :
    ]

    for transition in params.transition_s:
        si = si + swiglu_transition(si, transition, mask=token_mask, eps=eps)
    return si


def pair_conditioning(
    batch: Mapping[str, jnp.ndarray],
    zij_trunk: jnp.ndarray,
    params: DiffusionConditioningParams,
    *,
    max_relative_idx: int,
    max_relative_chain: int,
    token_mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Condition the pair representation on relative positions.

    The relative-position encoding is concatenated onto the trunk pair embedding
    before the projection, so ``layer_norm_z`` spans ``num_relpos_dims + c_z``
    rather than ``c_z``.

    Returns:
        ``[..., N_token, N_token, c_z]`` conditioned pair representation.
    """
    relpos = relpos_complex(
        batch,
        max_relative_idx=max_relative_idx,
        max_relative_chain=max_relative_chain,
    ).astype(zij_trunk.dtype)
    zij = jnp.concatenate([zij_trunk, relpos], axis=-1)
    zij = linear(layer_norm(zij, params.layer_norm_z, eps=eps), params.linear_z)

    pair_mask = (
        None
        if token_mask is None
        else token_mask[..., :, None] * token_mask[..., None, :]
    )
    for transition in params.transition_z:
        zij = zij + swiglu_transition(zij, transition, mask=pair_mask, eps=eps)
    return zij


def diffusion_conditioning(
    batch: Mapping[str, jnp.ndarray],
    si_input: jnp.ndarray,
    si_trunk: jnp.ndarray,
    zij_trunk: jnp.ndarray,
    t: jnp.ndarray,
    params: DiffusionConditioningParams,
    *,
    sigma_data: float,
    max_relative_idx: int,
    max_relative_chain: int,
    token_mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run both conditioning paths, returning ``(si, zij)``.

    The denoiser needs the conditioned ``zij`` in two places -- the atom attention
    encoder and the diffusion transformer -- so passing the raw trunk pair
    representation to either is wrong.
    """
    si = single_conditioning(
        si_input,
        si_trunk,
        t,
        params,
        sigma_data=sigma_data,
        token_mask=token_mask,
        eps=eps,
    )
    zij = pair_conditioning(
        batch,
        zij_trunk,
        params,
        max_relative_idx=max_relative_idx,
        max_relative_chain=max_relative_chain,
        token_mask=token_mask,
        eps=eps,
    )
    return si, zij
