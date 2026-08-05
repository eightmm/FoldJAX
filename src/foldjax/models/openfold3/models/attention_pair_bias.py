"""Attention with pair bias (AF3 Algorithm 24).

Only the trunk configuration is ported: ``use_ada_layer_norm=False`` and
``gating=True``. The diffusion configuration adds AdaLN conditioning and an
output gate driven by the single representation; it belongs with the diffusion
module and is deliberately not folded in here.

Note that this layer's inner attention gives ``linear_q`` a bias
(``att_pair_bias_mha_init``), unlike every other ``Attention`` in the model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.openfold3.models.attention import AttentionParams, attention
from foldjax.models.openfold3.models.primitives import (
    AdaLNParams,
    LayerNormParams,
    LinearParams,
    adaln,
    jax_sigmoid,
    layer_norm,
    linear,
)
from foldjax.models.openfold3.models.triangle import permute_final_dims


class AttentionPairBiasParams(NamedTuple):
    """Parameters for a trunk ``AttentionPairBias``."""

    layer_norm_a: LayerNormParams
    layer_norm_z: LayerNormParams
    linear_z: LinearParams
    mha: AttentionParams


def attention_pair_bias(
    a: jnp.ndarray,
    z: jnp.ndarray,
    params: AttentionPairBiasParams,
    *,
    no_heads: int,
    mask: jnp.ndarray | None = None,
    inf: float = 1e9,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply attention over ``a`` biased by the pair representation ``z``.

    Args:
        a: ``[..., N, C_q]`` token or atom embedding.
        z: ``[..., N, N, C_z]`` pair embedding.
        params: mapped layer parameters.
        no_heads: attention head count.
        mask: ``[..., N]`` mask; ``None`` means all ones.
        inf: masking constant.
        eps: layer norm epsilon.

    Returns:
        ``[..., N, C_q]`` update. Upstream returns the update only.
    """
    a = layer_norm(a, params.layer_norm_a, eps=eps)

    if mask is None:
        mask = jnp.ones(a.shape[:-1], dtype=a.dtype)

    # [..., 1, 1, N]
    mask_bias = (inf * (mask - 1.0))[..., None, None, :]

    # [..., N, N, H] -> [..., H, N, N]
    pair_bias = linear(layer_norm(z, params.layer_norm_z, eps=eps), params.linear_z)
    pair_bias = permute_final_dims(pair_bias, (2, 0, 1))

    return attention(
        a, a, params.mha, no_heads=no_heads, biases=(mask_bias, pair_bias)
    )


class AdaAttentionPairBiasParams(NamedTuple):
    """Parameters for the AdaLN-conditioned ``AttentionPairBias``.

    The diffusion configuration replaces ``layer_norm_a`` with an ``AdaLN`` driven
    by the single representation, and adds an AdaLN-zero output gate
    (``linear_ada_out``, bias initialized to -2). ``layer_norm_z`` is scale-only
    here, unlike the trunk variant.
    """

    layer_norm_a: AdaLNParams
    linear_ada_out: LinearParams
    layer_norm_z: LayerNormParams
    linear_z: LinearParams
    mha: AttentionParams


def ada_attention_pair_bias(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: AdaAttentionPairBiasParams,
    *,
    no_heads: int,
    mask: jnp.ndarray | None = None,
    inf: float = 1e9,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply AdaLN-conditioned pair-biased attention (AF3 Alg. 24, diffusion).

    Args:
        a: ``[..., N, C_q]`` activation.
        s: ``[..., N, C_s]`` conditioning single representation.
        z: ``[..., N, N, C_z]`` pair embedding.
        params: mapped parameters.
        no_heads: attention head count.
        mask: ``[..., N]`` mask; ``None`` means all ones.
        inf: masking constant.
        eps: layer norm epsilon.

    Returns:
        ``[..., N, C_q]`` update.
    """
    a = adaln(a, s, params.layer_norm_a, eps=eps)

    if mask is None:
        mask = jnp.ones(a.shape[:-1], dtype=a.dtype)
    mask_bias = (inf * (mask - 1.0))[..., None, None, :]

    pair_bias = linear(layer_norm(z, params.layer_norm_z, eps=eps), params.linear_z)
    pair_bias = permute_final_dims(pair_bias, (2, 0, 1))

    a = attention(
        a, a, params.mha, no_heads=no_heads, biases=(mask_bias, pair_bias)
    )
    return jax_sigmoid(linear(s, params.linear_ada_out)) * a


class CrossAttentionPairBiasParams(NamedTuple):
    """Parameters for ``CrossAttentionPairBias`` (sequence-local atom attention).

    Query and key get *separate* norms, because they are different blockings of
    the same tensor. There is no ``layer_norm_z`` here: unlike the trunk and
    diffusion variants, the pair bias is projected from an already-blocked ``z``
    without a norm, which the stack applies once instead.

    ``layer_norm_a_q``/``layer_norm_a_k`` are ``AdaLN`` when conditioned and plain
    ``LayerNorm`` otherwise; ``linear_ada_out`` exists only in the former case.
    """

    layer_norm_a_q: LayerNormParams | AdaLNParams
    layer_norm_a_k: LayerNormParams | AdaLNParams
    linear_z: LinearParams
    mha: AttentionParams
    linear_ada_out: LinearParams | None = None


def cross_attention_pair_bias(
    a: jnp.ndarray,
    z: jnp.ndarray,
    params: CrossAttentionPairBiasParams,
    *,
    no_heads: int,
    n_query: int,
    n_key: int,
    mask: jnp.ndarray | None = None,
    s: jnp.ndarray | None = None,
    inf: float = 1e9,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply sequence-local (blocked) attention with a pair bias.

    Args:
        a: ``[..., N_atom, C_q]`` atom embedding.
        z: ``[..., N_blocks, N_query, N_key, C_z]`` blocked pair conditioning.
        params: mapped parameters.
        no_heads: attention head count.
        n_query: query block height.
        n_key: key window width.
        mask: ``[..., N_atom]`` atom mask; ``None`` means all ones.
        s: ``[..., N_atom, C_s]`` conditioning single rep, required when the
            norms are ``AdaLN``.
        inf: masking constant.
        eps: layer norm epsilon.

    Returns:
        ``[..., N_atom, C_q]`` update, unblocked and trimmed back to ``N_atom``.
    """
    from foldjax.models.openfold3.models.atom_blocks import single_rep_to_blocks

    conditioned = params.linear_ada_out is not None
    if conditioned and s is None:
        raise ValueError("s is required when the norms are AdaLN")

    n_atom, n_dim = a.shape[-2:]
    if mask is None:
        mask = jnp.ones(a.shape[:-1], dtype=a.dtype)

    a_q, a_k, block_mask = single_rep_to_blocks(
        a, mask, n_query=n_query, n_key=n_key
    )

    # [..., N_blocks, 1, N_query, N_key]
    mask_bias = (inf * (block_mask - 1.0))[..., None, :, :]
    pair_bias = permute_final_dims(linear(z, params.linear_z), (2, 0, 1))

    if conditioned:
        s_q, s_k, _ = single_rep_to_blocks(s, mask, n_query=n_query, n_key=n_key)
        a_q = adaln(a_q, s_q, params.layer_norm_a_q, eps=eps)
        a_k = adaln(a_k, s_k, params.layer_norm_a_k, eps=eps)
    else:
        a_q = layer_norm(a_q, params.layer_norm_a_q, eps=eps)
        a_k = layer_norm(a_k, params.layer_norm_a_k, eps=eps)

    out = attention(
        a_q, a_k, params.mha, no_heads=no_heads, biases=(mask_bias, pair_bias)
    )

    # Flatten the block axis back to atoms and drop the block padding.
    out = out.reshape((*out.shape[:-3], -1, n_dim))[..., :n_atom, :]

    if conditioned:
        out = jax_sigmoid(linear(s, params.linear_ada_out)) * out
    return out
