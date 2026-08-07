"""MSA leaves: outer product mean and pair-weighted averaging.

``OuterProductMean`` (AF2 Alg. 10 / AF3 Alg. 9) turns the MSA into a pair update.
``MSAPairWeightedAveraging`` (AF3 Alg. 10) updates the MSA using the pair
representation as attention weights — it is a weighted average, not
query/key attention, so there is no query projection and no scaling.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.attention import flatten_heads, split_heads
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    jax_sigmoid,
    layer_norm,
    linear,
)
from foldjax.models.openfold3.models.row_chunking import map_row_chunks
from foldjax.models.openfold3.models.triangle import permute_final_dims


class OuterProductMeanParams(NamedTuple):
    """Parameters for ``OuterProductMean``.

    ``linear_out`` is the one projection here that carries a bias
    (``opm_init``).
    """

    layer_norm: LayerNormParams
    linear_1: LinearParams
    linear_2: LinearParams
    linear_out: LinearParams


class MSAPairWeightedAveragingParams(NamedTuple):
    """Parameters for ``MSAPairWeightedAveraging``; every projection is bias-free."""

    layer_norm_m: LayerNormParams
    layer_norm_z: LayerNormParams
    linear_z: LinearParams
    linear_v: LinearParams
    linear_g: LinearParams
    linear_o: LinearParams


def outer_product_mean(
    m: jnp.ndarray,
    params: OuterProductMeanParams,
    *,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-3,
    layer_norm_eps: float = 1e-5,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Project the MSA to a pair update.

    Args:
        m: ``[..., N_seq, N_token, C_m]`` MSA embedding.
        params: mapped parameters.
        mask: ``[..., N_seq, N_token]`` MSA mask; ``None`` means all ones.
        eps: normalization epsilon; upstream's ``OuterProductMean.eps``.
        layer_norm_eps: layer norm epsilon.
        chunk_size: rows of the first token axis to project at a time. ``None``
            builds the whole outer product first, which is the largest single
            buffer this model allocates: ``[N_token, N_token, C, C]`` is
            ``C ** 2 / C_z`` times the pair update it becomes, eight times over
            at the released widths, and 34.6 GiB at 3012 tokens against 4.6 GiB
            for the result. Upstream chunks it by the same ``chunk_size`` it
            gives the pair stack.

    Returns:
        ``[..., N_token, N_token, C_z]`` pair update.
    """
    if mask is None:
        mask = jnp.ones(m.shape[:-1], dtype=m.dtype)

    normed = layer_norm(m, params.layer_norm, eps=layer_norm_eps)
    mask = mask[..., None]

    a = linear(normed, params.linear_1) * mask
    b = linear(normed, params.linear_2) * mask

    # [..., N_token, N_seq, C]
    a = jnp.swapaxes(a, -2, -3)
    b = jnp.swapaxes(b, -2, -3)

    def project(rows: jnp.ndarray) -> jnp.ndarray:
        # [..., rows, N_token, C, C] -> flattened -> C_z. Projecting inside the
        # block is the whole point: it is what keeps the C * C tensor from ever
        # existing at full width.
        outer = jnp.einsum("...bac,...dae->...bdce", rows, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        return linear(outer, params.linear_out)

    outer = map_row_chunks(project, a, chunk_size=chunk_size, row_axes=-3)

    # [..., N_token, N_token, 1], too small to be worth chunking alongside.
    norm = jnp.einsum("...abc,...adc->...bdc", mask, mask) + eps
    return outer / norm


def msa_pair_weighted_averaging(
    m: jnp.ndarray,
    z: jnp.ndarray,
    params: MSAPairWeightedAveragingParams,
    *,
    no_heads: int,
    mask: jnp.ndarray | None = None,
    inf: float = 1e9,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Update the MSA by averaging values with pair-derived weights.

    Args:
        m: ``[..., N_seq, N_token, C_m]`` MSA embedding.
        z: ``[..., N_token, N_token, C_z]`` pair embedding.
        params: mapped parameters.
        no_heads: head count.
        mask: ``[..., N_token, N_token]`` *pair* mask; ``None`` means all ones.
        inf: masking constant.
        eps: layer norm epsilon.

    Returns:
        ``[..., N_seq, N_token, C_m]`` MSA update.
    """
    if mask is None:
        mask = jnp.ones(z.shape[:-1], dtype=z.dtype)

    # [..., 1, 1, N_token, N_token]
    mask_bias = (inf * (mask - 1.0))[..., None, None, :, :]

    # [..., 1, H, N_token, N_token]
    weights = linear(layer_norm(z, params.layer_norm_z, eps=eps), params.linear_z)
    weights = jnp.expand_dims(permute_final_dims(weights, (2, 0, 1)), -4)
    weights = jax.nn.softmax(weights + mask_bias, axis=-1)

    normed = layer_norm(m, params.layer_norm_m, eps=eps)

    # [..., N_seq, H, N_token, C_hidden]
    value = jnp.swapaxes(split_heads(linear(normed, params.linear_v), no_heads), -2, -3)

    # [..., N_seq, N_token, H, C_hidden]
    out = jnp.einsum("...hqk,...hkc->...qhc", weights, value)

    gate = split_heads(jax_sigmoid(linear(normed, params.linear_g)), no_heads)
    return linear(flatten_heads(out * gate), params.linear_o)
