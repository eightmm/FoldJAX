"""The pair stack block shared by the Pairformer and Evoformer.

Mirrors ``openfold3.core.model.latent.base_blocks.PairBlock``. Dropout is
identity at inference, so the residual structure reduces to five sequential
updates.

Two details are easy to get wrong and are pinned by the parity gate:

* ``tri_att_end`` is built with ``starting=True`` upstream, not
  ``starting=False``. The caller transposes ``z`` around it instead.
* The block returns the *updated* ``z``, not an update to be added. Each
  sub-layer's output is added internally.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.openfold3.models.primitives import (
    SwiGLUTransitionParams,
    swiglu_transition,
)
from foldjax.models.openfold3.models.row_chunking import map_row_chunks
from foldjax.models.openfold3.models.triangle import (
    TriangleMultiplicationParams,
    triangle_multiplication,
)
from foldjax.models.openfold3.models.triangle_attention import (
    TriangleAttentionParams,
    triangle_attention,
)


class PairBlockParams(NamedTuple):
    """Parameters for ``PairBlock`` with a SwiGLU transition.

    The ReLU transition variant (AF2) and the fused triangular multiplication
    variant are not mapped; only the AF3 SwiGLU / non-fused layout is.
    """

    tri_mul_out: TriangleMultiplicationParams
    tri_mul_in: TriangleMultiplicationParams
    tri_att_start: TriangleAttentionParams
    tri_att_end: TriangleAttentionParams
    pair_transition: SwiGLUTransitionParams


def tri_mul_out_in(
    z: jnp.ndarray,
    params: PairBlockParams,
    *,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply the outgoing then incoming triangular multiplicative updates."""
    z = z + triangle_multiplication(
        z, params.tri_mul_out, outgoing=True, mask=pair_mask, eps=eps
    )
    return z + triangle_multiplication(
        z, params.tri_mul_in, outgoing=False, mask=pair_mask, eps=eps
    )


def tri_att_start_end(
    z: jnp.ndarray,
    params: PairBlockParams,
    *,
    pair_mask: jnp.ndarray,
    no_heads_pair: int,
    inf: float = 1e9,
    eps: float = 1e-5,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply the starting then ending triangle attention layers.

    Both layers are starting-node modules upstream; the transposes around the
    second one are what make it an ending-node update.
    """
    z = z + triangle_attention(
        z,
        params.tri_att_start,
        no_heads=no_heads_pair,
        mask=pair_mask,
        inf=inf,
        eps=eps,
        chunk_size=chunk_size,
    )
    z = jnp.swapaxes(z, -2, -3)
    z = z + triangle_attention(
        z,
        params.tri_att_end,
        no_heads=no_heads_pair,
        mask=jnp.swapaxes(pair_mask, -1, -2),
        inf=inf,
        eps=eps,
        chunk_size=chunk_size,
    )
    return jnp.swapaxes(z, -2, -3)


def pair_block(
    z: jnp.ndarray,
    params: PairBlockParams,
    *,
    pair_mask: jnp.ndarray,
    no_heads_pair: int,
    inf: float = 1e9,
    mask_transition: bool = True,
    tri_mul_first: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply one pair block.

    Args:
        z: ``[..., N, N, C_z]`` pair representation.
        params: mapped block parameters.
        pair_mask: ``[..., N, N]`` pair mask.
        no_heads_pair: head count for both triangle attention layers.
        inf: triangle attention masking constant.
        mask_transition: upstream's ``_mask_trans``; ``True`` masks the
            transition output, which is the default.
        tri_mul_first: run the multiplicative updates before triangle attention.
            ``PairBlock`` is always ``True``; ``TemplatePairBlock`` exposes it as
            a configuration option.
        eps: layer norm epsilon.

    Returns:
        ``[..., N, N, C_z]`` updated pair representation.
    """
    if tri_mul_first:
        z = tri_mul_out_in(z, params, pair_mask=pair_mask, eps=eps)
        z = tri_att_start_end(
            z,
            params,
            pair_mask=pair_mask,
            no_heads_pair=no_heads_pair,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
        )
    else:
        z = tri_att_start_end(
            z,
            params,
            pair_mask=pair_mask,
            no_heads_pair=no_heads_pair,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
        )
        z = tri_mul_out_in(z, params, pair_mask=pair_mask, eps=eps)

    # SwiGLU widens the pair representation to ``4 * C_z`` twice before the output
    # projection reads the product. Rows are independent, so ``chunk_size`` caps that
    # widened tensor without changing a value -- the same knob the triangle attention
    # above uses, and for the same reason.
    if mask_transition:
        return z + map_row_chunks(
            lambda rows, mask: swiglu_transition(
                rows, params.pair_transition, mask=mask, eps=eps
            ),
            z,
            pair_mask,
            chunk_size=chunk_size,
            row_axes=(-3, -2),
        )
    return z + map_row_chunks(
        lambda rows: swiglu_transition(
            rows, params.pair_transition, mask=None, eps=eps
        ),
        z,
        chunk_size=chunk_size,
    )
