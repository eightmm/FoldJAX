"""What turns features into the pair and single representations.

Three of these are ordinary; the relative-position encoding is not, and its
concatenation order is the kind of thing that produces a working model with
quietly wrong geometry. Upstream lays it out as

    [rel_pos (2r+2) | rel_token (2r+2) | same_entity (1) | rel_chain (2c+2)]

with the scalar between the one-hots, and its chain branch inverted relative
to the other two -- `where(same_chain, out_of_range, clipped)` rather than the
`where(same_chain, clipped, out_of_range)` the other two use. Both are
reproduced rather than tidied.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models.esmfold2.models.primitives import layer_norm, linear
from foldjax.models.esmfold2.models.trunk import (
    msa_pair_weighted_averaging,
    outer_product_mean,
    transition,
    triangle_multiplicative,
)

Params = Mapping[str, jnp.ndarray]


def relative_position_encoding(
    residue_index: jnp.ndarray,
    asym_id: jnp.ndarray,
    sym_id: jnp.ndarray,
    entity_id: jnp.ndarray,
    token_index: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    n_residue_bins: int = 32,
    n_chain_bins: int = 2,
) -> jnp.ndarray:
    """`ResIdxAsymIdSymIdEntityIdEncoding`.

    Differences are `x[i] - x[j]` with `i` the row, clipped into `[0, 2r]` and
    pushed to the out-of-range bin `2r+1` when the pair fails the branch's
    condition.
    """
    dot = f"{prefix}." if prefix else ""
    r, c = n_residue_bins, n_chain_bins

    same_chain = asym_id[:, :, None] == asym_id[:, None, :]
    same_residue = residue_index[:, :, None] == residue_index[:, None, :]
    same_entity = entity_id[:, :, None] == entity_id[:, None, :]

    d_res = jnp.clip(
        residue_index[:, :, None] - residue_index[:, None, :] + r, 0, 2 * r
    )
    d_res = jnp.where(same_chain, d_res, 2 * r + 1)

    d_tok = jnp.clip(token_index[:, :, None] - token_index[:, None, :] + r, 0, 2 * r)
    d_tok = jnp.where(same_chain & same_residue, d_tok, 2 * r + 1)

    d_chain = jnp.clip(sym_id[:, :, None] - sym_id[:, None, :] + c, 0, 2 * c)
    # Inverted on purpose: same-chain pairs take the out-of-range bin here.
    d_chain = jnp.where(same_chain, 2 * c + 1, d_chain)

    features = jnp.concatenate(
        [
            jax.nn.one_hot(d_res, 2 * r + 2),
            jax.nn.one_hot(d_tok, 2 * r + 2),
            same_entity.astype(jnp.float32)[..., None],
            jax.nn.one_hot(d_chain, 2 * c + 2),
        ],
        axis=-1,
    )
    return linear(features, params, f"{dot}embed")


def single_to_pair(
    x: jnp.ndarray, params: Params, prefix: str = ""
) -> jnp.ndarray:
    """`SingleToPair`: product and difference, in that order.

    The difference is antisymmetric, so swapping the two halves would leave
    the shapes intact and the pair representation mirrored.
    """
    dot = f"{prefix}." if prefix else ""
    x = linear(x, params, f"{dot}downproject")
    outer = jnp.concatenate(
        [x[:, :, None, :] * x[:, None, :, :], x[:, :, None, :] - x[:, None, :, :]],
        axis=3,
    )
    hidden = linear(outer, params, f"{dot}output_mlp.0")
    hidden = jax.nn.gelu(hidden, approximate=False)
    return linear(hidden, params, f"{dot}output_mlp.2")


def row_attention_pooling(
    z: jnp.ndarray, mask: jnp.ndarray, params: Params, prefix: str = ""
) -> jnp.ndarray:
    """`RowAttentionPooling`: pool the pair rows into a single representation.

    The mask indexes the *second* token axis and the fill is -1e9, not -inf,
    so a fully masked row still produces a finite softmax.
    """
    dot = f"{prefix}." if prefix else ""
    scores = jnp.squeeze(linear(z, params, f"{dot}attn_proj"), axis=-1)
    scores = scores + jnp.where(mask[:, None, :].astype(bool), 0.0, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    pooled = jnp.einsum("bnm,bnmd->bnd", weights, z)
    return linear(pooled, params, f"{dot}out_proj")


def msa_encoder_block(
    msa: jnp.ndarray,
    pair: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    msa_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    is_final: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One `MSAEncoderBlock`.

    The final block has no MSA-side update at all -- upstream omits both
    submodules, so their keys are *absent* from the checkpoint rather than
    present and unused, and asking for them would raise.
    """
    dot = f"{prefix}." if prefix else ""
    pair = pair + outer_product_mean(
        msa, params, f"{dot}outer_product_mean", msa_mask=msa_mask
    )
    if not is_final:
        msa = msa + msa_pair_weighted_averaging(
            msa,
            pair,
            params,
            f"{dot}msa_pair_weighted_averaging",
            pair_mask=pair_mask,
        )
        msa = msa + transition(msa, params, f"{dot}msa_transition", residual=False)
    pair = pair + triangle_multiplicative(
        pair, params, f"{dot}tri_mul_out", outgoing=True, mask=pair_mask
    )
    pair = pair + triangle_multiplicative(
        pair, params, f"{dot}tri_mul_in", outgoing=False, mask=pair_mask
    )
    pair = pair + transition(pair, params, f"{dot}pair_transition", residual=False)
    return msa, pair


def msa_encoder(
    pair: jnp.ndarray,
    inputs: jnp.ndarray,
    msa_one_hot: jnp.ndarray,
    has_deletion: jnp.ndarray,
    deletion_value: jnp.ndarray,
    msa_mask: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    n_layers: int,
) -> jnp.ndarray:
    """`MSAEncoder`, returning the pair representation and discarding the MSA.

    Both embeddings are bias-free, so padded rows must already be zeroed by
    the caller -- upstream zeroes `msa_one_hot` against the mask before the
    call and relies on that here.
    """
    dot = f"{prefix}." if prefix else ""
    features = jnp.concatenate(
        [msa_one_hot, has_deletion[..., None], deletion_value[..., None]], axis=-1
    )
    msa = linear(features, params, f"{dot}embed")
    msa = msa + linear(inputs, params, f"{dot}project_inputs")[:, :, None, :]

    token_mask = msa_mask[:, :, 0].astype(bool)
    pair_mask = (token_mask[:, :, None] & token_mask[:, None, :]).astype(pair.dtype)

    for index in range(n_layers):
        msa, pair = msa_encoder_block(
            msa,
            pair,
            params,
            f"{dot}blocks.{index}",
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            is_final=index == n_layers - 1,
        )
    return pair


def inputs_embedder_tail(
    atom_representation: jnp.ndarray,
    res_type_one_hot: jnp.ndarray,
    profile: jnp.ndarray,
    deletion_mean: jnp.ndarray,
) -> jnp.ndarray:
    """The concatenation `InputsEmbedder` ends with: 384 + 33 + 33 + 1 = 451.

    The atom encoder does the work; this is the part that has to be in the
    right order, because `d_inputs` is a single width and every consumer
    slices it by position.
    """
    return jnp.concatenate(
        [atom_representation, res_type_one_hot, profile, deletion_mean[..., None]],
        axis=-1,
    )


__all__ = [
    "inputs_embedder_tail",
    "layer_norm",
    "msa_encoder",
    "msa_encoder_block",
    "relative_position_encoding",
    "row_attention_pooling",
    "single_to_pair",
]
