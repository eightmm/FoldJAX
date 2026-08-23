"""Template trunk blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    PairformerStackParams,
    pairformer_stack,
)


class TemplateEmbedderParams(NamedTuple):
    """Parameters for Protenix ``TemplateEmbedder``."""

    linear_z: LinearParams
    layernorm_z: LayerNormParams
    linear_a: LinearParams
    pairformer_stack: PairformerStackParams
    layernorm_v: LayerNormParams
    linear_u: LinearParams


def template_pair_features(
    input_feature_dict: dict[str, jnp.ndarray],
    template_id: int,
    pair_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Build one Protenix template pair-feature tensor."""

    dgram = input_feature_dict["template_distogram"][template_id]
    n_token = dgram.shape[-3]
    dtype = dgram.dtype
    if pair_mask is None:
        pair_mask = jnp.ones(dgram.shape[:-1], dtype=dtype)
    else:
        pair_mask = pair_mask.astype(dtype)
    asym_id = input_feature_dict["asym_id"]
    multichain_mask = (asym_id[..., :, None] == asym_id[..., None, :]).astype(dtype)
    pair_mask = pair_mask * multichain_mask

    pseudo_beta_mask = (
        input_feature_dict["template_pseudo_beta_mask"][template_id].astype(dtype)
        * pair_mask
    )
    aatype = input_feature_dict["template_aatype"][template_id]
    aatype = jnp.eye(32, dtype=dtype)[aatype]
    aatype_i = jnp.broadcast_to(aatype[..., None, :, :], dgram.shape[:-1] + (32,))
    aatype_j = jnp.broadcast_to(aatype[..., :, None, :], dgram.shape[:-1] + (32,))
    unit_vector = (
        input_feature_dict["template_unit_vector"][template_id].astype(dtype)
        * pair_mask[..., None]
    )
    backbone_mask = (
        input_feature_dict["template_backbone_frame_mask"][template_id].astype(dtype)
        * pair_mask
    )

    return jnp.concatenate(
        [
            dgram * pair_mask[..., None],
            pseudo_beta_mask[..., None],
            aatype_i,
            aatype_j,
            unit_vector,
            backbone_mask[..., None],
        ],
        axis=-1,
    ).reshape((n_token, n_token, 108))


def single_template_embedding(
    input_feature_dict: dict[str, jnp.ndarray],
    z_norm: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    template_id: int,
    params: TemplateEmbedderParams,
    *,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    triangle_attention_backend: str | None = None,
) -> jnp.ndarray:
    """Apply one-template Protenix template embedding path."""

    at = template_pair_features(input_feature_dict, template_id, pair_mask)
    v = linear(z_norm, params.linear_z) + linear(at, params.linear_a)
    _, v = pairformer_stack(
        None,
        v,
        pair_mask,
        params.pairformer_stack,
        use_scan=False,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        triangle_attention_backend=triangle_attention_backend,
    )
    return layer_norm(v, params.layernorm_v)


def template_embedder(
    input_feature_dict: dict[str, jnp.ndarray],
    z: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    params: TemplateEmbedderParams,
    *,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    triangle_attention_backend: str | None = None,
) -> jnp.ndarray:
    """Apply Protenix ``TemplateEmbedder`` in inference mode."""

    has_templates = "template_aatype" in input_feature_dict
    if not has_templates or not params.pairformer_stack.blocks:
        return jnp.zeros_like(z)
    num_templates = int(input_feature_dict["template_aatype"].shape[0])
    z_norm = layer_norm(z, params.layernorm_z)
    u = jnp.zeros(z.shape[:-1] + (params.linear_z.weight.shape[0],), dtype=z.dtype)
    # Identical template rows are deduplicated on the host, and each survivor
    # carries how many it stands for. Weighting by that and dividing by their
    # sum is the same average over the original rows, computed once per
    # *distinct* template instead of once per row. Absent -- legacy archives,
    # and any caller that did not deduplicate -- every row stands for itself.
    multiplicity = input_feature_dict.get("template_multiplicity")
    for template_id in range(num_templates):
        contribution = single_template_embedding(
            input_feature_dict,
            z_norm,
            pair_mask,
            template_id,
            params,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            triangle_attention_backend=triangle_attention_backend,
        )
        if multiplicity is not None:
            contribution = contribution * multiplicity[template_id].astype(
                contribution.dtype
            )
        u = u + contribution
    total = (
        num_templates
        if multiplicity is None
        else jnp.sum(multiplicity).astype(u.dtype)
    )
    u = u / (1e-7 + total)
    return linear(jnp.maximum(u, 0.0), params.linear_u)
