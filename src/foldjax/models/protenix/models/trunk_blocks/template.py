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

    asym_id = input_feature_dict["asym_id"]
    n_token = asym_id.shape[-1]
    dtype = input_feature_dict["template_aatype"].dtype
    dtype = jnp.float32 if not jnp.issubdtype(dtype, jnp.floating) else dtype
    if pair_mask is not None:
        dtype = pair_mask.dtype

    def pair_feature(name: str, *trailing: int) -> jnp.ndarray:
        """One template's slice of a pair feature, or in-graph zeros.

        The CLI drops a template array whose every value is zero -- the common
        no-template job ships 5.95 GiB of literal zeros as entry parameters
        otherwise, and XLA keeps entry parameters resident for the whole run.
        Rebuilt here as broadcast zeros, the values are identical and the
        algebraic simplifier folds them straight through the linears
        (dot(0, W) = 0, x + 0 = x) instead of storing them.
        """
        value = input_feature_dict.get(name)
        if value is not None:
            return value[template_id]
        return jnp.zeros((n_token, n_token, *trailing), dtype)

    dgram = pair_feature("template_distogram", 39)
    if pair_mask is None:
        pair_mask = jnp.ones(dgram.shape[:-1], dtype=dtype)
    else:
        pair_mask = pair_mask.astype(dtype)
    multichain_mask = (asym_id[..., :, None] == asym_id[..., None, :]).astype(dtype)
    pair_mask = pair_mask * multichain_mask

    pseudo_beta_mask = (
        pair_feature("template_pseudo_beta_mask").astype(dtype) * pair_mask
    )
    aatype = input_feature_dict["template_aatype"][template_id]
    aatype = jnp.eye(32, dtype=dtype)[aatype]
    aatype_i = jnp.broadcast_to(aatype[..., None, :, :], dgram.shape[:-1] + (32,))
    aatype_j = jnp.broadcast_to(aatype[..., :, None, :], dgram.shape[:-1] + (32,))
    unit_vector = (
        pair_feature("template_unit_vector", 3).astype(dtype)
        * pair_mask[..., None]
    )
    backbone_mask = (
        pair_feature("template_backbone_frame_mask").astype(dtype) * pair_mask
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
    for template_id in range(num_templates):
        u = u + single_template_embedding(
            input_feature_dict,
            z_norm,
            pair_mask,
            template_id,
            params,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            triangle_attention_backend=triangle_attention_backend,
        )
    u = u / (1e-7 + num_templates)
    return linear(jnp.maximum(u, 0.0), params.linear_u)
