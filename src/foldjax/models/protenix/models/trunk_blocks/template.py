"""Template trunk blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.protenix.data.template_features import (
    TEMPLATE_DISTOGRAM_BINS,
    ZERO_TEMPLATE_GEOMETRY_MARKER,
    has_compact_zero_template_geometry,
)
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


def _geometry(
    input_feature_dict: dict[str, jnp.ndarray],
    name: str,
    template_id: int,
    compact: tuple[jnp.ndarray, int] | None,
    trailing: tuple[int, ...],
) -> jnp.ndarray:
    """One template geometry operand: the stored row, or the rebuilt zero."""

    if compact is None:
        return input_feature_dict[name][template_id]
    zero, n_token = compact
    return jnp.broadcast_to(zero, (n_token, n_token, *trailing))


def _zero_template_geometry(
    input_feature_dict: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, int] | None:
    """The rebuilt-zero operand and token count, or ``None`` for dense inputs.

    A query with no template hits hands the device four all-zero quadratic
    tensors -- 5.9 GB at 4,100 tokens over the two deduplicated survivors --
    that exist only to be multiplied by a mask.
    :func:`~foldjax.models.protenix.data.template_features.compact_zero_template_geometry`
    drops them on the host and leaves this scalar behind; every operand below
    is a broadcast of it, so the arithmetic that follows is the arithmetic the
    dense path ran, on values that are bitwise the ones it received.

    The scalar stays a runtime operand rather than becoming ``jnp.zeros``: a
    literal zero would let a compiler fold ``0 * x`` and lose the dense path's
    ``0 * NaN`` and ``0 * Inf``.

    Its dtype is whatever reached the trunk, not float32. ``trunk_dtype``
    narrows every floating leaf of the feature tree by kind, so at ``bf16``
    this scalar arrives narrowed exactly as the four arrays it stands in for --
    and ``dgram.dtype``, which sets the precision of the 108-wide concatenation
    below, comes out the same on both paths. The float32 storage contract is
    enforced on the host instead, by
    :func:`~foldjax.models.protenix.data.template_features.validate_zero_template_geometry`.
    """

    if not has_compact_zero_template_geometry(input_feature_dict):
        return None
    zero = jnp.asarray(input_feature_dict[ZERO_TEMPLATE_GEOMETRY_MARKER])
    if zero.shape != ():
        raise ValueError("Protenix zero-template geometry marker must be a scalar")
    if not jnp.issubdtype(zero.dtype, jnp.floating):
        raise ValueError(
            "Protenix zero-template geometry marker must be a floating scalar"
        )
    if not isinstance(zero, jax.core.Tracer) and (
        bool(zero != 0.0) or bool(jnp.signbit(zero))
    ):
        raise ValueError("Protenix zero-template geometry marker must be exactly +0.0")
    n_token = int(input_feature_dict["template_aatype"].shape[-1])
    return zero, n_token


def template_pair_features(
    input_feature_dict: dict[str, jnp.ndarray],
    template_id: int,
    pair_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Build one Protenix template pair-feature tensor."""

    compact = _zero_template_geometry(input_feature_dict)
    if compact is None:
        dgram = input_feature_dict["template_distogram"][template_id]
        n_token = dgram.shape[-3]
    else:
        zero, n_token = compact
        dgram = jnp.broadcast_to(zero, (n_token, n_token, TEMPLATE_DISTOGRAM_BINS))
    dtype = dgram.dtype
    if pair_mask is None:
        pair_mask = jnp.ones(dgram.shape[:-1], dtype=dtype)
    else:
        pair_mask = pair_mask.astype(dtype)
    asym_id = input_feature_dict["asym_id"]
    multichain_mask = (asym_id[..., :, None] == asym_id[..., None, :]).astype(dtype)
    pair_mask = pair_mask * multichain_mask

    pseudo_beta_mask = (
        _geometry(
            input_feature_dict, "template_pseudo_beta_mask", template_id, compact, ()
        ).astype(dtype)
        * pair_mask
    )
    aatype = input_feature_dict["template_aatype"][template_id]
    aatype = jnp.eye(32, dtype=dtype)[aatype]
    aatype_i = jnp.broadcast_to(aatype[..., None, :, :], dgram.shape[:-1] + (32,))
    aatype_j = jnp.broadcast_to(aatype[..., :, None, :], dgram.shape[:-1] + (32,))
    unit_vector = (
        _geometry(
            input_feature_dict, "template_unit_vector", template_id, compact, (3,)
        ).astype(dtype)
        * pair_mask[..., None]
    )
    backbone_mask = (
        _geometry(
            input_feature_dict,
            "template_backbone_frame_mask",
            template_id,
            compact,
            (),
        ).astype(dtype)
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
