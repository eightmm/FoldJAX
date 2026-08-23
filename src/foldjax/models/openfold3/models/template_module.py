"""Template pair stack (AF2 Algorithm 16).

``TemplatePairBlock`` subclasses ``PairBlock`` and adds one option: whether the
multiplicative updates run before triangle attention (``tri_mul_first``). The
parameter layout is therefore identical to a pair block, so the same mapper is
reused. The stack applies a final layer norm that a plain pair stack does not.

Upstream loops over the template axis and concatenates, which is a memory
strategy rather than a different function; the vectorized form here computes the
same values because every template shares one set of weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.data.featurize import (
    _ZERO_TEMPLATE_PAIR_FEATURES,
    _ZERO_TEMPLATE_PAIR_MARKER,
)
from foldjax.models.openfold3.models.pair_block import PairBlockParams, pair_block
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)
from foldjax.models.openfold3.models.stacking import can_scan, scan_stack


class TemplatePairStackParams(NamedTuple):
    """Parameters for a ``TemplatePairStack``.

    ``layer_norm`` is applied once after every block, unlike ``PairFormerStack``.
    """

    blocks: tuple[PairBlockParams, ...]
    layer_norm: LayerNormParams


def template_pair_stack(
    t: jnp.ndarray,
    params: TemplatePairStackParams,
    *,
    mask: jnp.ndarray,
    no_heads: int,
    tri_mul_first: bool = True,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run every template pair block, then the final layer norm.

    Args:
        t: ``[..., N_templ, N_token, N_token, C_t]`` template pair embedding, or
            the same without the template axis.
        params: mapped stack parameters.
        mask: pair mask broadcasting to ``t.shape[:-1]``.
        no_heads: head count for triangle attention.
        tri_mul_first: multiplicative updates before triangle attention.
        inf: masking constant.
        mask_transition: upstream's ``_mask_trans``.
        eps: layer norm epsilon.

    Returns:
        The template pair embedding, same shape as ``t``.
    """
    settings = dict(
        pair_mask=mask,
        no_heads_pair=no_heads,
        inf=inf,
        mask_transition=mask_transition,
        tri_mul_first=tri_mul_first,
        eps=eps,
        chunk_size=chunk_size,
    )
    if can_scan(params.blocks):
        t = scan_stack(
            lambda carry, block: pair_block(carry, block, **settings), t, params.blocks
        )
    else:
        for block in params.blocks:
            t = pair_block(t, block, **settings)
    return layer_norm(t, params.layer_norm, eps=eps)

class TemplatePairEmbedderParams(NamedTuple):
    """Parameters for ``TemplatePairEmbedderAllAtom``.

    Every projection is bias-free; only ``layer_norm_z`` has an offset.
    """

    dgram_linear: LinearParams
    aatype_linear_1: LinearParams
    aatype_linear_2: LinearParams
    pseudo_beta_mask_linear: LinearParams
    x_linear: LinearParams
    y_linear: LinearParams
    z_linear: LinearParams
    backbone_mask_linear: LinearParams
    layer_norm_z: LayerNormParams
    linear_z: LinearParams


class TemplateEmbedderParams(NamedTuple):
    """Parameters for ``TemplateEmbedderAllAtom``."""

    template_pair_embedder: TemplatePairEmbedderParams
    template_pair_stack: TemplatePairStackParams
    linear_t: LinearParams


def template_pair_embedder(
    batch: Mapping[str, jnp.ndarray],
    z: jnp.ndarray,
    params: TemplatePairEmbedderParams,
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Embed the template features into ``[..., N_templ, N_token, N_token, C_t]``.

    Both pairwise masks are additionally restricted to within-chain pairs, so a
    template never contributes across a chain boundary.
    """
    restype = batch["template_restype"].astype(z.dtype)
    n_token = restype.shape[-2]
    # ti varies along the i axis, tj along j; both broadcast to the pair shape.
    restype_ti = jnp.broadcast_to(
        restype[..., :, None, :], (*restype.shape[:-1], n_token, restype.shape[-1])
    )
    restype_tj = jnp.broadcast_to(
        restype[..., None, :, :], (*restype.shape[:-2], n_token, *restype.shape[-2:])
    )

    # Unknown mapping keys have historically been ignored by this low-level API.
    # Treat the marker as provenance only for the representation the host helper
    # actually emits: marker present and every replaced dense field absent.  A
    # direct caller that happens to carry the private key beside dense geometry
    # must continue to use that geometry rather than silently discarding it.
    compact_zero_pairs = _ZERO_TEMPLATE_PAIR_MARKER in batch and all(
        name not in batch for name in _ZERO_TEMPLATE_PAIR_FEATURES
    )
    if compact_zero_pairs:
        marker = jnp.asarray(batch[_ZERO_TEMPLATE_PAIR_MARKER]).reshape(())

        def zero_pair_projection(projection: LinearParams) -> jnp.ndarray:
            # The marker remains dynamic so XLA cannot replace this with a
            # constant zero and erase 0 * NaN/Inf parameter semantics.  Evaluate
            # the historical reduction width once, then broadcast its result.
            zero_input = jnp.broadcast_to(marker, (projection.weight.shape[-1],))
            projected = linear(zero_input, projection)
            return jnp.broadcast_to(
                projected,
                (*restype.shape[:-2], n_token, n_token, projected.shape[-1]),
            )

        a = zero_pair_projection(params.dgram_linear)
        a = a + zero_pair_projection(params.pseudo_beta_mask_linear)
    else:
        asym_id = batch["asym_id"]
        # [..., 1, N_token, N_token, 1] so it broadcasts over templates/channels.
        same_chain = (asym_id[..., :, None] == asym_id[..., None, :]).astype(z.dtype)
        same_chain = same_chain[..., None, :, :, None]

        pseudo_beta = batch["template_pseudo_beta_mask"]
        pseudo_beta_pair = (
            pseudo_beta[..., :, None] * pseudo_beta[..., None, :]
        )[..., None] * same_chain
        backbone = batch["template_backbone_frame_mask"]
        backbone_pair = (
            backbone[..., :, None] * backbone[..., None, :]
        )[..., None] * same_chain

        # The unit vector's three components each get their own projection.
        unit_vector = batch["template_unit_vector"]
        x, y, w = (unit_vector[..., index] for index in range(3))

        a = linear(batch["template_distogram"], params.dgram_linear)
        a = a + linear(pseudo_beta_pair, params.pseudo_beta_mask_linear)
    a = a + linear(restype_ti, params.aatype_linear_1)
    a = a + linear(restype_tj, params.aatype_linear_2)
    if compact_zero_pairs:
        a = a + zero_pair_projection(params.x_linear)
        a = a + zero_pair_projection(params.y_linear)
        a = a + zero_pair_projection(params.z_linear)
        a = a + zero_pair_projection(params.backbone_mask_linear)
    else:
        a = a + linear(x[..., None], params.x_linear)
        a = a + linear(y[..., None], params.y_linear)
        a = a + linear(w[..., None], params.z_linear)
        a = a + linear(backbone_pair, params.backbone_mask_linear)

    z_embedded = linear(layer_norm(z, params.layer_norm_z, eps=eps), params.linear_z)
    return z_embedded[..., None, :, :, :] + a


def template_embedder(
    batch: Mapping[str, jnp.ndarray],
    z: jnp.ndarray,
    params: TemplateEmbedderParams,
    *,
    pair_mask: jnp.ndarray,
    no_heads: int,
    tri_mul_first: bool = True,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    scan_templates: bool = True,
) -> jnp.ndarray:
    """Embed templates into a pair update, ``[..., N_token, N_token, C_z]``.

    The template axis is averaged after the stack, then passed through ReLU and
    one projection. The caller adds the result to ``z``.

    ``scan_templates`` runs the pair stack on one template at a time and accumulates
    the sum, which is what upstream does -- ``TemplatePairBlock.forward`` loops the
    template axis internally. Running all four at once is mathematically the same,
    because templates do not interact until this average, but it makes every
    intermediate in the stack four times larger. Measured on the trunk alone, this
    stage was the dominant consumer by a wide margin: 23.17 GiB at 574 tokens and
    69.98 GiB at 832, against 18.05 GiB for the 48-block Pairformer stack.
    """
    t = template_pair_embedder(batch, z, params.template_pair_embedder, eps=eps)
    n_templ = t.shape[-4]
    template_weights = batch.get("template_padding_mask")
    if template_weights is None:
        # Default-off and legacy archives retain the historical behaviour: all
        # rows in their existing storage participate, including chemically
        # empty rows in the released fixed-width template axis.
        template_weights = jnp.ones(t.shape[:-3], dtype=t.dtype)
    else:
        template_weights = jnp.asarray(template_weights, dtype=t.dtype)
        expected = t.shape[:-3]
        if template_weights.shape != expected:
            raise ValueError(
                "template_padding_mask must have shape "
                f"{expected}, got {template_weights.shape}"
            )
    denominator = jnp.clip(jnp.sum(template_weights, axis=-1), min=1.0)
    denominator = denominator[..., None, None, None]

    settings = dict(
        no_heads=no_heads,
        tri_mul_first=tri_mul_first,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        chunk_size=chunk_size,
    )

    if scan_templates and n_templ > 1:
        # Templates are independent until the average below, so summing them one at
        # a time is exact and keeps only one template's intermediates alive.
        leading = jnp.moveaxis(t, -4, 0)
        leading_weights = jnp.moveaxis(template_weights, -1, 0)

        def accumulate(
            total: jnp.ndarray, item: tuple[jnp.ndarray, jnp.ndarray]
        ):
            one, weight = item
            updated = template_pair_stack(
                one, params.template_pair_stack, mask=pair_mask, **settings
            )
            return total + updated * weight[..., None, None, None], None

        total, _ = jax.lax.scan(
            accumulate,
            jnp.zeros_like(leading[0]),
            (leading, leading_weights),
        )
        t = total / denominator
    else:
        t = template_pair_stack(
            t,
            params.template_pair_stack,
            mask=pair_mask[..., None, :, :],
            **settings,
        )
        weights = template_weights[..., :, None, None, None]
        t = jnp.sum(t * weights, axis=-4) / denominator
    return linear(jax.nn.relu(t), params.linear_t)
