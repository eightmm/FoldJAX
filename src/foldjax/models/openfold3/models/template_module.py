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
    glu_backend: str = "xla",
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
        glu_backend=glu_backend,
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


def _project_template_restype(
    restype: jnp.ndarray,
    aatype_linear_1: LinearParams,
    aatype_linear_2: LinearParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project residue types before broadcasting them over token pairs."""
    restype_ti = linear(restype, aatype_linear_1)[..., :, None, :]
    restype_tj = linear(restype, aatype_linear_2)[..., None, :, :]
    return restype_ti, restype_tj


#: Where the template axis sits in each feature the pair embedder reads, as a
#: negative index so an optional leading batch axis needs no special case.
#: ``template_padding_mask`` is deliberately absent: the weighted average owns
#: that one and the embedder never reads it.
_TEMPLATE_FEATURE_AXES = {
    "template_restype": -3,
    "template_pseudo_beta_mask": -2,
    "template_backbone_frame_mask": -2,
    "template_distogram": -4,
    "template_unit_vector": -4,
}


def _embed_pair_state(
    z: jnp.ndarray,
    params: TemplatePairEmbedderParams,
    *,
    eps: float,
) -> jnp.ndarray:
    """Project the pair state: the one term of the embedding no template varies."""
    return linear(layer_norm(z, params.layer_norm_z, eps=eps), params.linear_z)


def _template_pair_features(
    batch: Mapping[str, jnp.ndarray],
    z_embedded: jnp.ndarray,
    params: TemplatePairEmbedderParams,
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """The template-dependent half of the embedding, given an already-projected ``z``.

    ``dtype`` is the pair state's own dtype rather than ``z_embedded.dtype``:
    the projection may widen, and the two casts below have always read the
    dtype ``z`` arrived in.
    """
    restype = batch["template_restype"].astype(dtype)
    n_token = restype.shape[-2]
    # Project each token once, then let the additions broadcast over the other
    # pair axis. This preserves the two historical dots and their add order
    # without presenting either dot with a quadratic residue-type operand.
    restype_ti, restype_tj = _project_template_restype(
        restype,
        params.aatype_linear_1,
        params.aatype_linear_2,
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
        same_chain = (asym_id[..., :, None] == asym_id[..., None, :]).astype(dtype)
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
    a = a + restype_ti
    a = a + restype_tj
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

    return z_embedded[..., None, :, :, :] + a


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
    return _template_pair_features(
        batch,
        _embed_pair_state(z, params, eps=eps),
        params,
        dtype=z.dtype,
    )


def _one_template_batch(
    batch: Mapping[str, jnp.ndarray],
    index: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Select one template from every feature that has a template axis.

    The selection is a dynamic slice of the caller's own arrays rather than a
    scan ``xs``: an ``xs`` has to be transposed template-major first, and the
    quadratic template features are far larger than the embedding this scan
    exists to shrink -- at 3,012 tokens the distogram alone is 5.26 GiB of
    float32 against the 2.15 GiB embedding. Slicing in the body leaves them as
    program arguments, which the program already holds.

    ``keepdims`` retains a length-one template axis so the embedder runs its
    stacked body unchanged at ``N_templ == 1``, and a single template cannot
    take a different arithmetic path from the all-at-once form. Keys the
    embedder does not read pass through untouched, which is what keeps the
    compact zero-pair marker's provenance test reading the key set it always
    has.
    """
    return {
        **batch,
        **{
            name: jax.lax.dynamic_index_in_dim(
                batch[name], index, axis=axis, keepdims=True
            )
            for name, axis in _TEMPLATE_FEATURE_AXES.items()
            if name in batch
        },
    }


def _template_weights(
    batch: Mapping[str, jnp.ndarray],
    *,
    prefix: tuple[int, ...],
    n_templ: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Per-template weights for the reduction, validated against the embedding."""
    template_weights = batch.get("template_padding_mask")
    expected = (*prefix, n_templ)
    if template_weights is None:
        # Default-off and legacy archives retain the historical behaviour: all
        # rows in their existing storage participate, including chemically
        # empty rows in the released fixed-width template axis.
        return jnp.ones(expected, dtype=dtype)
    template_weights = jnp.asarray(template_weights, dtype=dtype)
    if template_weights.shape != expected:
        raise ValueError(
            "template_padding_mask must have shape "
            f"{expected}, got {template_weights.shape}"
        )
    return template_weights


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
    glu_backend: str = "xla",
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

    The pair embedding is built inside that scan rather than ahead of it. Only
    the pair-state projection is template-independent, so hoisting just that one
    term leaves each step slicing its own template out of the raw features and
    materialising one ``[..., N_token, N_token, C_t]`` embedding instead of the
    whole ``[..., N_templ, N_token, N_token, C_t]`` stack. At 3,012 tokens and
    the released four-row template axis that replaces 8.58 GiB of float32 with
    2.15 GiB, against a hoisted projection of the same 2.15 GiB now live for
    the whole scan.

    This is not bit-exact on the dense path. Every term is unchanged -- each one
    is bitwise equal to the all-at-once form when materialised -- but XLA fuses
    the eight projections into the addition chain differently once the leading
    template extent is one, and the CPU result moves by a single unit in the
    last place on roughly a ninth of the elements from about 64 tokens up.
    Fencing the projections restores exact equality, which is what identifies
    the cause; it is not a usable remedy, because the fence materialises all
    eight. The error against a float64 reference is unchanged either way. The
    compact zero-template path stays bitwise at every size, having no
    quadratic dot to fuse.
    """
    embedder = params.template_pair_embedder
    z_embedded = _embed_pair_state(z, embedder, eps=eps)
    # The compact zero-pair form drops every dense field but this one, so the
    # residue types are the only feature that always carries the template axis.
    n_templ = batch["template_restype"].shape[-3]

    settings = dict(
        no_heads=no_heads,
        tri_mul_first=tri_mul_first,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        chunk_size=chunk_size,
        glu_backend=glu_backend,
    )

    if scan_templates and n_templ > 1:
        # Templates are independent until the average below, so summing them one at
        # a time is exact and keeps only one template's intermediates alive.
        def embed_one(index: jnp.ndarray) -> jnp.ndarray:
            one = _template_pair_features(
                _one_template_batch(batch, index),
                z_embedded,
                embedder,
                dtype=z.dtype,
            )
            return jnp.squeeze(one, -4)

        # The carry's shape and dtype come from the embedding rather than from
        # `z`, which the projection above may have widened.
        embedded = jax.eval_shape(embed_one, jax.ShapeDtypeStruct((), jnp.int32))
        template_weights = _template_weights(
            batch,
            prefix=embedded.shape[:-3],
            n_templ=n_templ,
            dtype=embedded.dtype,
        )
        denominator = jnp.clip(jnp.sum(template_weights, axis=-1), min=1.0)
        denominator = denominator[..., None, None, None]
        leading_weights = jnp.moveaxis(template_weights, -1, 0)

        def accumulate(total: jnp.ndarray, item: tuple[jnp.ndarray, jnp.ndarray]):
            index, weight = item
            updated = template_pair_stack(
                embed_one(index), params.template_pair_stack, mask=pair_mask, **settings
            )
            return total + updated * weight[..., None, None, None], None

        total, _ = jax.lax.scan(
            accumulate,
            jnp.zeros(embedded.shape, embedded.dtype),
            (jnp.arange(n_templ, dtype=jnp.int32), leading_weights),
        )
        t = total / denominator
    else:
        t = _template_pair_features(batch, z_embedded, embedder, dtype=z.dtype)
        template_weights = _template_weights(
            batch, prefix=t.shape[:-4], n_templ=n_templ, dtype=t.dtype
        )
        denominator = jnp.clip(jnp.sum(template_weights, axis=-1), min=1.0)
        denominator = denominator[..., None, None, None]
        t = template_pair_stack(
            t,
            params.template_pair_stack,
            mask=pair_mask[..., None, :, :],
            **settings,
        )
        weights = template_weights[..., :, None, None, None]
        t = jnp.sum(t * weights, axis=-4) / denominator
    return linear(jax.nn.relu(t), params.linear_t)
