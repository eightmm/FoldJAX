"""Explicit OpenFold3 ``state_dict`` to JAX parameter mapping.

Every mapper takes a flat ``{name: array}`` mapping and an optional prefix, so a
module can be mapped either standalone or from inside a full checkpoint. Missing
keys raise: a silently absent weight is the failure mode most likely to produce a
plausible-looking but wrong structure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np

from foldjax.models._stacking import prestack_layer_lists
from foldjax.models.openfold3.inference import InferenceParams
from foldjax.models.openfold3.models.atom_features import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderParams,
    AtomPairConditioningParams,
    NoisyPositionEmbedderParams,
    RefAtomFeatureEmbedderParams,
)
from foldjax.models.openfold3.models.attention import AttentionParams
from foldjax.models.openfold3.models.attention_pair_bias import (
    AdaAttentionPairBiasParams,
    AttentionPairBiasParams,
    CrossAttentionPairBiasParams,
)
from foldjax.models.openfold3.models.denoiser import DenoiserParams
from foldjax.models.openfold3.models.diffusion_conditioning import (
    DiffusionConditioningParams,
    FourierEmbeddingParams,
    SingleConditioningParams,
)
from foldjax.models.openfold3.models.diffusion_transformer import (
    AtomTransformerBlockParams,
    AtomTransformerParams,
    DiffusionTransformerBlockParams,
    DiffusionTransformerParams,
)
from foldjax.models.openfold3.models.heads import (
    AtomHeadParams,
    PairformerEmbeddingParams,
    PairHeadParams,
)
from foldjax.models.openfold3.models.input_embedders import (
    InputEmbedderParams,
    MSAEmbedderParams,
)
from foldjax.models.openfold3.models.msa import (
    MSAPairWeightedAveragingParams,
    OuterProductMeanParams,
)
from foldjax.models.openfold3.models.msa_module import (
    MSAModuleBlockParams,
    MSAModuleStackParams,
)
from foldjax.models.openfold3.models.pair_block import PairBlockParams
from foldjax.models.openfold3.models.pairformer import (
    PairformerBlockParams,
    PairformerStackParams,
)
from foldjax.models.openfold3.models.primitives import (
    AdaLNParams,
    ConditionedTransitionBlockParams,
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
)
from foldjax.models.openfold3.models.template_module import (
    TemplateEmbedderParams,
    TemplatePairEmbedderParams,
    TemplatePairStackParams,
)
from foldjax.models.openfold3.models.triangle import TriangleMultiplicationParams
from foldjax.models.openfold3.models.triangle_attention import TriangleAttentionParams
from foldjax.models.openfold3.models.trunk import TrunkParams


def to_array(value: Any) -> np.ndarray:
    """Convert one torch/numpy tensor to a host array without importing torch.

    Individual mappers deliberately stay on the host. ``map_inference_params``
    can then stack repeated layers before the parameter tree reaches a device,
    avoiding both an in-graph concatenate and a transient device-side copy of
    the unstacked checkpoint.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _require(state: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in state:
        raise KeyError(f"missing checkpoint key: {key}")
    return to_array(state[key])


def _optional(state: Mapping[str, Any], key: str) -> np.ndarray | None:
    return to_array(state[key]) if key in state else None


def _squeeze_trailing(array: Any) -> Any:
    """Drop trailing singleton axes from a 1-D-by-intent buffer."""
    result = np.asarray(array)
    while result.ndim > 1 and result.shape[-1] == 1:
        result = result[..., 0]
    return result


def map_linear(
    state: Mapping[str, Any], prefix: str = "", *, bias: bool | None = None
) -> LinearParams:
    """Map a ``Linear``. ``bias=None`` accepts whichever form is present."""
    weight = _require(state, _join(prefix, "weight"))
    key = _join(prefix, "bias")
    if bias is True:
        return LinearParams(weight=weight, bias=_require(state, key))
    if bias is False:
        if key in state:
            raise KeyError(f"unexpected bias for a bias-free linear: {key}")
        return LinearParams(weight=weight, bias=None)
    return LinearParams(weight=weight, bias=_optional(state, key))


def map_layer_norm(state: Mapping[str, Any], prefix: str = "") -> LayerNormParams:
    """Map a ``LayerNorm``, whose scale and offset are both optional upstream."""
    return LayerNormParams(
        weight=_optional(state, _join(prefix, "weight")),
        bias=_optional(state, _join(prefix, "bias")),
    )


def map_swiglu(state: Mapping[str, Any], prefix: str = "") -> SwiGLUParams:
    """Map a ``SwiGLU``; both projections are bias-free upstream."""
    return SwiGLUParams(
        linear_a=map_linear(state, _join(prefix, "linear_a"), bias=False),
        linear_b=map_linear(state, _join(prefix, "linear_b"), bias=False),
    )


def map_adaln(state: Mapping[str, Any], prefix: str = "") -> AdaLNParams:
    """Map an ``AdaLN``: ``linear_g`` has a bias, ``linear_s`` does not."""
    return AdaLNParams(
        layer_norm_a=map_layer_norm(state, _join(prefix, "layer_norm_a")),
        layer_norm_s=map_layer_norm(state, _join(prefix, "layer_norm_s")),
        linear_g=map_linear(state, _join(prefix, "linear_g"), bias=True),
        linear_s=map_linear(state, _join(prefix, "linear_s"), bias=False),
    )


def map_attention(
    state: Mapping[str, Any], prefix: str = "", *, q_bias: bool = False
) -> AttentionParams:
    """Map an ``Attention``.

    Projections are bias-free under ``mha_init``, but ``att_pair_bias_mha_init``
    gives ``linear_q`` a bias. That is the only difference between the two, so
    ``q_bias`` is explicit rather than inferred: silently accepting either form
    would hide a mismatched checkpoint.

    ``linear_g`` exists only for a gated module, so its absence is accepted and
    reported as ``None`` rather than raising.
    """
    gate_key = _join(prefix, "linear_g.weight")
    return AttentionParams(
        linear_q=map_linear(state, _join(prefix, "linear_q"), bias=q_bias),
        linear_k=map_linear(state, _join(prefix, "linear_k"), bias=False),
        linear_v=map_linear(state, _join(prefix, "linear_v"), bias=False),
        linear_o=map_linear(state, _join(prefix, "linear_o"), bias=False),
        linear_g=(
            map_linear(state, _join(prefix, "linear_g"), bias=False)
            if gate_key in state
            else None
        ),
    )


def map_triangle_multiplication(
    state: Mapping[str, Any], prefix: str = ""
) -> TriangleMultiplicationParams:
    """Map a ``TriangleMultiplicativeUpdate``.

    Outgoing and incoming share this layout; they differ only in the forward
    contraction, so ``outgoing`` is a call-site argument rather than a parameter.
    """
    return TriangleMultiplicationParams(
        layer_norm_in=map_layer_norm(state, _join(prefix, "layer_norm_in")),
        layer_norm_out=map_layer_norm(state, _join(prefix, "layer_norm_out")),
        linear_a_p=map_linear(state, _join(prefix, "linear_a_p"), bias=False),
        linear_a_g=map_linear(state, _join(prefix, "linear_a_g"), bias=False),
        linear_b_p=map_linear(state, _join(prefix, "linear_b_p"), bias=False),
        linear_b_g=map_linear(state, _join(prefix, "linear_b_g"), bias=False),
        linear_g=map_linear(state, _join(prefix, "linear_g"), bias=False),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
    )


def map_swiglu_transition(
    state: Mapping[str, Any], prefix: str = ""
) -> SwiGLUTransitionParams:
    """Map a ``SwiGLUTransition``."""
    return SwiGLUTransitionParams(
        layer_norm=map_layer_norm(state, _join(prefix, "layer_norm")),
        swiglu=map_swiglu(state, _join(prefix, "swiglu")),
        linear_out=map_linear(state, _join(prefix, "linear_out"), bias=False),
    )


def map_triangle_attention(
    state: Mapping[str, Any], prefix: str = ""
) -> TriangleAttentionParams:
    """Map a ``TriangleAttention``; the layout is identical for both directions."""
    return TriangleAttentionParams(
        layer_norm=map_layer_norm(state, _join(prefix, "layer_norm")),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
        mha=map_attention(state, _join(prefix, "mha")),
    )


def map_attention_pair_bias(
    state: Mapping[str, Any], prefix: str = ""
) -> AttentionPairBiasParams:
    """Map a trunk ``AttentionPairBias`` (no AdaLN conditioning).

    Its inner attention is the one place ``linear_q`` carries a bias.
    """
    return AttentionPairBiasParams(
        layer_norm_a=map_layer_norm(state, _join(prefix, "layer_norm_a")),
        layer_norm_z=map_layer_norm(state, _join(prefix, "layer_norm_z")),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
        mha=map_attention(state, _join(prefix, "mha"), q_bias=True),
    )


def map_pairformer_block(
    state: Mapping[str, Any], prefix: str = ""
) -> PairformerBlockParams:
    """Map one ``PairFormerBlock``."""
    return PairformerBlockParams(
        pair_stack=map_pair_block(state, _join(prefix, "pair_stack")),
        attn_pair_bias=map_attention_pair_bias(
            state, _join(prefix, "attn_pair_bias")
        ),
        single_transition=map_swiglu_transition(
            state, _join(prefix, "single_transition")
        ),
    )


def map_pairformer_stack(
    state: Mapping[str, Any], prefix: str = ""
) -> PairformerStackParams:
    """Map a ``PairFormerStack`` by discovering its ``blocks.N`` prefixes.

    The block count is read from the checkpoint rather than passed in, and the
    indices must be contiguous from zero: a gap would mean silently dropping a
    block from the stack.
    """
    root = _join(prefix, "blocks")
    indices = set()
    for key in state:
        if key.startswith(f"{root}."):
            index = key[len(root) + 1 :].split(".", 1)[0]
            if index.isdigit():
                indices.add(int(index))
    if not indices:
        raise KeyError(f"no pairformer blocks found under {root}")
    if sorted(indices) != list(range(len(indices))):
        raise KeyError(
            f"non-contiguous pairformer block indices under {root}: {sorted(indices)}"
        )
    return PairformerStackParams(
        blocks=tuple(
            map_pairformer_block(state, f"{root}.{index}")
            for index in range(len(indices))
        )
    )


def map_tri_mul(
    state: Mapping[str, Any], prefix: str = ""
) -> TriangleMultiplicationParams:
    """Map a triangular multiplicative update in whichever layout it is stored.

    Upstream picks the layout per stack via ``fuse_projection_weights``, so the
    decision belongs at each tri-mul prefix rather than at the checkpoint level:
    a model may in principle fuse one stack and not another. The fused variant is
    detected by its combined ``linear_ab_p`` projection and split by
    :func:`map_fused_triangle_multiplication`.
    """
    if _join(prefix, "linear_ab_p.weight") in state:
        return map_fused_triangle_multiplication(state, prefix)
    return map_triangle_multiplication(state, prefix)


def map_pair_block(state: Mapping[str, Any], prefix: str = "") -> PairBlockParams:
    """Map a ``PairBlock`` built with a SwiGLU transition.

    Either triangular-multiplication layout is accepted; see :func:`map_tri_mul`.
    """
    return PairBlockParams(
        tri_mul_out=map_tri_mul(state, _join(prefix, "tri_mul_out")),
        tri_mul_in=map_tri_mul(state, _join(prefix, "tri_mul_in")),
        tri_att_start=map_triangle_attention(state, _join(prefix, "tri_att_start")),
        tri_att_end=map_triangle_attention(state, _join(prefix, "tri_att_end")),
        pair_transition=map_swiglu_transition(
            state, _join(prefix, "pair_transition")
        ),
    )


def map_outer_product_mean(
    state: Mapping[str, Any], prefix: str = ""
) -> OuterProductMeanParams:
    """Map an ``OuterProductMean``; only ``linear_out`` carries a bias."""
    return OuterProductMeanParams(
        layer_norm=map_layer_norm(state, _join(prefix, "layer_norm")),
        linear_1=map_linear(state, _join(prefix, "linear_1"), bias=False),
        linear_2=map_linear(state, _join(prefix, "linear_2"), bias=False),
        linear_out=map_linear(state, _join(prefix, "linear_out"), bias=True),
    )


def map_msa_pair_weighted_averaging(
    state: Mapping[str, Any], prefix: str = ""
) -> MSAPairWeightedAveragingParams:
    """Map an ``MSAPairWeightedAveraging``; every projection is bias-free."""
    return MSAPairWeightedAveragingParams(
        layer_norm_m=map_layer_norm(state, _join(prefix, "layer_norm_m")),
        layer_norm_z=map_layer_norm(state, _join(prefix, "layer_norm_z")),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
        linear_v=map_linear(state, _join(prefix, "linear_v"), bias=False),
        linear_g=map_linear(state, _join(prefix, "linear_g"), bias=False),
        linear_o=map_linear(state, _join(prefix, "linear_o"), bias=False),
    )


def _block_indices(state: Mapping[str, Any], root: str) -> range:
    """Return the contiguous ``root.N`` block indices present in ``state``."""
    indices = set()
    for key in state:
        if key.startswith(f"{root}."):
            index = key[len(root) + 1 :].split(".", 1)[0]
            if index.isdigit():
                indices.add(int(index))
    if not indices:
        raise KeyError(f"no blocks found under {root}")
    if sorted(indices) != list(range(len(indices))):
        raise KeyError(f"non-contiguous block indices under {root}: {sorted(indices)}")
    return range(len(indices))


def map_msa_module_block(
    state: Mapping[str, Any], prefix: str = ""
) -> MSAModuleBlockParams:
    """Map one ``MSAModuleBlock``.

    A block with ``skip_msa_update`` has no ``msa_att_row``/``msa_transition``
    entries. Their absence is recognized, but a half-present pair is an error:
    that would mean the checkpoint disagrees with the block configuration.
    """
    has_att = any(key.startswith(_join(prefix, "msa_att_row") + ".") for key in state)
    has_trans = any(
        key.startswith(_join(prefix, "msa_transition") + ".") for key in state
    )
    if has_att != has_trans:
        raise KeyError(
            f"{prefix}: msa_att_row and msa_transition must both be present or "
            f"both absent (att={has_att}, transition={has_trans})"
        )
    return MSAModuleBlockParams(
        outer_product_mean=map_outer_product_mean(
            state, _join(prefix, "outer_product_mean")
        ),
        pair_stack=map_pair_block(state, _join(prefix, "pair_stack")),
        msa_att_row=(
            map_msa_pair_weighted_averaging(state, _join(prefix, "msa_att_row"))
            if has_att
            else None
        ),
        msa_transition=(
            map_swiglu_transition(state, _join(prefix, "msa_transition"))
            if has_trans
            else None
        ),
    )


def map_msa_module_stack(
    state: Mapping[str, Any], prefix: str = ""
) -> MSAModuleStackParams:
    """Map an ``MSAModuleStack`` by discovering its ``blocks.N`` prefixes."""
    root = _join(prefix, "blocks")
    return MSAModuleStackParams(
        blocks=tuple(
            map_msa_module_block(state, f"{root}.{index}")
            for index in _block_indices(state, root)
        )
    )


def map_template_pair_stack(
    state: Mapping[str, Any], prefix: str = ""
) -> TemplatePairStackParams:
    """Map a ``TemplatePairStack``.

    Its blocks share ``PairBlock``'s layout, so the pair-block mapper is reused;
    ``tri_mul_first`` is a forward-pass option, not a stored parameter.
    """
    root = _join(prefix, "blocks")
    return TemplatePairStackParams(
        blocks=tuple(
            map_pair_block(state, f"{root}.{index}")
            for index in _block_indices(state, root)
        ),
        layer_norm=map_layer_norm(state, _join(prefix, "layer_norm")),
    )


def map_ref_atom_feature_embedder(
    state: Mapping[str, Any], prefix: str = ""
) -> RefAtomFeatureEmbedderParams:
    """Map a ``RefAtomFeatureEmbedder``; every projection is bias-free."""
    names = (
        "linear_ref_pos",
        "linear_ref_charge",
        "linear_ref_mask",
        "linear_ref_element",
        "linear_ref_atom_chars",
        "linear_ref_offset",
        "linear_inv_sq_dists",
        "linear_valid_mask",
    )
    return RefAtomFeatureEmbedderParams(
        **{
            name: map_linear(state, _join(prefix, name), bias=False)
            for name in names
        }
    )


def map_noisy_position_embedder(
    state: Mapping[str, Any], prefix: str = ""
) -> NoisyPositionEmbedderParams:
    """Map a ``NoisyPositionEmbedder``; both layer norms are scale-only."""
    return NoisyPositionEmbedderParams(
        layer_norm_s=map_layer_norm(state, _join(prefix, "layer_norm_s")),
        linear_s=map_linear(state, _join(prefix, "linear_s"), bias=False),
        layer_norm_z=map_layer_norm(state, _join(prefix, "layer_norm_z")),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
        linear_r=map_linear(state, _join(prefix, "linear_r"), bias=False),
    )


def map_conditioned_transition_block(
    state: Mapping[str, Any], prefix: str = ""
) -> ConditionedTransitionBlockParams:
    """Map a ``ConditionedTransitionBlock``; ``linear_g`` has a bias, out does not."""
    return ConditionedTransitionBlockParams(
        layer_norm=map_adaln(state, _join(prefix, "layer_norm")),
        swiglu=map_swiglu(state, _join(prefix, "swiglu")),
        linear_g=map_linear(state, _join(prefix, "linear_g"), bias=True),
        linear_out=map_linear(state, _join(prefix, "linear_out"), bias=False),
    )


def map_ada_attention_pair_bias(
    state: Mapping[str, Any], prefix: str = ""
) -> AdaAttentionPairBiasParams:
    """Map the AdaLN-conditioned ``AttentionPairBias`` used inside diffusion."""
    return AdaAttentionPairBiasParams(
        layer_norm_a=map_adaln(state, _join(prefix, "layer_norm_a")),
        linear_ada_out=map_linear(
            state, _join(prefix, "linear_ada_out"), bias=True
        ),
        layer_norm_z=map_layer_norm(state, _join(prefix, "layer_norm_z")),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
        mha=map_attention(state, _join(prefix, "mha"), q_bias=True),
    )


def map_diffusion_transformer_block(
    state: Mapping[str, Any], prefix: str = ""
) -> DiffusionTransformerBlockParams:
    """Map one ``DiffusionTransformerBlock``."""
    return DiffusionTransformerBlockParams(
        attention_pair_bias=map_ada_attention_pair_bias(
            state, _join(prefix, "attention_pair_bias")
        ),
        conditioned_transition=map_conditioned_transition_block(
            state, _join(prefix, "conditioned_transition")
        ),
    )


def map_diffusion_transformer(
    state: Mapping[str, Any], prefix: str = ""
) -> DiffusionTransformerParams:
    """Map a ``DiffusionTransformer`` by discovering its ``blocks.N`` prefixes."""
    root = _join(prefix, "blocks")
    return DiffusionTransformerParams(
        blocks=tuple(
            map_diffusion_transformer_block(state, f"{root}.{index}")
            for index in _block_indices(state, root)
        )
    )


def map_pair_head(
    state: Mapping[str, Any], prefix: str = "", *, layer_norm: bool = True
) -> PairHeadParams:
    """Map a pair-level logit head.

    ``layer_norm=False`` is the distogram head, which has none. Every one of
    these heads uses a bias-free linear.
    """
    return PairHeadParams(
        linear=map_linear(state, _join(prefix, "linear"), bias=False),
        layer_norm=(
            map_layer_norm(state, _join(prefix, "layer_norm")) if layer_norm else None
        ),
    )


def map_atom_logit_head(
    state: Mapping[str, Any], prefix: str = ""
) -> AtomHeadParams:
    """Map a per-atom logit head (pLDDT or experimentally resolved)."""
    return AtomHeadParams(
        layer_norm=map_layer_norm(state, _join(prefix, "layer_norm")),
        linear=map_linear(state, _join(prefix, "linear"), bias=False),
    )


def map_cross_attention_pair_bias(
    state: Mapping[str, Any], prefix: str = ""
) -> CrossAttentionPairBiasParams:
    """Map a ``CrossAttentionPairBias``.

    The AdaLN form is detected from the checkpoint: it carries a
    ``linear_ada_out`` and its norms have their own sub-projections.
    """
    ada_key = _join(prefix, "linear_ada_out.weight")
    conditioned = ada_key in state
    norm = map_adaln if conditioned else map_layer_norm
    return CrossAttentionPairBiasParams(
        layer_norm_a_q=norm(state, _join(prefix, "layer_norm_a_q")),
        layer_norm_a_k=norm(state, _join(prefix, "layer_norm_a_k")),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
        mha=map_attention(state, _join(prefix, "mha"), q_bias=True),
        linear_ada_out=(
            map_linear(state, _join(prefix, "linear_ada_out"), bias=True)
            if conditioned
            else None
        ),
    )


def map_atom_pair_conditioning(
    state: Mapping[str, Any], prefix: str = ""
) -> AtomPairConditioningParams:
    """Map the atom pair conditioning stage.

    Upstream stores the pair MLP as an ``nn.Sequential`` whose ReLUs occupy
    indices 0, 2 and 4, so the three Linears land at ``pair_mlp.1``, ``.3`` and
    ``.5`` rather than ``.0``, ``.1``, ``.2``.
    """
    return AtomPairConditioningParams(
        linear_l=map_linear(state, _join(prefix, "linear_l"), bias=False),
        linear_m=map_linear(state, _join(prefix, "linear_m"), bias=False),
        pair_mlp_1=map_linear(state, _join(prefix, "pair_mlp.1"), bias=False),
        pair_mlp_2=map_linear(state, _join(prefix, "pair_mlp.3"), bias=False),
        pair_mlp_3=map_linear(state, _join(prefix, "pair_mlp.5"), bias=False),
    )


def map_atom_transformer(
    state: Mapping[str, Any], prefix: str = ""
) -> AtomTransformerParams:
    """Map the cross-attention atom transformer.

    Its ``layer_norm_z`` sits at the stack level, not inside each block.
    """
    root = _join(prefix, "blocks")
    return AtomTransformerParams(
        blocks=tuple(
            AtomTransformerBlockParams(
                attention_pair_bias=map_cross_attention_pair_bias(
                    state, f"{root}.{index}.attention_pair_bias"
                ),
                conditioned_transition=map_conditioned_transition_block(
                    state, f"{root}.{index}.conditioned_transition"
                ),
            )
            for index in _block_indices(state, root)
        ),
        layer_norm_z=map_layer_norm(state, _join(prefix, "layer_norm_z")),
    )


def map_atom_attention_encoder(
    state: Mapping[str, Any], prefix: str = ""
) -> AtomAttentionEncoderParams:
    """Map an ``AtomAttentionEncoder``.

    The noisy-position path is detected from the checkpoint: it exists only when
    the encoder was built with ``add_noisy_pos=True``. ``linear_q`` is a
    ``Sequential(Linear, ReLU)``, so its weight lives at ``linear_q.0``.
    """
    noisy_prefix = _join(prefix, "noisy_position_embedder")
    has_noisy = any(key.startswith(noisy_prefix + ".") for key in state)
    return AtomAttentionEncoderParams(
        ref_atom_feature_embedder=map_ref_atom_feature_embedder(
            state, _join(prefix, "ref_atom_feature_embedder")
        ),
        pair_conditioning=map_atom_pair_conditioning(state, prefix),
        atom_transformer=map_atom_transformer(
            state, _join(prefix, "atom_transformer")
        ),
        linear_q=map_linear(state, _join(prefix, "linear_q.0"), bias=False),
        noisy_position_embedder=(
            map_noisy_position_embedder(state, noisy_prefix) if has_noisy else None
        ),
    )


def map_atom_attention_decoder(
    state: Mapping[str, Any], prefix: str = ""
) -> AtomAttentionDecoderParams:
    """Map an ``AtomAttentionDecoder``; its layer norm is scale-only."""
    return AtomAttentionDecoderParams(
        linear_q_in=map_linear(state, _join(prefix, "linear_q_in"), bias=False),
        atom_transformer=map_atom_transformer(
            state, _join(prefix, "atom_transformer")
        ),
        layer_norm=map_layer_norm(state, _join(prefix, "layer_norm")),
        linear_q_out=map_linear(state, _join(prefix, "linear_q_out"), bias=False),
    )


def map_denoiser(state: Mapping[str, Any], prefix: str = "") -> DenoiserParams:
    """Map the diffusion denoiser body."""
    return DenoiserParams(
        atom_attn_enc=map_atom_attention_encoder(
            state, _join(prefix, "atom_attn_enc")
        ),
        layer_norm_s=map_layer_norm(state, _join(prefix, "layer_norm_s")),
        linear_s=map_linear(state, _join(prefix, "linear_s"), bias=False),
        diffusion_transformer=map_diffusion_transformer(
            state, _join(prefix, "diffusion_transformer")
        ),
        layer_norm_a=map_layer_norm(state, _join(prefix, "layer_norm_a")),
        atom_attn_dec=map_atom_attention_decoder(
            state, _join(prefix, "atom_attn_dec")
        ),
    )


def map_diffusion_conditioning(
    state: Mapping[str, Any], prefix: str = ""
) -> SingleConditioningParams:
    """Map a ``DiffusionConditioning``: both the single and the pair path.

    ``fourier_emb.w``/``.b`` are buffers sampled from a seeded torch generator, so
    they must come from the checkpoint; JAX cannot reproduce that stream.
    ``transition_s`` is an ``nn.ModuleList``, discovered by index.
    """
    root = _join(prefix, "transition_s")
    has_transitions = any(key.startswith(root + ".") for key in state)
    z_root = _join(prefix, "transition_z")
    has_pair = _join(prefix, "linear_z.weight") in state
    return DiffusionConditioningParams(
        layer_norm_s=map_layer_norm(state, _join(prefix, "layer_norm_s")),
        linear_s=map_linear(state, _join(prefix, "linear_s"), bias=False),
        fourier_emb=FourierEmbeddingParams(
            # Upstream declares these buffers as ``torch.empty(c)``, but the released
            # checkpoint stores ``w`` as ``(c, 1)``. They are a vector of frequencies
            # and a vector of phases, applied to an ``[..., 1]`` input, so a trailing
            # singleton is a storage artifact -- left in place it broadcasts to
            # ``[..., c, 1]`` and the conditioning fails to add to the single
            # representation. Found only when real weights were loaded.
            w=_squeeze_trailing(_require(state, _join(prefix, "fourier_emb.w"))),
            b=_squeeze_trailing(_require(state, _join(prefix, "fourier_emb.b"))),
        ),
        layer_norm_n=map_layer_norm(state, _join(prefix, "layer_norm_n")),
        linear_n=map_linear(state, _join(prefix, "linear_n"), bias=False),
        transition_s=tuple(
            map_swiglu_transition(state, f"{root}.{index}")
            for index in (_block_indices(state, root) if has_transitions else ())
        ),
        layer_norm_z=(
            map_layer_norm(state, _join(prefix, "layer_norm_z")) if has_pair else None
        ),
        linear_z=(
            map_linear(state, _join(prefix, "linear_z"), bias=False)
            if has_pair
            else None
        ),
        transition_z=tuple(
            map_swiglu_transition(state, f"{z_root}.{index}")
            for index in (_block_indices(state, z_root) if has_pair else ())
        ),
    )


def map_msa_embedder(
    state: Mapping[str, Any], prefix: str = ""
) -> MSAEmbedderParams:
    """Map an ``MSAModuleEmbedder``'s projections."""
    return MSAEmbedderParams(
        linear_m=map_linear(state, _join(prefix, "linear_m"), bias=False),
        linear_s_input=map_linear(state, _join(prefix, "linear_s_input"), bias=False),
    )


def map_input_embedder(
    state: Mapping[str, Any], prefix: str = ""
) -> InputEmbedderParams:
    """Map an ``InputEmbedderAllAtom``."""
    return InputEmbedderParams(
        atom_attn_enc=map_atom_attention_encoder(
            state, _join(prefix, "atom_attn_enc")
        ),
        linear_s=map_linear(state, _join(prefix, "linear_s"), bias=False),
        linear_z_i=map_linear(state, _join(prefix, "linear_z_i"), bias=False),
        linear_z_j=map_linear(state, _join(prefix, "linear_z_j"), bias=False),
        linear_relpos=map_linear(state, _join(prefix, "linear_relpos"), bias=False),
        linear_token_bonds=map_linear(
            state, _join(prefix, "linear_token_bonds"), bias=False
        ),
    )


def map_trunk(state: Mapping[str, Any], prefix: str = "") -> TrunkParams:
    """Map the trunk, including its two recycling projections."""
    return TrunkParams(
        input_embedder=map_input_embedder(state, _join(prefix, "input_embedder")),
        msa_module_embedder=map_msa_embedder(
            state, _join(prefix, "msa_module_embedder")
        ),
        msa_module=map_msa_module_stack(state, _join(prefix, "msa_module")),
        pairformer_stack=map_pairformer_stack(
            state, _join(prefix, "pairformer_stack")
        ),
        layer_norm_z=map_layer_norm(state, _join(prefix, "layer_norm_z")),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
        layer_norm_s=map_layer_norm(state, _join(prefix, "layer_norm_s")),
        linear_s=map_linear(state, _join(prefix, "linear_s"), bias=False),
        # A checkpoint trained without templates has no tower at all, so this is
        # recognized by absence rather than required.
        template_embedder=(
            map_template_embedder(state, _join(prefix, "template_embedder"))
            if any(
                key.startswith(_join(prefix, "template_embedder") + ".")
                for key in state
            )
            else None
        ),
    )


def map_fused_triangle_multiplication(
    state: Mapping[str, Any], prefix: str = ""
) -> TriangleMultiplicationParams:
    """Map a *fused* ``FusedTriangleMultiplicativeUpdate`` into unfused params.

    The fused variant computes one ``linear_ab_p``/``linear_ab_g`` projection of
    width ``2 * c_hidden`` and splits the result in half, where the unfused one
    runs two projections of width ``c_hidden``. Those are the same function, so
    splitting the stored weights row-wise converts a fused checkpoint into the
    unfused parameters this port's forward pass already implements — no second
    forward path needed.

    Row-major split order matters: the first half is ``a``, the second is ``b``,
    matching upstream's ``ab[..., :c_hidden]`` / ``ab[..., c_hidden:]``.
    """
    ab_p = _require(state, _join(prefix, "linear_ab_p.weight"))
    ab_g = _require(state, _join(prefix, "linear_ab_g.weight"))
    if ab_p.shape[0] % 2 or ab_g.shape[0] % 2:
        raise ValueError(
            f"{prefix}: fused projection output width must be even, got "
            f"{ab_p.shape[0]} and {ab_g.shape[0]}"
        )
    half_p = ab_p.shape[0] // 2
    half_g = ab_g.shape[0] // 2
    return TriangleMultiplicationParams(
        layer_norm_in=map_layer_norm(state, _join(prefix, "layer_norm_in")),
        layer_norm_out=map_layer_norm(state, _join(prefix, "layer_norm_out")),
        linear_a_p=LinearParams(weight=ab_p[:half_p]),
        linear_a_g=LinearParams(weight=ab_g[:half_g]),
        linear_b_p=LinearParams(weight=ab_p[half_p:]),
        linear_b_g=LinearParams(weight=ab_g[half_g:]),
        linear_g=map_linear(state, _join(prefix, "linear_g"), bias=False),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
    )


# Where each parameter group sits relative to the model root, read off
# `projects/of3_all_atom/model.py` and `core/model/structure/diffusion_module.py`:
# the trunk modules are attributes of the model itself, the denoiser body is
# `diffusion_module`, its conditioning is one level further in, and the four
# logit heads hang off `aux_heads`.
INFERENCE_PREFIXES = {
    "trunk": "",
    "diffusion_conditioning": "diffusion_module.diffusion_conditioning",
    "denoiser": "diffusion_module",
    "pairformer_embedding": "aux_heads.pairformer_embedding",
    "plddt_head": "aux_heads.plddt",
    "pae_head": "aux_heads.pae",
    "pde_head": "aux_heads.pde",
    "distogram_head": "aux_heads.distogram",
    "experimentally_resolved_head": "aux_heads.experimentally_resolved",
}


def find_model_prefix(state: Mapping[str, Any]) -> str | None:
    """Locate the model root by the one stack that is certain to be under it.

    Checkpoints wrap the model differently (a bare state dict, a Lightning
    ``state_dict``, a training wrapper that nests it under ``model``), and
    ``load_checkpoint`` only strips the wrappers it knows. Rather than guess from
    a fixed list, find ``pairformer_stack.blocks.`` and take what precedes it.

    Returns ``None`` if no Pairformer stack is present, which means this is not
    an OpenFold3 model checkpoint.
    """
    marker = "pairformer_stack.blocks."
    for key in state:
        index = key.find(marker)
        if index == -1:
            continue
        # `aux_heads.pairformer_embedding.pairformer_stack.blocks.` also matches;
        # the trunk stack is the shallowest occurrence, so prefer it.
        prefix = key[:index].rstrip(".")
        if "aux_heads" not in prefix:
            return prefix
    return None


def resolve_model_prefix(
    state: Mapping[str, Any], prefix: str | None = None
) -> str:
    """Return an explicit model root or detect it with the mapper's error."""
    if prefix is not None:
        return prefix
    detected = find_model_prefix(state)
    if detected is None:
        raise KeyError(
            "no pairformer_stack.blocks.* keys, so this is not an OpenFold3 "
            "model checkpoint; inspect it with openfold3-jax-inspect-checkpoint"
        )
    return detected


def _bitwise_identical_checkpoint_arrays(left: Any, right: Any) -> bool:
    """Whether two loaded tensors have exactly the same contiguous bytes."""
    left_array = to_array(left)
    right_array = to_array(right)
    if (
        left_array.dtype != right_array.dtype
        or left_array.shape != right_array.shape
        or not left_array.flags.c_contiguous
        or not right_array.flags.c_contiguous
    ):
        return False
    if left_array.dtype.hasobject:
        return False
    try:
        return np.array_equal(
            left_array.reshape(-1).view(np.uint8),
            right_array.reshape(-1).view(np.uint8),
        )
    except (TypeError, ValueError):
        return False


def prune_sample_diffusion_aliases(
    state: dict[str, Any], *, prefix: str
) -> int:
    """Discard one complete, byte-identical sampler alias group in place.

    Upstream registers the denoiser both as ``diffusion_module`` and below
    ``sample_diffusion``. Torch preserves that shared storage, but FoldJAX's
    restricted archive reader returns owning NumPy arrays and therefore holds
    the checkpoint payload twice while inference parameters are prestacked and
    transferred. The inference mapper reads only the canonical module.

    Be conservative for other checkpoint layouts: an absent, partial,
    non-contiguous, differently typed, or byte-different alias group is left
    untouched. ``prefix`` must already be resolved so similarly named groups
    elsewhere in the checkpoint cannot be removed.
    """
    root = f"{prefix}." if prefix else ""
    alias_root = f"{root}sample_diffusion."
    canonical_root = f"{root}diffusion_module."
    aliases = {
        key[len(alias_root) :]: key
        for key in state
        if key.startswith(alias_root)
    }
    canonical = {
        key[len(root) :]: key
        for key in state
        if key.startswith(canonical_root)
    }
    if not aliases or aliases.keys() != canonical.keys():
        return 0
    if any(
        not _bitwise_identical_checkpoint_arrays(
            state[alias_key], state[canonical[suffix]]
        )
        for suffix, alias_key in aliases.items()
    ):
        return 0
    for key in aliases.values():
        del state[key]
    return len(aliases)


def map_pairformer_embedding(
    state: Mapping[str, Any], prefix: str = ""
) -> PairformerEmbeddingParams:
    """Map the confidence heads' ``PairformerEmbedding``."""
    return PairformerEmbeddingParams(
        linear_i=map_linear(state, _join(prefix, "linear_i"), bias=False),
        linear_j=map_linear(state, _join(prefix, "linear_j"), bias=False),
        linear_distance=map_linear(
            state, _join(prefix, "linear_distance"), bias=False
        ),
        pairformer_stack=map_pairformer_stack(
            state, _join(prefix, "pairformer_stack")
        ),
    )


def _prestack_scanned_blocks(params: Any) -> Any:
    """Pre-stack only fields that OpenFold3 passes to ``scan_stack``.

    The shared recursive helper intentionally recognises any homogeneous layer
    sequence. OpenFold3 also has two-element ``transition_s``/``transition_z``
    sequences that run through Python loops; stacking those replaces separate
    parameters with slices of one large buffer and increases XLA temporary
    memory. Every scanned model stack is stored in a field named ``blocks``, so
    keep this bridge-specific selection narrow and leave all other sequences in
    their original layout.
    """
    if isinstance(params, dict):
        return {
            name: (
                prestack_layer_lists(value)
                if name == "blocks"
                else _prestack_scanned_blocks(value)
            )
            for name, value in params.items()
        }
    if hasattr(params, "_fields") and isinstance(params, tuple):
        return type(params)(
            *(
                prestack_layer_lists(value)
                if name == "blocks"
                else _prestack_scanned_blocks(value)
                for name, value in zip(params._fields, params, strict=True)
            )
        )
    if isinstance(params, list):
        return [_prestack_scanned_blocks(value) for value in params]
    if isinstance(params, tuple):
        return tuple(_prestack_scanned_blocks(value) for value in params)
    return params


def map_inference_params(
    state: Mapping[str, Any], prefix: str | None = None, *, prestack: bool = True
) -> InferenceParams:
    """Assemble every parameter group :func:`predict` needs from a checkpoint.

    Args:
        state: a loaded checkpoint, already stripped of wrapper prefixes.
        prefix: the model root. Detected via :func:`find_model_prefix` when
            omitted; pass ``""`` to force the top level.
        prestack: stack homogeneous scanned block fields on the host before
            moving the complete parameter tree to JAX. Disable only for
            diagnostics that need the historical per-layer parameter layout.

    Raises:
        KeyError: a group is missing. The PAE head is the one upstream omits
            conditionally -- it is only constructed when ``config.pae.enabled``
            -- so its absence is reported as a checkpoint property rather than a
            mapping bug.
    """
    prefix = resolve_model_prefix(state, prefix)

    pae_root = _join(prefix, INFERENCE_PREFIXES["pae_head"])
    if not any(key.startswith(pae_root + ".") for key in state):
        raise KeyError(
            f"{pae_root}.* is absent. Upstream only builds the PAE head when "
            "config.pae.enabled is set, so this checkpoint cannot produce PAE or "
            "ipTM. Map the other groups individually if that is acceptable."
        )

    params = InferenceParams(
        trunk=map_trunk(state, _join(prefix, INFERENCE_PREFIXES["trunk"])),
        diffusion_conditioning=map_diffusion_conditioning(
            state, _join(prefix, INFERENCE_PREFIXES["diffusion_conditioning"])
        ),
        denoiser=map_denoiser(state, _join(prefix, INFERENCE_PREFIXES["denoiser"])),
        # pLDDT is per-atom; the other three are pair heads, and the distogram
        # alone has no layer norm.
        plddt_head=map_atom_logit_head(
            state, _join(prefix, INFERENCE_PREFIXES["plddt_head"])
        ),
        pairformer_embedding=map_pairformer_embedding(
            state, _join(prefix, INFERENCE_PREFIXES["pairformer_embedding"])
        ),
        pae_head=map_pair_head(state, pae_root),
        pde_head=map_pair_head(state, _join(prefix, INFERENCE_PREFIXES["pde_head"])),
        distogram_head=map_pair_head(
            state,
            _join(prefix, INFERENCE_PREFIXES["distogram_head"]),
            layer_norm=False,
        ),
        experimentally_resolved_head=map_atom_logit_head(
            state, _join(prefix, INFERENCE_PREFIXES["experimentally_resolved_head"])
        ),
    )
    if prestack:
        params = _prestack_scanned_blocks(params)
    # `device_put`, not `jnp.asarray`: the latter is a traced operation, so it
    # compiles one `jit(stage)` program per distinct shape and dtype in the
    # tree -- 86 of them once the blocks are stacked. None is large enough to
    # clear `enable_compilation_cache`'s one-second floor, so every process
    # that loads weights compiles the whole set again before it can predict.
    # `device_put` is a transfer, canonicalises dtypes identically, and moves
    # the tree in one call.
    return jax.tree.map(
        lambda leaf: jax.device_put(leaf) if isinstance(leaf, np.ndarray) else leaf,
        params,
        is_leaf=lambda leaf: leaf is None,
    )


def map_template_pair_embedder(
    state: Mapping[str, Any], prefix: str = ""
) -> TemplatePairEmbedderParams:
    """Map a ``TemplatePairEmbedderAllAtom``; every projection is bias-free."""
    return TemplatePairEmbedderParams(
        dgram_linear=map_linear(state, _join(prefix, "dgram_linear"), bias=False),
        aatype_linear_1=map_linear(
            state, _join(prefix, "aatype_linear_1"), bias=False
        ),
        aatype_linear_2=map_linear(
            state, _join(prefix, "aatype_linear_2"), bias=False
        ),
        pseudo_beta_mask_linear=map_linear(
            state, _join(prefix, "pseudo_beta_mask_linear"), bias=False
        ),
        x_linear=map_linear(state, _join(prefix, "x_linear"), bias=False),
        y_linear=map_linear(state, _join(prefix, "y_linear"), bias=False),
        z_linear=map_linear(state, _join(prefix, "z_linear"), bias=False),
        backbone_mask_linear=map_linear(
            state, _join(prefix, "backbone_mask_linear"), bias=False
        ),
        layer_norm_z=map_layer_norm(state, _join(prefix, "layer_norm_z")),
        linear_z=map_linear(state, _join(prefix, "linear_z"), bias=False),
    )


def map_template_embedder(
    state: Mapping[str, Any], prefix: str = ""
) -> TemplateEmbedderParams:
    """Map a ``TemplateEmbedderAllAtom``."""
    return TemplateEmbedderParams(
        template_pair_embedder=map_template_pair_embedder(
            state, _join(prefix, "template_pair_embedder")
        ),
        template_pair_stack=map_template_pair_stack(
            state, _join(prefix, "template_pair_stack")
        ),
        linear_t=map_linear(state, _join(prefix, "linear_t"), bias=False),
    )


# The old name, kept so existing callers do not break; it described only the
# single path, which the type outgrew when the pair path was added.
map_single_conditioning = map_diffusion_conditioning
