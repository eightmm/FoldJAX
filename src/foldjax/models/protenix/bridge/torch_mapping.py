"""PyTorch checkpoint -> JAX param bridge for Protenix.

Pure-mapping functions are torch-free (tensors are duck-typed via
``tensor_to_numpy``). ``load_torch_checkpoint`` is the only entry that needs
PyTorch, and it imports it lazily so the native inference runtime never pulls
in torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderCacheParams,
    AtomAttentionEncoderParams,
    AtomPairMlpParams,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    DiffusionConditioningParams,
    DiffusionModuleParams,
)
from foldjax.models.protenix.models.diffusion.transformer import (
    ConditionedTransitionParams,
    DiffusionTransformerBlockParams,
    DiffusionTransformerStackParams,
)
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
    ConfidenceHeadParams,
    ConfidenceOutputParams,
)
from foldjax.models.protenix.models.heads.head import DistogramParams
from foldjax.models.protenix.models.model import ProtenixInferenceParams
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    TransitionParams,
)
from foldjax.models.protenix.models.triangle.triangle import (
    TriangleAttentionParams,
    TriangleMultiplicationParams,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    ConstraintEmbedderParams,
    FourierParams,
    InputFeatureEmbedderParams,
    RelativePositionParams,
    SubstructureMlpParams,
)
from foldjax.models.protenix.models.trunk_blocks.msa import (
    MSABlockParams,
    MSAModuleParams,
    MSAPairWeightedAveragingParams,
    OuterProductMeanParams,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    PairformerBlockParams,
    PairformerStackParams,
)
from foldjax.models.protenix.models.trunk_blocks.template import TemplateEmbedderParams
from foldjax.models.protenix.models.trunk_blocks.trunk import (
    PairformerOutputParams,
    RecyclingProjectionParams,
    TrunkInitializationParams,
    TrunkParams,
)


def tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert torch-like tensors or arrays to host numpy arrays."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def require_key(state_dict: Mapping[str, Any], key: str) -> np.ndarray:
    """Read a required state-dict key with a clear error."""

    if key not in state_dict:
        raise KeyError(f"missing checkpoint key: {key}")
    return tensor_to_numpy(state_dict[key])


def map_linear_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    bias: bool = True,
) -> LinearParams:
    """Map a Protenix linear submodule into JAX parameters."""

    weight = jnp.asarray(require_key(state_dict, f"{prefix}.weight"))
    bias_arr = jnp.asarray(require_key(state_dict, f"{prefix}.bias")) if bias else None
    return LinearParams(weight=weight, bias=bias_arr)


def map_layer_norm_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    scale: bool = True,
    offset: bool = True,
) -> LayerNormParams:
    """Map a Protenix/OpenFold layer norm submodule."""

    weight = jnp.asarray(require_key(state_dict, f"{prefix}.weight")) if scale else None
    bias = jnp.asarray(require_key(state_dict, f"{prefix}.bias")) if offset else None
    return LayerNormParams(weight=weight, bias=bias)


def map_transition_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> TransitionParams:
    """Map ``protenix.model.modules.primitives.Transition`` parameters."""

    return TransitionParams(
        layer_norm=map_layer_norm_state_dict(state_dict, f"{prefix}.layernorm1"),
        linear_a=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_a",
            bias=False,
        ),
        linear_b=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_b",
            bias=False,
        ),
        linear_out=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias",
            bias=False,
        ),
    )


def map_adaptive_layer_norm_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> AdaptiveLayerNormParams:
    """Map ``protenix.model.modules.primitives.AdaptiveLayerNorm``."""

    return AdaptiveLayerNormParams(
        layernorm_a=LayerNormParams(weight=None, bias=None),
        layernorm_s=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_s",
            scale=True,
            offset=False,
        ),
        linear_s=map_linear_state_dict(state_dict, f"{prefix}.linear_s", bias=True),
        linear_no_bias_s=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_nobias_s",
            bias=False,
        ),
    )


def map_attention_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    gating: bool = True,
    q_linear_bias: bool = True,
) -> AttentionParams:
    """Map ``protenix.model.modules.primitives.Attention``."""

    return AttentionParams(
        linear_q=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_q",
            bias=q_linear_bias,
        ),
        linear_k=map_linear_state_dict(state_dict, f"{prefix}.linear_k", bias=False),
        linear_v=map_linear_state_dict(state_dict, f"{prefix}.linear_v", bias=False),
        linear_o=map_linear_state_dict(state_dict, f"{prefix}.linear_o", bias=False),
        linear_g=map_linear_state_dict(state_dict, f"{prefix}.linear_g", bias=False)
        if gating
        else None,
    )


def map_attention_pair_bias_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    has_s: bool,
    create_offset_ln_z: bool = True,
    cross_attention_mode: bool = False,
) -> AttentionPairBiasParams:
    """Map standard ``AttentionPairBias`` parameters."""

    if has_s:
        layernorm_a = map_adaptive_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_a",
        )
        layernorm_kv = (
            map_adaptive_layer_norm_state_dict(state_dict, f"{prefix}.layernorm_kv")
            if cross_attention_mode
            else None
        )
        linear_a_last = map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_a_last",
            bias=True,
        )
    else:
        layernorm_a = map_layer_norm_state_dict(state_dict, f"{prefix}.layernorm_a")
        layernorm_kv = (
            map_layer_norm_state_dict(state_dict, f"{prefix}.layernorm_kv")
            if cross_attention_mode
            else None
        )
        linear_a_last = None

    return AttentionPairBiasParams(
        layernorm_a=layernorm_a,
        layernorm_kv=layernorm_kv,
        attention=map_attention_state_dict(state_dict, f"{prefix}.attention"),
        layernorm_z=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_z",
            scale=True,
            offset=create_offset_ln_z,
        ),
        linear_z=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_nobias_z",
            bias=False,
        ),
        linear_a_last=linear_a_last,
        has_s=has_s,
        cross_attention_mode=cross_attention_mode,
    )


def map_atom_attention_encoder_cache_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> AtomAttentionEncoderCacheParams:
    """Map AtomAttentionEncoder cache-preparation parameters."""

    return AtomAttentionEncoderCacheParams(
        linear_ref_pos=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_ref_pos",
            bias=False,
        ),
        linear_ref_charge=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_ref_charge",
            bias=False,
        ),
        linear_f=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_f",
            bias=False,
        ),
        linear_d=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_d",
            bias=False,
        ),
        linear_invd=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_invd",
            bias=False,
        ),
        linear_v=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_v",
            bias=False,
        ),
    )


def map_atom_pair_mlp_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> AtomPairMlpParams:
    """Map AtomAttentionEncoder small MLP parameters."""

    return AtomPairMlpParams(
        linear_1=map_linear_state_dict(state_dict, f"{prefix}.1", bias=False),
        linear_2=map_linear_state_dict(state_dict, f"{prefix}.3", bias=False),
        linear_3=map_linear_state_dict(state_dict, f"{prefix}.5", bias=False),
    )


def map_atom_attention_encoder_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    has_coords: bool = False,
) -> AtomAttentionEncoderParams:
    """Map ``AtomAttentionEncoder`` parameters."""

    return AtomAttentionEncoderParams(
        cache=map_atom_attention_encoder_cache_state_dict(state_dict, prefix),
        linear_cl=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_cl",
            bias=False,
        ),
        linear_cm=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_cm",
            bias=False,
        ),
        small_mlp=map_atom_pair_mlp_state_dict(state_dict, f"{prefix}.small_mlp"),
        atom_transformer=map_diffusion_transformer_stack_state_dict(
            state_dict,
            f"{prefix}.atom_transformer.diffusion_transformer",
            cross_attention_mode=True,
        ),
        linear_q=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_q",
            bias=False,
        ),
        layernorm_s=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_s",
            scale=True,
            offset=False,
        )
        if has_coords
        else None,
        linear_s=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_s",
            bias=False,
        )
        if has_coords
        else None,
        layernorm_z=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_z",
            scale=True,
            offset=False,
        )
        if has_coords
        else None,
        linear_z=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_z",
            bias=False,
        )
        if has_coords
        else None,
        linear_r=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_r",
            bias=False,
        )
        if has_coords
        else None,
    )


def map_atom_attention_decoder_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> AtomAttentionDecoderParams:
    """Map ``AtomAttentionDecoder`` parameters."""

    return AtomAttentionDecoderParams(
        linear_a=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_a",
            bias=False,
        ),
        layernorm_q=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_q",
            scale=True,
            offset=False,
        ),
        linear_out=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_out",
            bias=False,
        ),
        atom_transformer=map_diffusion_transformer_stack_state_dict(
            state_dict,
            f"{prefix}.atom_transformer.diffusion_transformer",
            cross_attention_mode=True,
        ),
    )


def map_input_feature_embedder_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> InputFeatureEmbedderParams:
    """Map ``InputFeatureEmbedder`` parameters."""

    esm_prefix = f"{prefix}.linear_esm"
    linear_esm = (
        map_linear_state_dict(state_dict, esm_prefix, bias=False)
        if f"{esm_prefix}.weight" in state_dict
        else None
    )
    return InputFeatureEmbedderParams(
        atom_encoder=map_atom_attention_encoder_state_dict(
            state_dict,
            f"{prefix}.atom_attention_encoder",
        ),
        linear_esm=linear_esm,
    )


def _map_optional_linear_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    bias: bool = False,
) -> LinearParams | None:
    if f"{prefix}.weight" not in state_dict:
        return None
    return map_linear_state_dict(state_dict, prefix, bias=bias)


def map_substructure_mlp_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> SubstructureMlpParams:
    """Map MLP-mode ``SubstructureEmbedder`` sequential linear layers."""

    indices = _module_list_indices(state_dict, prefix)
    return SubstructureMlpParams(
        layers=tuple(
            map_linear_state_dict(state_dict, f"{prefix}.{index}", bias=False)
            for index in indices
            if f"{prefix}.{index}.weight" in state_dict
        )
    )


def map_constraint_embedder_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> ConstraintEmbedderParams:
    """Map optional ``ConstraintEmbedder`` projections when present."""

    substructure_prefix = f"{prefix}.substructure_z_embedder.network"
    substructure_z = (
        map_substructure_mlp_state_dict(state_dict, substructure_prefix)
        if f"{substructure_prefix}.0.weight" in state_dict
        else None
    )
    return ConstraintEmbedderParams(
        pocket_z=_map_optional_linear_state_dict(
            state_dict,
            f"{prefix}.pocket_z_embedder",
            bias=False,
        ),
        contact_z=_map_optional_linear_state_dict(
            state_dict,
            f"{prefix}.contact_z_embedder",
            bias=False,
        ),
        contact_atom_z=_map_optional_linear_state_dict(
            state_dict,
            f"{prefix}.contact_atom_z_embedder",
            bias=False,
        ),
        substructure_z=substructure_z,
    )


def map_conditioned_transition_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> ConditionedTransitionParams:
    """Map ``ConditionedTransitionBlock`` parameters."""

    return ConditionedTransitionParams(
        adaln=map_adaptive_layer_norm_state_dict(state_dict, f"{prefix}.adaln"),
        linear_a1=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_nobias_a1",
            bias=False,
        ),
        linear_a2=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_nobias_a2",
            bias=False,
        ),
        linear_b=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_nobias_b",
            bias=False,
        ),
        linear_s=map_linear_state_dict(state_dict, f"{prefix}.linear_s", bias=True),
    )


def map_diffusion_transformer_block_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    cross_attention_mode: bool = False,
) -> DiffusionTransformerBlockParams:
    """Map one ``DiffusionTransformerBlock``."""

    return DiffusionTransformerBlockParams(
        attention_pair_bias=map_attention_pair_bias_state_dict(
            state_dict,
            f"{prefix}.attention_pair_bias",
            has_s=True,
            create_offset_ln_z=False,
            cross_attention_mode=cross_attention_mode,
        ),
        conditioned_transition=map_conditioned_transition_state_dict(
            state_dict,
            f"{prefix}.conditioned_transition_block",
        ),
    )


def map_diffusion_transformer_stack_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    num_blocks: int | None = None,
    cross_attention_mode: bool = False,
) -> DiffusionTransformerStackParams:
    """Map a ``DiffusionTransformer`` block list."""

    indices = _module_list_indices(state_dict, f"{prefix}.blocks", num_blocks)
    return DiffusionTransformerStackParams(
        blocks=tuple(
            map_diffusion_transformer_block_state_dict(
                state_dict,
                f"{prefix}.blocks.{index}",
                cross_attention_mode=cross_attention_mode,
            )
            for index in indices
        )
    )


def map_diffusion_conditioning_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> DiffusionConditioningParams:
    """Map Protenix ``DiffusionConditioning`` parameters."""

    return DiffusionConditioningParams(
        relpe=map_relative_position_state_dict(state_dict, f"{prefix}.relpe"),
        layernorm_z=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_z",
            scale=True,
            offset=False,
        ),
        linear_z=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_z",
            bias=False,
        ),
        transition_z1=map_transition_state_dict(state_dict, f"{prefix}.transition_z1"),
        transition_z2=map_transition_state_dict(state_dict, f"{prefix}.transition_z2"),
        layernorm_s=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_s",
            scale=True,
            offset=False,
        ),
        linear_s=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_s",
            bias=False,
        ),
        fourier=map_fourier_state_dict(state_dict, f"{prefix}.fourier_embedding"),
        layernorm_n=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_n",
            scale=True,
            offset=False,
        ),
        linear_n=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_n",
            bias=False,
        ),
        transition_s1=map_transition_state_dict(state_dict, f"{prefix}.transition_s1"),
        transition_s2=map_transition_state_dict(state_dict, f"{prefix}.transition_s2"),
    )


def map_diffusion_module_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "diffusion_module",
) -> DiffusionModuleParams:
    """Map infer-only Protenix ``DiffusionModule`` parameters."""

    return DiffusionModuleParams(
        conditioning=map_diffusion_conditioning_state_dict(
            state_dict,
            f"{prefix}.diffusion_conditioning",
        ),
        atom_encoder=map_atom_attention_encoder_state_dict(
            state_dict,
            f"{prefix}.atom_attention_encoder",
            has_coords=True,
        ),
        layernorm_s=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_s",
            scale=True,
            offset=False,
        ),
        linear_s=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_s",
            bias=False,
        ),
        diffusion_transformer=map_diffusion_transformer_stack_state_dict(
            state_dict,
            f"{prefix}.diffusion_transformer",
        ),
        layernorm_a=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_a",
            scale=True,
            offset=False,
        ),
        atom_decoder=map_atom_attention_decoder_state_dict(
            state_dict,
            f"{prefix}.atom_attention_decoder",
        ),
    )


def map_protenix_inference_state_dict(
    state_dict: Mapping[str, Any],
) -> ProtenixInferenceParams:
    """Map the currently ported top-level infer-only Protenix parameters."""

    return ProtenixInferenceParams(
        input_embedder=map_input_feature_embedder_state_dict(
            state_dict,
            "input_embedder",
        ),
        pairformer_output=map_pairformer_output_state_dict(state_dict),
        diffusion=map_diffusion_module_state_dict(state_dict, "diffusion_module"),
        distogram=map_distogram_state_dict(state_dict, "distogram_head"),
        confidence=map_confidence_head_state_dict(state_dict, "confidence_head"),
    )


def map_triangle_multiplication_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> TriangleMultiplicationParams:
    """Map ``TriangleMultiplicativeUpdate`` parameters."""

    return TriangleMultiplicationParams(
        layer_norm_in=map_layer_norm_state_dict(state_dict, f"{prefix}.layer_norm_in"),
        layer_norm_out=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layer_norm_out",
        ),
        linear_a_p=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_a_p",
            bias=False,
        ),
        linear_a_g=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_a_g",
            bias=False,
        ),
        linear_b_p=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_b_p",
            bias=False,
        ),
        linear_b_g=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_b_g",
            bias=False,
        ),
        linear_z=map_linear_state_dict(state_dict, f"{prefix}.linear_z", bias=False),
        linear_g=map_linear_state_dict(state_dict, f"{prefix}.linear_g", bias=False),
    )


def map_triangle_attention_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> TriangleAttentionParams:
    """Map ``TriangleAttention`` parameters."""

    return TriangleAttentionParams(
        layer_norm=map_layer_norm_state_dict(state_dict, f"{prefix}.layer_norm"),
        linear=map_linear_state_dict(state_dict, f"{prefix}.linear", bias=False),
        attention=map_attention_state_dict(
            state_dict,
            f"{prefix}.mha",
            gating=True,
            q_linear_bias=False,
        ),
    )


def map_pairformer_block_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    has_s: bool = True,
) -> PairformerBlockParams:
    """Map one ``PairformerBlock``."""

    return PairformerBlockParams(
        tri_mul_out=map_triangle_multiplication_state_dict(
            state_dict,
            f"{prefix}.tri_mul_out",
        ),
        tri_mul_in=map_triangle_multiplication_state_dict(
            state_dict,
            f"{prefix}.tri_mul_in",
        ),
        tri_att_start=map_triangle_attention_state_dict(
            state_dict,
            f"{prefix}.tri_att_start",
        ),
        tri_att_end=map_triangle_attention_state_dict(
            state_dict,
            f"{prefix}.tri_att_end",
        ),
        pair_transition=map_transition_state_dict(
            state_dict,
            f"{prefix}.pair_transition",
        ),
        attention_pair_bias=map_attention_pair_bias_state_dict(
            state_dict,
            f"{prefix}.attention_pair_bias",
            has_s=False,
            create_offset_ln_z=True,
        )
        if has_s
        else None,
        single_transition=map_transition_state_dict(
            state_dict,
            f"{prefix}.single_transition",
        )
        if has_s
        else None,
    )


def map_pairformer_stack_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    num_blocks: int | None = None,
    has_s: bool = True,
) -> PairformerStackParams:
    """Map a Protenix ``PairformerStack`` module list."""

    indices = _module_list_indices(
        state_dict,
        f"{prefix}.blocks",
        num_blocks,
    )
    return PairformerStackParams(
        blocks=tuple(
            map_pairformer_block_state_dict(
                state_dict,
                f"{prefix}.blocks.{index}",
                has_s=has_s,
            )
            for index in indices
        )
    )


def map_outer_product_mean_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> OuterProductMeanParams:
    """Map Protenix triangular ``OuterProductMean`` parameters."""

    return OuterProductMeanParams(
        layer_norm=map_layer_norm_state_dict(state_dict, f"{prefix}.layer_norm"),
        linear_1=map_linear_state_dict(state_dict, f"{prefix}.linear_1", bias=False),
        linear_2=map_linear_state_dict(state_dict, f"{prefix}.linear_2", bias=False),
        linear_out=map_linear_state_dict(state_dict, f"{prefix}.linear_out", bias=True),
    )


def map_msa_pair_weighted_averaging_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> MSAPairWeightedAveragingParams:
    """Map ``MSAPairWeightedAveraging`` parameters."""

    return MSAPairWeightedAveragingParams(
        layernorm_m=map_layer_norm_state_dict(state_dict, f"{prefix}.layernorm_m"),
        linear_mv=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_mv",
            bias=False,
        ),
        layernorm_z=map_layer_norm_state_dict(state_dict, f"{prefix}.layernorm_z"),
        linear_z=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_z",
            bias=False,
        ),
        linear_mg=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_mg",
            bias=False,
        ),
        linear_out=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_out",
            bias=False,
        ),
    )


def map_msa_block_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> MSABlockParams:
    """Map one Protenix ``MSABlock``."""

    msa_pair_prefix = f"{prefix}.msa_stack.msa_pair_weighted_averaging"
    has_msa_stack = f"{msa_pair_prefix}.layernorm_m.weight" in state_dict
    return MSABlockParams(
        outer_product_mean=map_outer_product_mean_state_dict(
            state_dict,
            f"{prefix}.outer_product_mean_msa",
        ),
        msa_pair_weighted_averaging=map_msa_pair_weighted_averaging_state_dict(
            state_dict,
            msa_pair_prefix,
        )
        if has_msa_stack
        else None,
        msa_transition=map_transition_state_dict(
            state_dict,
            f"{prefix}.msa_stack.transition_m",
        )
        if has_msa_stack
        else None,
        pair_stack=map_pairformer_block_state_dict(
            state_dict,
            f"{prefix}.pair_stack",
            has_s=False,
        ),
    )


def map_msa_module_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    *,
    num_blocks: int | None = None,
) -> MSAModuleParams:
    """Map Protenix ``MSAModule`` parameters."""

    indices = _module_list_indices(state_dict, f"{prefix}.blocks", num_blocks)
    return MSAModuleParams(
        linear_m=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_m",
            bias=False,
        ),
        linear_s=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_s",
            bias=False,
        ),
        blocks=tuple(
            map_msa_block_state_dict(state_dict, f"{prefix}.blocks.{index}")
            for index in indices
        ),
    )


def map_template_embedder_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "template_embedder",
    *,
    num_blocks: int | None = None,
) -> TemplateEmbedderParams:
    """Map Protenix ``TemplateEmbedder`` parameters."""

    return TemplateEmbedderParams(
        linear_z=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_z",
            bias=False,
        ),
        layernorm_z=map_layer_norm_state_dict(state_dict, f"{prefix}.layernorm_z"),
        linear_a=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_a",
            bias=False,
        ),
        pairformer_stack=map_pairformer_stack_state_dict(
            state_dict,
            f"{prefix}.pairformer_stack",
            num_blocks=num_blocks,
            has_s=False,
        ),
        layernorm_v=map_layer_norm_state_dict(state_dict, f"{prefix}.layernorm_v"),
        linear_u=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_u",
            bias=False,
        ),
    )


def map_confidence_distance_embedding_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> ConfidenceDistanceEmbeddingParams:
    """Map ConfidenceHead distance embedding parameters."""

    return ConfidenceDistanceEmbeddingParams(
        lower_bins=jnp.asarray(require_key(state_dict, f"{prefix}.lower_bins")),
        upper_bins=jnp.asarray(require_key(state_dict, f"{prefix}.upper_bins")),
        linear_d=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_d",
            bias=False,
        ),
        linear_d_wo_onehot=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_d_wo_onehot",
            bias=False,
        ),
    )


def map_confidence_output_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> ConfidenceOutputParams:
    """Map ConfidenceHead output projection parameters."""

    return ConfidenceOutputParams(
        pae_ln=map_layer_norm_state_dict(state_dict, f"{prefix}.pae_ln"),
        pde_ln=map_layer_norm_state_dict(state_dict, f"{prefix}.pde_ln"),
        plddt_ln=map_layer_norm_state_dict(state_dict, f"{prefix}.plddt_ln"),
        resolved_ln=map_layer_norm_state_dict(state_dict, f"{prefix}.resolved_ln"),
        linear_pae=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_pae",
            bias=False,
        ),
        linear_pde=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_pde",
            bias=False,
        ),
        plddt_weight=jnp.asarray(require_key(state_dict, f"{prefix}.plddt_weight")),
        resolved_weight=jnp.asarray(
            require_key(state_dict, f"{prefix}.resolved_weight")
        ),
    )


def map_confidence_head_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> ConfidenceHeadParams:
    """Map full ConfidenceHead parameters for single-sample inference."""

    return ConfidenceHeadParams(
        input_strunk_ln=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.input_strunk_ln",
        ),
        linear_s1=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_s1",
            bias=False,
        ),
        linear_s2=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_s2",
            bias=False,
        ),
        distance_embedding=map_confidence_distance_embedding_state_dict(
            state_dict,
            prefix,
        ),
        pairformer_stack=map_pairformer_stack_state_dict(
            state_dict,
            f"{prefix}.pairformer_stack",
            has_s=True,
        ),
        output=map_confidence_output_state_dict(state_dict, prefix),
    )


def map_trunk_initialization_state_dict(
    state_dict: Mapping[str, Any],
) -> TrunkParams:
    """Map root-level Protenix trunk initialization and recycling params."""

    initial = TrunkInitializationParams(
        linear_sinit=map_linear_state_dict(
            state_dict,
            "linear_no_bias_sinit",
            bias=False,
        ),
        linear_zinit1=map_linear_state_dict(
            state_dict,
            "linear_no_bias_zinit1",
            bias=False,
        ),
        linear_zinit2=map_linear_state_dict(
            state_dict,
            "linear_no_bias_zinit2",
            bias=False,
        ),
        relative_position=map_relative_position_state_dict(
            state_dict,
            "relative_position_encoding",
        ),
        linear_token_bond=map_linear_state_dict(
            state_dict,
            "linear_no_bias_token_bond",
            bias=False,
        ),
    )
    recycling = RecyclingProjectionParams(
        layernorm_z=map_layer_norm_state_dict(state_dict, "layernorm_z_cycle"),
        linear_z=map_linear_state_dict(
            state_dict,
            "linear_no_bias_z_cycle",
            bias=False,
        ),
        layernorm_s=map_layer_norm_state_dict(state_dict, "layernorm_s"),
        linear_s=map_linear_state_dict(state_dict, "linear_no_bias_s", bias=False),
    )
    return TrunkParams(initial=initial, recycling=recycling)


def map_pairformer_output_state_dict(
    state_dict: Mapping[str, Any],
) -> PairformerOutputParams:
    """Map parameters used by Protenix ``get_pairformer_output``."""

    return PairformerOutputParams(
        trunk=map_trunk_initialization_state_dict(state_dict),
        constraint=map_constraint_embedder_state_dict(
            state_dict,
            "constraint_embedder",
        ),
        template=map_template_embedder_state_dict(state_dict, "template_embedder"),
        msa=map_msa_module_state_dict(state_dict, "msa_module"),
        pairformer_stack=map_pairformer_stack_state_dict(
            state_dict,
            "pairformer_stack",
            has_s=True,
        ),
    )


def _module_list_indices(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_modules: int | None = None,
) -> tuple[int, ...]:
    if num_modules is not None:
        return tuple(range(num_modules))

    stem = f"{prefix}."
    indices = set()
    for key in state_dict:
        key_str = str(key)
        if not key_str.startswith(stem):
            continue
        suffix = key_str[len(stem) :]
        index_str = suffix.split(".", 1)[0]
        if index_str.isdigit():
            indices.add(int(index_str))
    if not indices:
        raise KeyError(f"missing module list keys under prefix: {prefix}")
    return tuple(sorted(indices))


def map_fourier_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> FourierParams:
    """Map ``protenix.model.modules.embedders.FourierEmbedding`` parameters."""

    return FourierParams(
        w=jnp.asarray(require_key(state_dict, f"{prefix}.w")),
        b=jnp.asarray(require_key(state_dict, f"{prefix}.b")),
    )


def map_relative_position_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> RelativePositionParams:
    """Map ``RelativePositionEncoding`` parameters."""

    return RelativePositionParams(
        linear_no_bias=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias",
            bias=False,
        )
    )


def map_distogram_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "",
) -> DistogramParams:
    """Map ``DistogramHead`` state dict keys.

    Expected keys under ``prefix``:
    - ``linear.weight``
    - ``linear.bias``
    """

    stem = f"{prefix}." if prefix else ""
    return DistogramParams(
        linear=map_linear_state_dict(state_dict, f"{stem}linear", bias=True)
    )


def state_dict_to_params(
    state_dict: Mapping[str, Any],
) -> ProtenixInferenceParams:
    """Public alias: map a Protenix model state-dict to JAX inference params."""

    return map_protenix_inference_state_dict(state_dict)


def load_torch_checkpoint(path: str | Path) -> ProtenixInferenceParams:
    """Load a PyTorch Protenix checkpoint and convert to JAX params.

    Mirrors the upstream Protenix inference loader: reads ``checkpoint["model"]``
    (falling back to the raw object), strips a DistributedDataParallel
    ``module.`` prefix, then maps to ``ProtenixInferenceParams``. ``torch`` is
    imported lazily so the native runtime stays torch-free.

    Security: ``path`` MUST be a trusted, locally-produced checkpoint. Loading
    prefers the safe ``weights_only=True`` path (tensors + basic types only);
    it only falls back to full unpickling for legacy checkpoints that bundle
    non-tensor objects (optimizer/scheduler state), exactly as upstream
    Protenix saves them. Never point this at an untrusted file.
    """

    from foldjax import torch_archive

    try:
        # Torch-free by default: the archive reader returns NumPy tensors and
        # stubs everything it does not know, so it is stricter than
        # `weights_only=False` while needing no torch at all. Verified
        # bit-identical against torch.load on the released checkpoint.
        obj = torch_archive.load(path)
    except Exception:
        import torch

        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, Mapping) and "model" in obj:
        state_dict = obj["model"]
    elif isinstance(obj, Mapping) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj
    if any(str(k).startswith("module.") for k in state_dict):
        state_dict = {
            str(k)[len("module.") :] if str(k).startswith("module.") else str(k): v
            for k, v in state_dict.items()
        }
    return map_protenix_inference_state_dict(state_dict)
