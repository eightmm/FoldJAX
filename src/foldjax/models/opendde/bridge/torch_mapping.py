"""PyTorch state-dict to JAX mappings for OpenDDE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from foldjax.models.opendde.models.diffusion_conditioning import (
    DiffusionConditioningParams,
)
from foldjax.models.opendde.models.model import OpenDDEInferenceParams
from foldjax.models.opendde.models.structural_tokens import (
    StructuralTokenExpanderParams,
)
from foldjax.models.protenix.bridge.torch_mapping import (
    map_confidence_head_state_dict,
    map_input_feature_embedder_state_dict,
    map_layer_norm_state_dict,
    map_linear_state_dict,
    map_pairformer_output_state_dict,
    map_pairformer_stack_state_dict,
    require_key,
)
from foldjax.models.protenix.bridge.torch_mapping import (
    map_diffusion_conditioning_state_dict as _map_diffusion_conditioning_state_dict,
)
from foldjax.models.protenix.bridge.torch_mapping import (
    map_diffusion_module_state_dict as _map_diffusion_module_state_dict,
)
from foldjax.models.protenix.bridge.torch_mapping import (
    map_distogram_state_dict as _map_distogram_state_dict,
)
from foldjax.models.protenix.models.diffusion.diffusion import DiffusionModuleParams
from foldjax.models.protenix.models.heads.head import DistogramParams
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams

RELEASED_PAIR_CHANNELS = 384
RELEASED_DISTOGRAM_BINS = 96
RELEASED_STRUCTURAL_TOKEN_ROLES = 7
RELEASED_STRUCTURAL_REFINER_BLOCKS = 4
RELEASED_SINGLE_CHANNELS = 384
RELEASED_DIFFUSION_PAIR_CHANNELS = 128


def map_distogram_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "distogram_head",
) -> DistogramParams:
    """Map an OpenDDE distogram head using its native checkpoint keys."""

    return _map_distogram_state_dict(state_dict, prefix)


def map_released_distogram_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "distogram_head",
) -> DistogramParams:
    """Map and validate the released OpenDDE-v1 distogram contract."""

    params = map_distogram_state_dict(state_dict, prefix)
    expected_weight = (RELEASED_DISTOGRAM_BINS, RELEASED_PAIR_CHANNELS)
    expected_bias = (RELEASED_DISTOGRAM_BINS,)
    if tuple(params.linear.weight.shape) != expected_weight:
        raise ValueError(
            "released OpenDDE distogram weight expected "
            f"{expected_weight}, got {tuple(params.linear.weight.shape)}"
        )
    if params.linear.bias is None or tuple(params.linear.bias.shape) != expected_bias:
        actual = None if params.linear.bias is None else tuple(params.linear.bias.shape)
        raise ValueError(
            f"released OpenDDE distogram bias expected {expected_bias}, got {actual}"
        )
    return params


def map_structural_token_expander_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "structural_token_expander",
    *,
    n_roles: int = RELEASED_STRUCTURAL_TOKEN_ROLES,
) -> StructuralTokenExpanderParams:
    """Map OpenDDE's full structural role-pair expander."""

    pair_weights = jnp.stack(
        [
            jnp.asarray(
                require_key(state_dict, f"{prefix}.pair_block_proj.{index}.weight")
            )
            for index in range(n_roles * n_roles)
        ]
    )
    pair_weights = pair_weights.reshape(
        n_roles,
        n_roles,
        *pair_weights.shape[-2:],
    )
    return StructuralTokenExpanderParams(
        single_split_layer_norm=map_layer_norm_state_dict(
            state_dict, f"{prefix}.single_split_mlp.0"
        ),
        single_split_linear_in=map_linear_state_dict(
            state_dict,
            f"{prefix}.single_split_mlp.1",
            bias=False,
        ),
        single_split_linear_out=map_linear_state_dict(
            state_dict,
            f"{prefix}.single_split_mlp.3",
            bias=False,
        ),
        single_input_role_embedding=jnp.asarray(
            require_key(state_dict, f"{prefix}.single_input_role_embedding.weight")
        ),
        single_role_embedding=jnp.asarray(
            require_key(state_dict, f"{prefix}.single_role_embedding.weight")
        ),
        pair_block_proj=pair_weights,
        same_parent_embedding=jnp.asarray(
            require_key(state_dict, f"{prefix}.same_parent_embedding.weight")
        ),
        same_residue_twin_embedding=jnp.asarray(
            require_key(state_dict, f"{prefix}.same_residue_twin_embedding.weight")
        ),
        prev_bb_chain_embedding=jnp.asarray(
            require_key(state_dict, f"{prefix}.prev_bb_chain_embedding.weight")
        ),
        next_bb_chain_embedding=jnp.asarray(
            require_key(state_dict, f"{prefix}.next_bb_chain_embedding.weight")
        ),
        role_pair_type_embedding=jnp.asarray(
            require_key(state_dict, f"{prefix}.role_pair_type_embedding.weight")
        ),
        attn_bias_same_parent=jnp.asarray(
            require_key(state_dict, f"{prefix}.attn_bias_same_parent")
        ),
        attn_bias_same_residue_twin=jnp.asarray(
            require_key(state_dict, f"{prefix}.attn_bias_same_residue_twin")
        ),
        attn_bias_prev_bb_chain=jnp.asarray(
            require_key(state_dict, f"{prefix}.attn_bias_prev_bb_chain")
        ),
        attn_bias_next_bb_chain=jnp.asarray(
            require_key(state_dict, f"{prefix}.attn_bias_next_bb_chain")
        ),
        attn_bias_role_pair_type=jnp.asarray(
            require_key(state_dict, f"{prefix}.attn_bias_role_pair_type")
        ),
    )


def map_structural_refiner_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "structural_token_refiner",
    *,
    num_blocks: int | None = None,
) -> PairformerStackParams:
    """Map OpenDDE's structural-token Pairformer stack."""

    return map_pairformer_stack_state_dict(
        state_dict,
        prefix,
        num_blocks=num_blocks,
        has_s=True,
    )


def map_released_structural_refiner_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "structural_token_refiner",
) -> PairformerStackParams:
    """Map and validate the released four-block structural refiner."""

    params = map_structural_refiner_state_dict(
        state_dict,
        prefix,
        num_blocks=RELEASED_STRUCTURAL_REFINER_BLOCKS,
    )
    for index, block in enumerate(params.blocks):
        pair_shape = tuple(block.tri_mul_out.linear_a_p.weight.shape)
        pair_heads_shape = tuple(block.tri_att_start.linear.weight.shape)
        single_heads_shape = tuple(block.attention_pair_bias.linear_z.weight.shape)
        if pair_shape != (RELEASED_PAIR_CHANNELS, RELEASED_PAIR_CHANNELS):
            raise ValueError(
                f"released structural refiner block {index} pair projection "
                f"expected {(RELEASED_PAIR_CHANNELS, RELEASED_PAIR_CHANNELS)}, "
                f"got {pair_shape}"
            )
        if pair_heads_shape != (12, RELEASED_PAIR_CHANNELS):
            raise ValueError(
                f"released structural refiner block {index} triangle heads "
                f"expected {(12, RELEASED_PAIR_CHANNELS)}, got {pair_heads_shape}"
            )
        if single_heads_shape != (8, RELEASED_PAIR_CHANNELS):
            raise ValueError(
                f"released structural refiner block {index} single heads "
                f"expected {(8, RELEASED_PAIR_CHANNELS)}, got {single_heads_shape}"
            )
        single_shape = tuple(block.single_transition.linear_out.weight.shape)
        if single_shape != (RELEASED_SINGLE_CHANNELS, 4 * RELEASED_SINGLE_CHANNELS):
            raise ValueError(
                f"released structural refiner block {index} single transition "
                f"expected {(RELEASED_SINGLE_CHANNELS, 4 * RELEASED_SINGLE_CHANNELS)}, "
                f"got {single_shape}"
            )
    return params


def map_diffusion_conditioning_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "diffusion_module.diffusion_conditioning",
) -> DiffusionConditioningParams:
    """Map OpenDDE diffusion conditioning, including pair compression."""

    base = _map_diffusion_conditioning_state_dict(state_dict, prefix)
    return DiffusionConditioningParams(
        relpe=base.relpe,
        layernorm_z_trunk=map_layer_norm_state_dict(
            state_dict,
            f"{prefix}.layernorm_z_trunk",
            scale=True,
            offset=False,
        ),
        linear_z_trunk=map_linear_state_dict(
            state_dict,
            f"{prefix}.linear_no_bias_z_trunk",
            bias=False,
        ),
        layernorm_z=base.layernorm_z,
        linear_z=base.linear_z,
        transition_z1=base.transition_z1,
        transition_z2=base.transition_z2,
        layernorm_s=base.layernorm_s,
        linear_s=base.linear_s,
        fourier=base.fourier,
        layernorm_n=base.layernorm_n,
        linear_n=base.linear_n,
        transition_s1=base.transition_s1,
        transition_s2=base.transition_s2,
    )


def map_released_diffusion_conditioning_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "diffusion_module.diffusion_conditioning",
) -> DiffusionConditioningParams:
    """Map and validate OpenDDE-v1 diffusion conditioning dimensions."""

    params = map_diffusion_conditioning_state_dict(state_dict, prefix)
    return _validate_released_diffusion_conditioning_params(params)


def _validate_released_diffusion_conditioning_params(
    params: DiffusionConditioningParams,
) -> DiffusionConditioningParams:
    expected_shapes = {
        "relpe": (
            tuple(params.relpe.linear_no_bias.weight.shape),
            (RELEASED_DIFFUSION_PAIR_CHANNELS, 139),
        ),
        "layernorm_z_trunk": (
            tuple(params.layernorm_z_trunk.weight.shape),
            (RELEASED_PAIR_CHANNELS,),
        ),
        "linear_z_trunk": (
            tuple(params.linear_z_trunk.weight.shape),
            (RELEASED_DIFFUSION_PAIR_CHANNELS, RELEASED_PAIR_CHANNELS),
        ),
        "layernorm_z": (
            tuple(params.layernorm_z.weight.shape),
            (2 * RELEASED_DIFFUSION_PAIR_CHANNELS,),
        ),
        "linear_z": (
            tuple(params.linear_z.weight.shape),
            (
                RELEASED_DIFFUSION_PAIR_CHANNELS,
                2 * RELEASED_DIFFUSION_PAIR_CHANNELS,
            ),
        ),
        "layernorm_s": (
            tuple(params.layernorm_s.weight.shape),
            (RELEASED_SINGLE_CHANNELS + 449,),
        ),
        "linear_s": (
            tuple(params.linear_s.weight.shape),
            (RELEASED_SINGLE_CHANNELS, RELEASED_SINGLE_CHANNELS + 449),
        ),
        "fourier": (tuple(params.fourier.w.shape), (256,)),
        "linear_n": (
            tuple(params.linear_n.weight.shape),
            (RELEASED_SINGLE_CHANNELS, 256),
        ),
    }
    for name, (actual, expected) in expected_shapes.items():
        if actual != expected:
            raise ValueError(
                f"released diffusion conditioning {name} expected "
                f"{expected}, got {actual}"
            )
    return params


def map_diffusion_module_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "diffusion_module",
) -> DiffusionModuleParams:
    """Map the full OpenDDE denoiser with compressed conditioning."""

    params = _map_diffusion_module_state_dict(state_dict, prefix)
    conditioning = map_diffusion_conditioning_state_dict(
        state_dict,
        f"{prefix}.diffusion_conditioning",
    )
    return params._replace(conditioning=conditioning)


def map_released_diffusion_module_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "diffusion_module",
) -> DiffusionModuleParams:
    """Map and validate the released OpenDDE diffusion module."""

    params = map_diffusion_module_state_dict(state_dict, prefix)
    conditioning = _validate_released_diffusion_conditioning_params(params.conditioning)
    return params._replace(conditioning=conditioning)


def map_opendde_inference_state_dict(
    state_dict: Mapping[str, Any],
) -> OpenDDEInferenceParams:
    """Map the released OpenDDE residue/structural dual-branch graph."""

    return OpenDDEInferenceParams(
        input_embedder=map_input_feature_embedder_state_dict(
            state_dict,
            "input_embedder",
        ),
        pairformer_output=map_pairformer_output_state_dict(state_dict),
        structural_expander=map_structural_token_expander_state_dict(state_dict),
        structural_refiner=map_released_structural_refiner_state_dict(state_dict),
        diffusion=map_released_diffusion_module_state_dict(state_dict),
        distogram=map_released_distogram_state_dict(state_dict),
        confidence=map_confidence_head_state_dict(
            state_dict,
            "confidence_head",
        ),
    )
