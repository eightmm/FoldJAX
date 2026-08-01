"""Top-level orchestration for the Chai-1 diffusion denoiser.

This module owns the five tensors stored directly on ``diffusion_module`` and
also exposes a complete 343-tensor mapper/callable for production use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from foldjax.models.chai.models.diffusion_atom_decoder import (
    AtomDecoderParams,
    atom_decoder_forward,
    map_atom_decoder,
)
from foldjax.models.chai.models.diffusion_atom_encoder import (
    DiffusionAtomEncoderOutput,
    DiffusionAtomEncoderParams,
    diffusion_atom_encoder_with_aux,
    map_diffusion_atom_encoder,
)
from foldjax.models.chai.models.diffusion_conditioning import (
    DiffusionConditioningParams,
    diffusion_conditioning,
    map_diffusion_conditioning,
)
from foldjax.models.chai.models.diffusion_transformer import (
    DiffusionTransformerParams,
    diffusion_transformer_stack,
    map_diffusion_transformer,
)
from foldjax.models.chai.models.primitives import layer_norm, linear

_CONDITIONING_PREFIX = "diffusion_conditioning."
_ATOM_ENCODER_PREFIX = "atom_attention_encoder."
_TRANSFORMER_PREFIX = "diffusion_transformer."
_ATOM_DECODER_PREFIX = "atom_attention_decoder."
_TOP_LEVEL_SHAPES = {
    "structure_cond_to_token_structure_proj.weight": (768, 384),
    "post_attn_layernorm.weight": (768,),
    "post_attn_layernorm.bias": (768,),
    "post_atom_cond_layernorm.weight": (128,),
    "post_atom_cond_layernorm.bias": (128,),
}
_SIGMA_DATA_SQUARED = 256.0
_SIGMA_DATA = 16.0


class DiffusionDenoiserParams(NamedTuple):
    structure_projection_weight: jnp.ndarray
    post_attention_norm_weight: jnp.ndarray
    post_attention_norm_bias: jnp.ndarray
    post_atom_condition_norm_weight: jnp.ndarray
    post_atom_condition_norm_bias: jnp.ndarray


AtomEncoderOutput = DiffusionAtomEncoderOutput


class FullDiffusionDenoiserParams(NamedTuple):
    """All 343 tensors in the official Chai diffusion component."""

    conditioning: DiffusionConditioningParams
    atom_encoder: DiffusionAtomEncoderParams
    transformer: DiffusionTransformerParams
    atom_decoder: AtomDecoderParams
    top_level: DiffusionDenoiserParams


class DiffusionStateDecomposition(NamedTuple):
    conditioning: frozenset[str]
    atom_encoder: frozenset[str]
    transformer: frozenset[str]
    atom_decoder: frozenset[str]
    top_level: frozenset[str]
    unknown: frozenset[str]


def decompose_diffusion_state(
    state: Mapping[str, Any],
) -> DiffusionStateDecomposition:
    """Partition every state key by its official denoiser owner."""
    conditioning = frozenset(
        key for key in state if key.startswith(_CONDITIONING_PREFIX)
    )
    atom_encoder = frozenset(
        key for key in state if key.startswith(_ATOM_ENCODER_PREFIX)
    )
    transformer = frozenset(
        key for key in state if key.startswith(_TRANSFORMER_PREFIX)
    )
    atom_decoder = frozenset(
        key for key in state if key.startswith(_ATOM_DECODER_PREFIX)
    )
    top_level = frozenset(key for key in state if key in _TOP_LEVEL_SHAPES)
    known = conditioning | atom_encoder | transformer | atom_decoder | top_level
    return DiffusionStateDecomposition(
        conditioning=conditioning,
        atom_encoder=atom_encoder,
        transformer=transformer,
        atom_decoder=atom_decoder,
        top_level=top_level,
        unknown=frozenset(state) - known,
    )


def map_diffusion_denoiser(
    state: Mapping[str, Any],
) -> tuple[DiffusionDenoiserParams, frozenset[str]]:
    """Map the five top-level tensors and report every child-owned tensor."""
    decomposition = decompose_diffusion_state(state)
    expected = set(_TOP_LEVEL_SHAPES)
    missing = sorted(expected - decomposition.top_level)
    if missing or decomposition.unknown:
        raise ValueError(
            "diffusion denoiser state mismatch: "
            f"missing={missing}, unknown={sorted(decomposition.unknown)}"
        )
    for key, shape in _TOP_LEVEL_SHAPES.items():
        actual_shape = tuple(state[key].shape)
        if actual_shape != shape:
            raise ValueError(
                f"wrong shape for {key}: expected {shape}, got {actual_shape}"
            )

    params = DiffusionDenoiserParams(
        structure_projection_weight=jnp.asarray(
            state["structure_cond_to_token_structure_proj.weight"]
        ),
        post_attention_norm_weight=jnp.asarray(
            state["post_attn_layernorm.weight"]
        ),
        post_attention_norm_bias=jnp.asarray(state["post_attn_layernorm.bias"]),
        post_atom_condition_norm_weight=jnp.asarray(
            state["post_atom_cond_layernorm.weight"]
        ),
        post_atom_condition_norm_bias=jnp.asarray(
            state["post_atom_cond_layernorm.bias"]
        ),
    )
    unconsumed = (
        decomposition.conditioning
        | decomposition.atom_encoder
        | decomposition.transformer
        | decomposition.atom_decoder
    )
    return params, unconsumed


def map_full_diffusion_denoiser(
    state: Mapping[str, Any],
) -> FullDiffusionDenoiserParams:
    """Map the complete official component and reject incomplete checkpoints."""
    decomposition = decompose_diffusion_state(state)
    actual_counts = tuple(
        len(group)
        for group in (
            decomposition.conditioning,
            decomposition.atom_encoder,
            decomposition.transformer,
            decomposition.atom_decoder,
            decomposition.top_level,
        )
    )
    expected_counts = (31, 46, 224, 37, 5)
    if decomposition.unknown or actual_counts != expected_counts:
        raise ValueError(
            "full diffusion state mismatch: "
            f"counts={actual_counts}, expected={expected_counts}, "
            f"unknown={sorted(decomposition.unknown)}"
        )
    top_level, unconsumed = map_diffusion_denoiser(state)
    if len(unconsumed) != 338:
        raise ValueError(f"expected 338 child tensors, got {len(unconsumed)}")
    return FullDiffusionDenoiserParams(
        conditioning=map_diffusion_conditioning(state),
        atom_encoder=map_diffusion_atom_encoder(state),
        transformer=map_diffusion_transformer(state),
        atom_decoder=map_atom_decoder(state),
        top_level=top_level,
    )


def diffusion_denoiser(
    token_single_initial_repr: jnp.ndarray,
    token_pair_initial_repr: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    token_pair_trunk_repr: jnp.ndarray,
    atom_single_input_feats: jnp.ndarray,
    atom_block_pair_input_feats: jnp.ndarray,
    atom_single_mask: jnp.ndarray,
    atom_block_pair_mask: jnp.ndarray,
    token_single_mask: jnp.ndarray,
    block_indices_h: jnp.ndarray,
    block_indices_w: jnp.ndarray,
    atom_noised_coords: jnp.ndarray,
    noise_sigma: jnp.ndarray,
    atom_token_indices: jnp.ndarray,
    params: DiffusionDenoiserParams,
    *,
    conditioning_fn: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
    atom_encoder_fn: Callable[..., AtomEncoderOutput],
    transformer_fn: Callable[..., jnp.ndarray],
    atom_decoder_fn: Callable[..., jnp.ndarray],
) -> jnp.ndarray:
    """Run the exact top-level denoiser graph using injected model slices.

    The output follows the official component ABI and flattens batch and
    diffusion-sample axes to ``(batch * samples, atoms, 3)``.
    """
    conditioned_single, conditioned_pair = conditioning_fn(
        token_single_initial_repr,
        token_pair_initial_repr,
        token_single_trunk_repr,
        token_pair_trunk_repr,
        noise_sigma,
    )

    sigma = noise_sigma[..., None, None]
    scaled_coords = atom_noised_coords / jnp.sqrt(
        sigma**2 + _SIGMA_DATA_SQUARED
    )
    encoder_output = atom_encoder_fn(
        atom_single_input_feats,
        token_single_trunk_repr,
        conditioned_pair,
        scaled_coords,
        atom_block_pair_input_feats,
        atom_single_mask,
        atom_block_pair_mask,
        block_indices_h,
        block_indices_w,
        atom_token_indices,
    )

    structure_condition = linear(
        conditioned_single, params.structure_projection_weight
    )
    transformer_input = encoder_output.token_single + structure_condition
    transformed = transformer_fn(
        transformer_input,
        conditioned_single,
        conditioned_pair,
        token_single_mask,
    )
    transformed = layer_norm(
        transformed,
        params.post_attention_norm_weight,
        params.post_attention_norm_bias,
    )
    atom_condition = layer_norm(
        encoder_output.atom_condition,
        params.post_atom_condition_norm_weight,
        params.post_atom_condition_norm_bias,
    )
    unit_update = atom_decoder_fn(
        transformed,
        encoder_output.atom_single,
        atom_condition,
        encoder_output.atom_pair,
        atom_single_mask,
        atom_block_pair_mask,
        atom_token_indices,
    )

    output_scale = sigma * _SIGMA_DATA / jnp.sqrt(
        sigma**2 + _SIGMA_DATA_SQUARED
    )
    skip_scale = _SIGMA_DATA_SQUARED / (
        sigma**2 + _SIGMA_DATA_SQUARED
    )
    output = atom_noised_coords * skip_scale + unit_update * output_scale
    batch, samples, atoms, coordinates = output.shape
    return output.reshape(batch * samples, atoms, coordinates)


def full_diffusion_denoiser(
    token_single_initial_repr: jnp.ndarray,
    token_pair_initial_repr: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    token_pair_trunk_repr: jnp.ndarray,
    atom_single_input_feats: jnp.ndarray,
    atom_block_pair_input_feats: jnp.ndarray,
    atom_single_mask: jnp.ndarray,
    atom_block_pair_mask: jnp.ndarray,
    token_single_mask: jnp.ndarray,
    block_indices_h: jnp.ndarray,
    block_indices_w: jnp.ndarray,
    atom_noised_coords: jnp.ndarray,
    noise_sigma: jnp.ndarray,
    atom_token_indices: jnp.ndarray,
    params: FullDiffusionDenoiserParams,
    *,
    query_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run all official diffusion slices without dependency injection."""

    def conditioning_fn(*args):
        return diffusion_conditioning(*args, params.conditioning)

    def atom_encoder_fn(*args):
        return diffusion_atom_encoder_with_aux(*args, params.atom_encoder)

    def transformer_fn(*args):
        return diffusion_transformer_stack(
            *args,
            params.transformer,
            query_chunk_size=query_chunk_size,
        )

    def atom_decoder_fn(*args):
        return atom_decoder_forward(*args, params.atom_decoder)

    return diffusion_denoiser(
        token_single_initial_repr,
        token_pair_initial_repr,
        token_single_trunk_repr,
        token_pair_trunk_repr,
        atom_single_input_feats,
        atom_block_pair_input_feats,
        atom_single_mask,
        atom_block_pair_mask,
        token_single_mask,
        block_indices_h,
        block_indices_w,
        atom_noised_coords,
        noise_sigma,
        atom_token_indices,
        params.top_level,
        conditioning_fn=conditioning_fn,
        atom_encoder_fn=atom_encoder_fn,
        transformer_fn=transformer_fn,
        atom_decoder_fn=atom_decoder_fn,
    )
