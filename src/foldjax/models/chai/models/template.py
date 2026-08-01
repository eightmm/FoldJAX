"""Chai trunk template embedding with the fixed four-template bucket."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._stacking import stacked_or_stack
from foldjax.models.chai.models.pairformer import (
    PairformerPairBlockParams,
    map_pairformer_pair_block,
    pairformer_pair_block,
)
from foldjax.models.chai.models.primitives import layer_norm, linear_bf16


class TemplateEmbedderParams(NamedTuple):
    proj_in_norm_weight: jnp.ndarray
    proj_in_norm_bias: jnp.ndarray
    proj_in_weight: jnp.ndarray
    blocks: tuple[PairformerPairBlockParams, ...]
    output_norm_weight: jnp.ndarray
    output_norm_bias: jnp.ndarray
    proj_out_weight: jnp.ndarray


def map_template_embedder(
    state: Mapping[str, Any], prefix: str = ""
) -> TemplateEmbedderParams:
    """Map all 34 official template-embedder tensors."""
    base = f"{prefix}." if prefix else ""
    return TemplateEmbedderParams(
        proj_in_norm_weight=jnp.asarray(state[f"{base}proj_in.0.weight"]),
        proj_in_norm_bias=jnp.asarray(state[f"{base}proj_in.0.bias"]),
        proj_in_weight=jnp.asarray(state[f"{base}proj_in.1.weight"]),
        blocks=tuple(
            map_pairformer_pair_block(
                state, f"{base}pairformer.blocks.{index}"
            )
            for index in range(2)
        ),
        output_norm_weight=jnp.asarray(
            state[f"{base}template_layernorm.weight"]
        ),
        output_norm_bias=jnp.asarray(
            state[f"{base}template_layernorm.bias"]
        ),
        proj_out_weight=jnp.asarray(state[f"{base}proj_out.1.weight"]),
    )


def template_embedding(
    z: jnp.ndarray,
    template_features: jnp.ndarray,
    template_masks: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: TemplateEmbedderParams,
    use_scan: bool = True,
) -> jnp.ndarray:
    """Apply Chai's four-template stack and return the updated pair state."""
    if template_features.shape[1] != 4 or template_masks.shape[1] != 4:
        raise ValueError("Chai trunk requires exactly four templates")
    pair_base = linear_bf16(
        layer_norm(
            z.astype(jnp.float32),
            params.proj_in_norm_weight,
            params.proj_in_norm_bias,
        ),
        params.proj_in_weight,
    )
    combined_mask = template_masks & pair_mask[:, None]
    # The block stack is scanned rather than unrolled, so the module holds one copy
    # of the block body instead of one per template per block.
    stacked_blocks = (
        stacked_or_stack(params.blocks)
        if use_scan and len(params.blocks) > 1
        else None
    )

    outputs = []
    for template_index in range(4):
        value = pair_base + template_features[:, template_index]
        mask = combined_mask[:, template_index]
        if stacked_blocks is None:
            for block in params.blocks:
                value = pairformer_pair_block(value, mask, block)
        else:
            value = jax.lax.scan(
                lambda carry, block: (
                    pairformer_pair_block(carry, mask, block),
                    None,
                ),
                value,
                stacked_blocks,
            )[0]
        outputs.append(value)

    stacked = jnp.stack(outputs, axis=1)
    stacked = layer_norm(
        stacked.astype(jnp.float32),
        params.output_norm_weight,
        params.output_norm_bias,
    )
    stacked = stacked * combined_mask[..., None]
    template_count = jnp.maximum(
        jnp.any(template_masks, axis=(-2, -1)).sum(axis=1), 1
    )
    pooled = jnp.sum(stacked, axis=1, dtype=jnp.float32)
    pooled = pooled / template_count[:, None, None, None]
    return z + linear_bf16(jax.nn.relu(pooled), params.proj_out_weight)
