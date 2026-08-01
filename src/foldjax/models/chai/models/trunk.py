"""Complete Chai trunk mapping and ``forward_256`` orchestration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._stacking import stacked_or_stack
from foldjax.models.chai.models.msa import MSAModuleParams, map_msa_module, msa_module
from foldjax.models.chai.models.pairformer import (
    PairformerBlockParams,
    map_pairformer_block,
    pairformer_block,
)
from foldjax.models.chai.models.primitives import layer_norm, linear_bf16
from foldjax.models.chai.models.template import (
    TemplateEmbedderParams,
    map_template_embedder,
    template_embedding,
)


class RecyclingParams(NamedTuple):
    layer_norm_weight: jnp.ndarray
    layer_norm_bias: jnp.ndarray
    linear_weight: jnp.ndarray


class TrunkParams(NamedTuple):
    single_recycling: RecyclingParams
    pair_recycling: RecyclingParams
    template: TemplateEmbedderParams
    msa: MSAModuleParams
    pairformer_blocks: tuple[PairformerBlockParams, ...]


class _TrackingMapping(Mapping[str, Any]):
    """Read-only mapping that records every state tensor consumed by mappers."""

    def __init__(self, source: Mapping[str, Any]):
        self.source = source
        self.accesses: Counter[str] = Counter()

    def __getitem__(self, key: str) -> Any:
        value = self.source[key]
        self.accesses[key] += 1
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self.source)

    def __len__(self) -> int:
        return len(self.source)


def map_recycling(state: Mapping[str, Any], prefix: str = "") -> RecyclingParams:
    """Map a Chai ``LayerNorm -> biasless Linear`` recycling projection."""
    base = f"{prefix}." if prefix else ""
    return RecyclingParams(
        layer_norm_weight=jnp.asarray(state[f"{base}0.weight"]),
        layer_norm_bias=jnp.asarray(state[f"{base}0.bias"]),
        linear_weight=jnp.asarray(state[f"{base}1.weight"]),
    )


def recycling_projection(x: jnp.ndarray, params: RecyclingParams) -> jnp.ndarray:
    """Apply the official Chai recycling projection without torch."""
    normalized = layer_norm(
        x.astype(jnp.float32),
        params.layer_norm_weight,
        params.layer_norm_bias,
    )
    return linear_bf16(normalized, params.linear_weight)


def map_trunk(state: Mapping[str, Any], *, strict: bool = True) -> TrunkParams:
    """Map the 1,398 tensors in the official Chai ``trunk.pt``.

    In strict mode every provided tensor must be consumed exactly once.  This
    catches both archive drift and accidental duplicate parameter binding.
    """
    tracked = _TrackingMapping(state)
    params = TrunkParams(
        single_recycling=map_recycling(tracked, "token_single_recycle_proj"),
        pair_recycling=map_recycling(tracked, "token_pair_recycle_proj"),
        template=map_template_embedder(tracked, "template_embedder"),
        msa=map_msa_module(tracked, "msa_module"),
        pairformer_blocks=tuple(
            map_pairformer_block(tracked, f"pairformer_stack.blocks.{index}")
            for index in range(48)
        ),
    )
    if strict:
        provided = set(state)
        consumed = set(tracked.accesses)
        missing = sorted(provided - consumed)
        duplicates = sorted(
            key for key, count in tracked.accesses.items() if count != 1
        )
        if missing or duplicates:
            details = []
            if missing:
                details.append(f"unconsumed={missing}")
            if duplicates:
                details.append(f"multiply_consumed={duplicates}")
            raise ValueError("invalid Chai trunk mapping: " + "; ".join(details))
    return params


def _pairformer_stack(
    single: jnp.ndarray,
    pair: jnp.ndarray,
    token_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    blocks: tuple[PairformerBlockParams, ...],
    *,
    use_scan: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if len(blocks) != 48:
        raise ValueError("Chai trunk requires exactly 48 Pairformer blocks")
    if use_scan:
        stacked_blocks = stacked_or_stack(blocks)

        def body(carry, block):
            single_value, pair_value = carry
            return (
                pairformer_block(
                    single_value,
                    pair_value,
                    token_mask,
                    pair_mask,
                    block,
                ),
                None,
            )

        return jax.lax.scan(body, (single, pair), stacked_blocks)[0]
    for block in blocks:
        single, pair = pairformer_block(single, pair, token_mask, pair_mask, block)
    return single, pair


def trunk_forward(
    token_single_trunk_initial_repr: jnp.ndarray,
    token_pair_trunk_initial_repr: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    token_pair_trunk_repr: jnp.ndarray,
    msa_input_feats: jnp.ndarray,
    msa_mask: jnp.ndarray,
    template_input_feats: jnp.ndarray,
    template_input_masks: jnp.ndarray,
    token_single_mask: jnp.ndarray,
    token_pair_mask: jnp.ndarray,
    params: TrunkParams,
    *,
    outer_product_chunk_size: int = 4096,
    weighted_averaging_chunk_size: int = 8192,
    use_scan: bool = True,
    skip_templates: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the exported Chai trunk contract and return ``(single, pair)``.

    The input order and return order match TorchScript ``forward_256``.  Bond
    and restraint features are already part of the initial pair representation
    produced by ``token_embedder``; the trunk graph adds no separate restraint
    tensor.
    """
    pair = token_pair_trunk_initial_repr + recycling_projection(
        token_pair_trunk_repr, params.pair_recycling
    )
    single = token_single_trunk_initial_repr + recycling_projection(
        token_single_trunk_repr, params.single_recycling
    )
    if not skip_templates:
        pair = template_embedding(
            pair,
            template_input_feats,
            template_input_masks,
            token_pair_mask,
            params.template,
        )
    # The official exported graph retains the template-conditioned pair state
    # and adds it once more to the complete MSA module output (``z20`` in
    # ``forward_256``).  This is an outer trunk residual, distinct from the
    # residuals within the four MSA blocks.
    pair_before_msa = pair
    pair = pair_before_msa + msa_module(
        msa_input_feats,
        msa_mask,
        single,
        pair,
        token_pair_mask,
        params.msa,
        outer_product_chunk_size=outer_product_chunk_size,
        weighted_averaging_chunk_size=weighted_averaging_chunk_size,
    )
    return _pairformer_stack(
        single,
        pair,
        token_single_mask,
        token_pair_mask,
        params.pairformer_blocks,
        use_scan=use_scan,
    )
