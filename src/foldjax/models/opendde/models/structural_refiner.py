"""OpenDDE structural-token Pairformer refinement."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.attention import (
    attention_pair_bias,
)
from foldjax.models.protenix.models.primitives.primitives import transition
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    PairformerBlockParams,
    PairformerStackParams,
    pairformer_block,
    stack_pairformer_block_params,
)


def structural_refiner_block(
    s: jnp.ndarray,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    extra_attn_bias: jnp.ndarray | None,
    params: PairformerBlockParams,
    *,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    single_attention_backend: str = "xla",
    triangle_attention_backend: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply one OpenDDE structural refiner block.

    The pair update is the standard Protenix Pairformer operation. OpenDDE's
    structural branch additionally adds a scalar role-pair bias to every head
    of the single attention logits.
    """

    pair_only_params = params._replace(
        attention_pair_bias=None,
        single_transition=None,
    )
    _, z = pairformer_block(
        None,
        z,
        pair_mask,
        pair_only_params,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        triangle_attention_backend=triangle_attention_backend,
    )
    if params.attention_pair_bias is None or params.single_transition is None:
        raise ValueError("structural refiner requires the Pairformer single path")
    attention_params = params.attention_pair_bias._replace(
        has_s=False,
        cross_attention_mode=False,
    )
    num_heads = int(attention_params.linear_z.weight.shape[0])
    s = s + attention_pair_bias(
        s,
        None,
        z,
        attention_params,
        num_heads=num_heads,
        q_chunk_size=single_att_q_chunk_size,
        attention_backend=single_attention_backend,
        extra_attn_bias=extra_attn_bias,
    )
    return s + transition(s, params.single_transition), z


def structural_refiner_stack(
    s: jnp.ndarray,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    extra_attn_bias: jnp.ndarray | None,
    params: PairformerStackParams,
    *,
    use_scan: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    single_attention_backend: str = "xla",
    triangle_attention_backend: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the OpenDDE structural-token refiner stack."""

    if not params.blocks:
        raise ValueError("structural refiner requires at least one block")

    def apply_block(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        block_params: PairformerBlockParams,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return structural_refiner_block(
            *carry,
            pair_mask,
            extra_attn_bias,
            block_params,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
            single_attention_backend=single_attention_backend,
            triangle_attention_backend=triangle_attention_backend,
        )

    if not use_scan:
        carry = (s, z)
        for block_params in params.blocks:
            carry = apply_block(carry, block_params)
        return carry

    stacked = stack_pairformer_block_params(params.blocks)

    def body(carry, block_params):
        return apply_block(carry, block_params), None

    (s, z), _ = jax.lax.scan(body, (s, z), stacked)
    return s, z
