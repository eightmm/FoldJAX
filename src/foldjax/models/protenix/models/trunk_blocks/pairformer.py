"""Pairformer block composition for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._cp import shard_pair_rows
from foldjax.models._stacking import stacked_or_stack
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    attention_pair_bias,
)
from foldjax.models.protenix.models.primitives.primitives import (
    TransitionParams,
    compiled_transition,
    transition,
)
from foldjax.models.protenix.models.triangle.triangle import (
    TriangleAttentionParams,
    TriangleMultiplicationParams,
    triangle_attention,
    triangle_multiplication,
)


class PairformerBlockParams(NamedTuple):
    """Parameters for one Protenix ``PairformerBlock``."""

    tri_mul_out: TriangleMultiplicationParams
    tri_mul_in: TriangleMultiplicationParams
    tri_att_start: TriangleAttentionParams
    tri_att_end: TriangleAttentionParams
    pair_transition: TransitionParams
    attention_pair_bias: AttentionPairBiasParams | None = None
    single_transition: TransitionParams | None = None


class PairformerStackParams(NamedTuple):
    """Parameters for a homogeneous Protenix ``PairformerStack``."""

    blocks: tuple[PairformerBlockParams, ...]


def pairformer_block(
    s: jnp.ndarray | None,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    params: PairformerBlockParams,
    *,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    single_attention_backend: str = "xla",
    triangle_attention_backend: str | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Apply one inference-mode Protenix Pairformer block.

    Dropout is omitted because this port targets inference/eval only.
    """

    # Under context parallelism the pair representation is sharded along its
    # rows; pinning it here (and after the transpose below) keeps every block
    # of every stack -- trunk, MSA, template, refiner, confidence -- on the
    # same layout without the partitioner re-deriving it per consumer.
    z = shard_pair_rows(z)
    pair_gate = None
    single_gate = None
    single_attention_bias = None
    if pair_mask is not None:
        pair_valid = jnp.asarray(pair_mask).astype(bool)
        pair_gate = pair_valid.astype(z.dtype)[..., None]
        z = z * pair_gate
        single_valid = jnp.diagonal(pair_valid, axis1=-2, axis2=-1)
        single_gate = single_valid.astype(z.dtype)[..., None]
        # Mask keys in the single path and gate queries below. Merely masking
        # triangle updates is insufficient: otherwise padded keys still enter
        # the single-attention softmax denominator and move every real token.
        single_attention_bias = jnp.where(
            single_valid[..., :, None] & single_valid[..., None, :],
            jnp.asarray(0.0, dtype=z.dtype),
            jnp.asarray(-1.0e10, dtype=z.dtype),
        )
        if s is not None:
            s = s * single_gate.astype(s.dtype)

    z = z + triangle_multiplication(
        z,
        pair_mask,
        params.tri_mul_out,
        "outgoing",
        chunk_size=triangle_mul_chunk_size,
        use_jit=(triangle_attention_backend or "").endswith("_jit"),
    )
    if pair_gate is not None:
        z = z * pair_gate
    z = z + triangle_multiplication(
        z,
        pair_mask,
        params.tri_mul_in,
        "incoming",
        chunk_size=triangle_mul_chunk_size,
        use_jit=(triangle_attention_backend or "").endswith("_jit"),
    )
    if pair_gate is not None:
        z = z * pair_gate

    tri_heads = int(params.tri_att_start.linear.weight.shape[0])
    z = z + triangle_attention(
        z,
        pair_mask,
        params.tri_att_start,
        num_heads=tri_heads,
        q_chunk_size=triangle_att_q_chunk_size,
        attention_backend=triangle_attention_backend,
    )
    if pair_gate is not None:
        z = z * pair_gate
    z_t = jnp.swapaxes(z, -2, -3)
    # The Fold-CP transpose exchange: ending-node attention treats columns as
    # rows, so the sharded axis moves with the transpose (an all-to-all under
    # the partitioner, an identity otherwise).
    z_t = shard_pair_rows(z_t)
    pair_mask_t = None if pair_mask is None else jnp.swapaxes(pair_mask, -1, -2)
    z_t = z_t + triangle_attention(
        z_t,
        pair_mask_t,
        params.tri_att_end,
        num_heads=tri_heads,
        q_chunk_size=triangle_att_q_chunk_size,
        attention_backend=triangle_attention_backend,
    )
    z = jnp.swapaxes(z_t, -2, -3)
    z = shard_pair_rows(z)
    if pair_gate is not None:
        z = z * pair_gate

    pair_transition_fn = (
        compiled_transition
        if (triangle_attention_backend or "").endswith("_jit")
        else transition
    )
    z = z + pair_transition_fn(z, params.pair_transition)
    if pair_gate is not None:
        z = z * pair_gate

    if params.attention_pair_bias is not None:
        if s is None:
            raise ValueError("PairformerBlock single path requires s")
        attention_pair_bias_params = params.attention_pair_bias._replace(
            has_s=False,
            cross_attention_mode=False,
        )
        pair_heads = int(attention_pair_bias_params.linear_z.weight.shape[0])
        s = s + attention_pair_bias(
            s,
            None,
            z,
            attention_pair_bias_params,
            num_heads=pair_heads,
            q_chunk_size=single_att_q_chunk_size,
            attention_backend=single_attention_backend,
            extra_attn_bias=single_attention_bias,
        )
        if single_gate is not None:
            s = s * single_gate.astype(s.dtype)
        if params.single_transition is None:
            raise ValueError("missing single_transition for single path")
        single_transition_fn = (
            compiled_transition if single_attention_backend == "xla_jit" else transition
        )
        s = s + single_transition_fn(s, params.single_transition)
        if single_gate is not None:
            s = s * single_gate.astype(s.dtype)

    return s, z


def pairformer_stack(
    s: jnp.ndarray | None,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    params: PairformerStackParams,
    *,
    use_scan: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    single_attention_backend: str = "xla",
    triangle_attention_backend: str | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Apply a Protenix PairformerStack in inference mode."""

    if not params.blocks:
        raise ValueError("PairformerStack requires at least one block")

    if not use_scan:
        for block_params in params.blocks:
            s, z = pairformer_block(
                s,
                z,
                pair_mask,
                block_params,
                triangle_mul_chunk_size=triangle_mul_chunk_size,
                triangle_att_q_chunk_size=triangle_att_q_chunk_size,
                single_att_q_chunk_size=single_att_q_chunk_size,
                single_attention_backend=single_attention_backend,
                triangle_attention_backend=triangle_attention_backend,
            )
        return s, z

    stacked = stack_pairformer_block_params(params.blocks)

    def body(carry, block_params):
        s_c, z_c = carry
        s_c, z_c = pairformer_block(
            s_c,
            z_c,
            pair_mask,
            block_params,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
            single_attention_backend=single_attention_backend,
            triangle_attention_backend=triangle_attention_backend,
        )
        return (s_c, z_c), None

    (s, z), _ = jax.lax.scan(body, (s, z), stacked)
    return s, z


def stack_pairformer_block_params(
    blocks: tuple[PairformerBlockParams, ...],
) -> PairformerBlockParams:
    """Stack block params on a leading layer axis for ``lax.scan``."""

    if not blocks:
        raise ValueError("stack_pairformer_block_params requires at least one block")
    return stacked_or_stack(blocks)
