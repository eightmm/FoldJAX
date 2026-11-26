"""Transformer leaf blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._glu import gated_linear_unit, site_backend
from foldjax.models._stacking import stacked_or_stack
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    attention_pair_bias,
    local_attention_pair_bias,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LinearParams,
    _gate_kernel,
    adaptive_layer_norm,
    linear,
    reject_fused_glu_under_cp,
    sigmoid,
    silu,
)


class ConditionedTransitionParams(NamedTuple):
    """Parameters for ``ConditionedTransitionBlock``."""

    adaln: AdaptiveLayerNormParams
    linear_a1: LinearParams
    linear_a2: LinearParams
    linear_b: LinearParams
    linear_s: LinearParams


class DiffusionTransformerBlockParams(NamedTuple):
    """Parameters for one Protenix ``DiffusionTransformerBlock``."""

    attention_pair_bias: AttentionPairBiasParams
    conditioned_transition: ConditionedTransitionParams


class DiffusionTransformerStackParams(NamedTuple):
    """Parameters for a Protenix ``DiffusionTransformer`` stack."""

    blocks: tuple[DiffusionTransformerBlockParams, ...]


def conditioned_transition_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    params: ConditionedTransitionParams,
    *,
    glu_backend: str = "xla",
) -> jnp.ndarray:
    """Apply Protenix ``ConditionedTransitionBlock``.

    ``glu_backend="tokamax"`` runs the SwiGLU as one fused Triton kernel
    instead of two matmuls and a product, which never writes the two widened
    branches. Opt-in, GPU-only and a numerics change; ``"xla"``, the default,
    is the historical arithmetic unchanged. See :mod:`foldjax.models._glu`.
    """

    a = adaptive_layer_norm(a, s, params.adaln)
    # Not a plain transition: `pallas` leaves it on XLA.
    glu_backend = site_backend(glu_backend, released="xla", width=None)
    if glu_backend != "xla":
        reject_fused_glu_under_cp(glu_backend)
        hidden = gated_linear_unit(
            a,
            _gate_kernel(params.linear_a1, "linear_a1"),
            _gate_kernel(params.linear_a2, "linear_a2"),
            jax.nn.silu,
            backend=glu_backend,
        )
    else:
        hidden = silu(linear(a, params.linear_a1)) * linear(a, params.linear_a2)
    return sigmoid(linear(s, params.linear_s)) * linear(hidden, params.linear_b)


_compiled_conditioned_transition_block = jax.jit(
    conditioned_transition_block,
    static_argnames=("glu_backend",),
)


def build_diffusion_pair_bias_cache(
    transformer_z: jnp.ndarray,
    params: DiffusionTransformerStackParams,
    *,
    extra_attn_bias: jnp.ndarray | None = None,
    sequence_mask: jnp.ndarray | None = None,
    z_is_normalized: bool = True,
    use_scan: bool = False,
    mask_dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Precompute per-block additive token pair bias outside the sampler loop.

    Follows the exact operation order of the uncached global
    ``diffusion_transformer_block`` -> ``attention_pair_bias`` path when
    ``z_is_normalized=True``: apply ``layernorm_z`` weight then bias to
    ``transformer_z``, project through ``linear_z``, move the head axis from
    ``-1`` to ``-3``, and add ``extra_attn_bias`` plus the ``sequence_mask``
    bias.

    The returned tensor has shape ``[n_blocks, *transformer_z.shape[:-3], H, S, S]``
    (preserving any leading sample axis at ``-4`` on ``transformer_z``), so
    slicing axis 0 yields each block's additive bias ready for
    ``attention_pair_bias(..., precomputed_pair_bias=...)``.

    Memory cost: ``n_blocks * H * S**2 * itemsize`` bytes (for 24 blocks,
    16 heads, and ``S = 1,902`` in float32, about 5.2 GiB).
    """

    if not params.blocks:
        raise ValueError("DiffusionTransformerStack requires at least one block")
    if not z_is_normalized:
        raise ValueError(
            "build_diffusion_pair_bias_cache requires z_is_normalized=True "
            "(use_efficient_fusion=True)"
        )
    if transformer_z.ndim < 3:
        raise ValueError(
            "transformer_z must have at least 3 axes [..., S, S, c_z]; "
            f"got shape {tuple(transformer_z.shape)}"
        )
    seq_len = transformer_z.shape[-2]
    if transformer_z.shape[-3] != seq_len:
        raise ValueError(
            "transformer_z pair axes must be square [..., S, S, c_z]; "
            f"got shape {tuple(transformer_z.shape)}"
        )

    combined_extra = extra_attn_bias
    if sequence_mask is not None:
        sequence_mask = jnp.asarray(sequence_mask).astype(bool)
        if sequence_mask.ndim != 1 or sequence_mask.shape[0] != seq_len:
            raise ValueError("diffusion transformer sequence_mask must have shape [N]")
        mask_bias = jnp.where(
            sequence_mask[..., :, None] & sequence_mask[..., None, :],
            jnp.asarray(0.0, dtype=mask_dtype),
            jnp.asarray(-1.0e10, dtype=mask_dtype),
        )
        combined_extra = (
            mask_bias
            if combined_extra is None
            else jnp.asarray(combined_extra, dtype=mask_dtype) + mask_bias
        )
    if combined_extra is not None and combined_extra.shape[-2:] != (
        seq_len,
        seq_len,
    ):
        raise ValueError(
            "extra attention bias pair axes must match projected pair bias: "
            f"{combined_extra.shape[-2:]} != {(seq_len, seq_len)}"
        )

    def _block_bias(layernorm_z, linear_z) -> jnp.ndarray:
        z_norm = transformer_z
        if layernorm_z.weight is not None:
            z_norm = z_norm * layernorm_z.weight
        if layernorm_z.bias is not None:
            z_norm = z_norm + layernorm_z.bias
        bias = linear(z_norm, linear_z)
        bias = jnp.moveaxis(bias, -1, -3)
        if combined_extra is not None:
            extra = jnp.asarray(combined_extra, dtype=bias.dtype)
            while extra.ndim < bias.ndim - 1:
                extra = jnp.expand_dims(extra, axis=0)
            if extra.ndim == bias.ndim - 1:
                extra = jnp.expand_dims(extra, axis=-3)
            bias = bias + extra
        return bias

    if not use_scan:
        return jnp.stack(
            [
                _block_bias(
                    block.attention_pair_bias.layernorm_z,
                    block.attention_pair_bias.linear_z,
                )
                for block in params.blocks
            ],
            axis=0,
        )

    stacked = stack_diffusion_transformer_block_params(params.blocks)

    def _scan_body(carry, xs):
        layernorm_z, linear_z = xs
        return carry, _block_bias(layernorm_z, linear_z)

    _, biases = jax.lax.scan(
        _scan_body,
        None,
        (
            stacked.attention_pair_bias.layernorm_z,
            stacked.attention_pair_bias.linear_z,
        ),
    )
    return biases


def diffusion_transformer_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray | None,
    params: DiffusionTransformerBlockParams,
    *,
    num_heads: int,
    n_queries: int | None = None,
    n_keys: int | None = None,
    global_q_chunk_size: int | None = None,
    attention_backend: str = "xla",
    glu_backend: str = "xla",
    z_is_normalized: bool = False,
    extra_attn_bias: jnp.ndarray | None = None,
    sequence_mask: jnp.ndarray | None = None,
    precomputed_pair_bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply one inference-mode DiffusionTransformer block."""

    gate = None
    if sequence_mask is not None:
        sequence_mask = jnp.asarray(sequence_mask).astype(bool)
        if sequence_mask.ndim != 1 or sequence_mask.shape[0] != a.shape[-2]:
            raise ValueError("diffusion transformer sequence_mask must have shape [N]")
        gate = sequence_mask.astype(a.dtype)[..., None]
        a = a * gate

    if n_queries is not None and n_keys is not None:
        if precomputed_pair_bias is not None:
            raise ValueError(
                "precomputed_pair_bias is only supported by global token attention"
            )
        if extra_attn_bias is not None:
            raise ValueError(
                "extra attention bias is only supported by global token attention"
            )
        attn_out = local_attention_pair_bias(
            a,
            s,
            z,
            params.attention_pair_bias,
            num_heads=num_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            sequence_mask=sequence_mask,
            attention_backend=attention_backend,
        )
    elif precomputed_pair_bias is not None:
        if n_queries is not None or n_keys is not None:
            raise ValueError(
                "precomputed_pair_bias is only supported by global token attention"
            )
        if not z_is_normalized:
            raise ValueError(
                "precomputed_pair_bias requires z_is_normalized=True "
                "(use_efficient_fusion=True)"
            )
        if extra_attn_bias is not None:
            raise ValueError(
                "precomputed_pair_bias already carries additive bias; "
                "extra_attn_bias must be None"
            )
        attn_out = attention_pair_bias(
            a,
            s,
            z,
            params.attention_pair_bias,
            num_heads=num_heads,
            q_chunk_size=global_q_chunk_size,
            attention_backend=attention_backend,
            z_is_normalized=z_is_normalized,
            precomputed_pair_bias=precomputed_pair_bias,
        )
    else:
        if sequence_mask is not None:
            mask_bias = jnp.where(
                sequence_mask[..., :, None] & sequence_mask[..., None, :],
                jnp.asarray(0.0, dtype=a.dtype),
                jnp.asarray(-1.0e10, dtype=a.dtype),
            )
            extra_attn_bias = (
                mask_bias
                if extra_attn_bias is None
                else jnp.asarray(extra_attn_bias, dtype=a.dtype) + mask_bias
            )
        attn_out = attention_pair_bias(
            a,
            s,
            z,
            params.attention_pair_bias,
            num_heads=num_heads,
            q_chunk_size=global_q_chunk_size,
            attention_backend=attention_backend,
            z_is_normalized=z_is_normalized,
            extra_attn_bias=extra_attn_bias,
        )
    a = a + attn_out
    if gate is not None:
        a = a * gate
    transition_fn = (
        _compiled_conditioned_transition_block
        if attention_backend == "xla_jit"
        else conditioned_transition_block
    )
    a = a + transition_fn(a, s, params.conditioned_transition, glu_backend=glu_backend)
    if gate is not None:
        a = a * gate
    return a


def diffusion_transformer_stack(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray | None,
    params: DiffusionTransformerStackParams,
    *,
    num_heads: int,
    n_queries: int | None = None,
    n_keys: int | None = None,
    use_scan: bool = False,
    global_q_chunk_size: int | None = None,
    attention_backend: str = "xla",
    glu_backend: str = "xla",
    z_is_normalized: bool = False,
    extra_attn_bias: jnp.ndarray | None = None,
    sequence_mask: jnp.ndarray | None = None,
    precomputed_pair_bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply a DiffusionTransformer stack in inference mode."""

    if not params.blocks:
        raise ValueError("DiffusionTransformerStack requires at least one block")
    if precomputed_pair_bias is not None:
        if n_queries is not None or n_keys is not None:
            raise ValueError(
                "precomputed_pair_bias is only supported by global token attention"
            )
        if not z_is_normalized:
            raise ValueError(
                "precomputed_pair_bias requires z_is_normalized=True "
                "(use_efficient_fusion=True)"
            )
        if extra_attn_bias is not None:
            raise ValueError(
                "precomputed_pair_bias already carries additive bias; "
                "extra_attn_bias must be None"
            )
        if precomputed_pair_bias.ndim < 4 or precomputed_pair_bias.shape[0] != len(
            params.blocks
        ):
            raise ValueError(
                "precomputed_pair_bias must have shape [n_blocks, ..., H, S, S] "
                f"with leading axis {len(params.blocks)}; "
                f"got {tuple(precomputed_pair_bias.shape)}"
            )
    if not use_scan:
        if precomputed_pair_bias is not None:
            for block_index, block in enumerate(params.blocks):
                a = diffusion_transformer_block(
                    a,
                    s,
                    z,
                    block,
                    num_heads=num_heads,
                    global_q_chunk_size=global_q_chunk_size,
                    attention_backend=attention_backend,
                    glu_backend=glu_backend,
                    z_is_normalized=z_is_normalized,
                    sequence_mask=sequence_mask,
                    precomputed_pair_bias=precomputed_pair_bias[block_index],
                )
            return a
        for block in params.blocks:
            a = diffusion_transformer_block(
                a,
                s,
                z,
                block,
                num_heads=num_heads,
                n_queries=n_queries,
                n_keys=n_keys,
                global_q_chunk_size=global_q_chunk_size,
                attention_backend=attention_backend,
                glu_backend=glu_backend,
                z_is_normalized=z_is_normalized,
                extra_attn_bias=extra_attn_bias,
                sequence_mask=sequence_mask,
            )
        return a

    has_s = params.blocks[0].attention_pair_bias.has_s
    cross_attention_mode = params.blocks[0].attention_pair_bias.cross_attention_mode
    for block in params.blocks[1:]:
        block_attention = block.attention_pair_bias
        if (
            block_attention.has_s != has_s
            or block_attention.cross_attention_mode != cross_attention_mode
        ):
            raise ValueError(
                "scan requires homogeneous attention pair-bias configuration"
            )
    stacked = stack_diffusion_transformer_block_params(params.blocks)

    if precomputed_pair_bias is not None:

        def body_cached(a_carry, xs):
            block_params, block_pair_bias = xs
            attention_params = block_params.attention_pair_bias._replace(
                has_s=has_s,
                cross_attention_mode=cross_attention_mode,
            )
            block_params = block_params._replace(attention_pair_bias=attention_params)
            return (
                diffusion_transformer_block(
                    a_carry,
                    s,
                    z,
                    block_params,
                    num_heads=num_heads,
                    global_q_chunk_size=global_q_chunk_size,
                    attention_backend=attention_backend,
                    glu_backend=glu_backend,
                    z_is_normalized=z_is_normalized,
                    sequence_mask=sequence_mask,
                    precomputed_pair_bias=block_pair_bias,
                ),
                None,
            )

        a, _ = jax.lax.scan(body_cached, a, (stacked, precomputed_pair_bias))
        return a

    def body(a_carry, block_params):
        attention_params = block_params.attention_pair_bias._replace(
            has_s=has_s,
            cross_attention_mode=cross_attention_mode,
        )
        block_params = block_params._replace(attention_pair_bias=attention_params)
        return (
            diffusion_transformer_block(
                a_carry,
                s,
                z,
                block_params,
                num_heads=num_heads,
                n_queries=n_queries,
                n_keys=n_keys,
                global_q_chunk_size=global_q_chunk_size,
                attention_backend=attention_backend,
                glu_backend=glu_backend,
                z_is_normalized=z_is_normalized,
                extra_attn_bias=extra_attn_bias,
                sequence_mask=sequence_mask,
            ),
            None,
        )

    a, _ = jax.lax.scan(body, a, stacked)
    return a


def stack_diffusion_transformer_block_params(
    blocks: tuple[DiffusionTransformerBlockParams, ...],
) -> DiffusionTransformerBlockParams:
    """Stack DiffusionTransformer block params for ``lax.scan``."""

    if not blocks:
        raise ValueError("stack_diffusion_transformer_block_params requires blocks")
    return stacked_or_stack(blocks)
