"""MSA trunk blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn as jnn

from foldjax.models._stacking import stacked_or_stack
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    TransitionParams,
    layer_norm,
    linear,
    sigmoid,
    transition,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    PairformerBlockParams,
    pairformer_block,
)


class OuterProductMeanParams(NamedTuple):
    """Parameters for Protenix ``OuterProductMean``."""

    layer_norm: LayerNormParams
    linear_1: LinearParams
    linear_2: LinearParams
    linear_out: LinearParams


class MSAPairWeightedAveragingParams(NamedTuple):
    """Parameters for ``MSAPairWeightedAveraging``."""

    layernorm_m: LayerNormParams
    linear_mv: LinearParams
    layernorm_z: LayerNormParams
    linear_z: LinearParams
    linear_mg: LinearParams
    linear_out: LinearParams


class MSABlockParams(NamedTuple):
    """Parameters for one Protenix MSA block."""

    outer_product_mean: OuterProductMeanParams
    msa_pair_weighted_averaging: MSAPairWeightedAveragingParams | None
    msa_transition: TransitionParams | None
    pair_stack: PairformerBlockParams


class MSAModuleParams(NamedTuple):
    """Parameters for Protenix ``MSAModule``."""

    linear_m: LinearParams
    linear_s: LinearParams
    blocks: tuple[MSABlockParams, ...]


def pad_msa_features_to_bucket(
    input_feature_dict: dict,
    *,
    bucket_size: int = 64,
    max_padding_rows: int | None = None,
) -> dict:
    """Pad aligned MSA fields to a fixed row bucket and attach a real-row mask."""

    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    if max_padding_rows is not None and max_padding_rows < 0:
        raise ValueError("max_padding_rows must be non-negative")
    msa_fields = ("msa", "has_deletion", "deletion_value")
    if any(field not in input_feature_dict for field in msa_fields):
        return input_feature_dict
    n_msa = int(input_feature_dict["msa"].shape[-2])
    padded_depth = ((n_msa + bucket_size - 1) // bucket_size) * bucket_size
    padding_rows = padded_depth - n_msa
    if padding_rows == 0 or (
        max_padding_rows is not None and padding_rows > max_padding_rows
    ):
        return input_feature_dict
    output = dict(input_feature_dict)
    for field in msa_fields:
        values = np.asarray(input_feature_dict[field])
        pad_width = [(0, 0)] * values.ndim
        pad_width[-2] = (0, padded_depth - n_msa)
        output[field] = np.pad(values, pad_width, mode="constant")
    mask = np.zeros(output["msa"].shape, dtype=np.float32)
    real_slice = [slice(None)] * mask.ndim
    real_slice[-2] = slice(0, n_msa)
    mask[tuple(real_slice)] = 1.0
    output["msa_mask"] = mask
    return output


def sample_msa_cycle_features(
    input_feature_dict: dict,
    *,
    n_cycle: int,
    seed: int,
    bucket_size: int = 64,
) -> tuple[dict[str, np.ndarray], ...]:
    """Build deterministic, fixed-shape MSA subsets using upstream's policy."""

    if n_cycle <= 0:
        raise ValueError("n_cycle must be positive")
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    msa_fields = ("msa", "has_deletion", "deletion_value")
    if any(field not in input_feature_dict for field in msa_fields):
        return tuple()
    n_msa = int(input_feature_dict["msa"].shape[-2])
    if n_msa <= 0:
        return tuple()
    rng = np.random.default_rng(seed)
    sampled_cycles = []
    for _ in range(n_cycle):
        sample_size = int(rng.integers(1, n_msa + 1))
        indices = rng.permutation(n_msa)[:sample_size]
        sampled_cycles.append(
            {
                field: np.take(np.asarray(input_feature_dict[field]), indices, axis=-2)
                for field in msa_fields
            }
        )
    max_depth = max(int(cycle["msa"].shape[-2]) for cycle in sampled_cycles)
    padded_depth = min(
        n_msa, ((max_depth + bucket_size - 1) // bucket_size) * bucket_size
    )
    cycles = []
    for cycle in sampled_cycles:
        real_depth = int(cycle["msa"].shape[-2])
        pad_width = [(0, 0)] * cycle["msa"].ndim
        pad_width[-2] = (0, padded_depth - real_depth)
        padded = {
            field: np.pad(values, pad_width, mode="constant")
            for field, values in cycle.items()
        }
        mask = np.zeros(padded["msa"].shape, dtype=np.float32)
        real_slice = [slice(None)] * mask.ndim
        real_slice[-2] = slice(0, real_depth)
        mask[tuple(real_slice)] = 1.0
        padded["msa_mask"] = mask
        cycles.append(padded)
    return tuple(cycles)


def outer_product_mean(
    m: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: OuterProductMeanParams,
    *,
    eps: float = 1e-3,
) -> jnp.ndarray:
    """Apply Protenix ``OuterProductMean`` in dense inference mode."""

    if mask is None:
        mask = jnp.ones(m.shape[:-1], dtype=m.dtype)
    mask = mask.astype(m.dtype)

    m_norm = layer_norm(m, params.layer_norm)
    a = linear(m_norm, params.linear_1) * mask[..., None]
    b = linear(m_norm, params.linear_2) * mask[..., None]
    outer = jnp.einsum("...mic,...mjd->...ijcd", a, b)
    outer = outer.reshape(outer.shape[:-2] + (-1,))
    outer = linear(outer, params.linear_out)
    norm = jnp.einsum("...mi,...mj->...ij", mask, mask)[..., None] + eps
    return outer / norm


def msa_pair_weighted_averaging(
    m: jnp.ndarray,
    z: jnp.ndarray,
    params: MSAPairWeightedAveragingParams,
) -> jnp.ndarray:
    """Apply inference-mode ``MSAPairWeightedAveraging``."""

    m_norm = layer_norm(m, params.layernorm_m)
    num_heads = int(params.linear_z.weight.shape[0])
    v = linear(m_norm, params.linear_mv)
    v = v.reshape(v.shape[:-1] + (num_heads, -1))
    b = linear(layer_norm(z, params.layernorm_z), params.linear_z)
    weights = jnn.softmax(b.astype(jnp.float32), axis=-2).astype(v.dtype)
    gate = sigmoid(linear(m_norm, params.linear_mg))
    gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
    out = gate * jnp.einsum("...ijh,...mjhc->...mihc", weights, v)
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.linear_out)


def msa_block(
    m: jnp.ndarray | None,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    params: MSABlockParams,
    *,
    msa_mask: jnp.ndarray | None = None,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    triangle_attention_backend: str | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Apply one inference-mode Protenix MSA block."""

    if m is None:
        raise ValueError("MSABlock requires m before the final block output")
    if params.msa_pair_weighted_averaging is not None:
        m = m + msa_pair_weighted_averaging(
            m,
            z,
            params.msa_pair_weighted_averaging,
        )
        if params.msa_transition is None:
            raise ValueError("missing MSA transition for non-final MSA block")
        m = m + transition(m, params.msa_transition)
    z = z + outer_product_mean(m, msa_mask, params.outer_product_mean)
    _, z = pairformer_block(
        None,
        z,
        pair_mask,
        params.pair_stack,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        triangle_attention_backend=triangle_attention_backend,
    )
    if params.msa_pair_weighted_averaging is None:
        return None, z
    return m, z


def msa_module(
    input_feature_dict: dict[str, jnp.ndarray],
    z: jnp.ndarray,
    s_inputs: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    params: MSAModuleParams,
    *,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    triangle_attention_backend: str | None = None,
    use_scan: bool = True,
) -> jnp.ndarray:
    """Apply Protenix ``MSAModule`` to already-materialized MSA features.

    ``use_scan`` runs the blocks as one ``lax.scan`` over stacked parameters rather
    than emitting one copy of the block body per block, matching what
    ``pairformer_stack`` and ``diffusion_transformer_stack`` already do here. Same
    arithmetic; it trades a parameter copy for a much smaller HLO module.
    """

    if not params.blocks or "msa" not in input_feature_dict:
        return z
    msa = input_feature_dict["msa"]
    if msa.ndim < 2:
        return z

    msa_one_hot = jnp.eye(32, dtype=s_inputs.dtype)[msa]
    target_shape = msa_one_hot.shape[:-1]
    msa_sample = jnp.concatenate(
        [
            msa_one_hot,
            input_feature_dict["has_deletion"].reshape(target_shape + (1,)),
            input_feature_dict["deletion_value"].reshape(target_shape + (1,)),
        ],
        axis=-1,
    )
    m = linear(msa_sample, params.linear_m)
    m = m + linear(s_inputs, params.linear_s)[..., None, :, :]
    msa_mask = input_feature_dict.get("msa_mask")

    settings = dict(
        msa_mask=msa_mask,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        triangle_attention_backend=triangle_attention_backend,
    )
    # Protenix drops the MSA path from its *last* block, so the stack is uniform
    # only up to that point. Scanning the uniform prefix and looping the remainder
    # is exact and is what makes this fire at all: a whole-stack check would always
    # be false for the released 4-block configuration.
    prefix = _uniform_prefix(params.blocks) if use_scan else 0
    if prefix > 1:
        stacked = stack_msa_block_params(params.blocks[:prefix])

        def body(carry, block_params):
            m_c, z_c = carry
            return msa_block(m_c, z_c, pair_mask, block_params, **settings), None

        (m, z), _ = jax.lax.scan(body, (m, z), stacked)
        remaining = params.blocks[prefix:]
    else:
        remaining = params.blocks

    for block_params in remaining:
        m, z = msa_block(m, z, pair_mask, block_params, **settings)
    return z

def _uniform_prefix(blocks: tuple[MSABlockParams, ...]) -> int:
    """How many leading blocks share one parameter tree and leaf shapes.

    ``msa_pair_weighted_averaging`` and ``msa_transition`` are optional per block --
    Protenix omits them from the final block -- so only a prefix is stackable.
    Shapes are compared as well as structure, because ``jnp.stack`` on mismatched
    shapes raises from inside ``jax.tree.map`` naming neither the block nor the
    field.
    """
    if not blocks:
        return 0
    reference = jax.tree.structure(blocks[0])
    shapes = [jnp.shape(leaf) for leaf in jax.tree.leaves(blocks[0])]
    count = 1
    for block in blocks[1:]:
        if jax.tree.structure(block) != reference:
            break
        if [jnp.shape(leaf) for leaf in jax.tree.leaves(block)] != shapes:
            break
        count += 1
    return count


def stack_msa_block_params(
    blocks: tuple[MSABlockParams, ...],
) -> MSABlockParams:
    """Stack MSA block params on a leading layer axis for ``lax.scan``."""

    if not blocks:
        raise ValueError("stack_msa_block_params requires at least one block")
    return stacked_or_stack(blocks)
