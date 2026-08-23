"""MSA trunk blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn as jnn

from foldjax.models._cp import cp_mesh
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


# The outer product widens ``[M, N, C]`` into ``[N, N, C, C]`` before
# ``linear_out`` narrows it back to ``[N, N, C_z]``. At 1,003 tokens and the
# released widths that intermediate is 1,965 MiB, and it exists twice: the
# contraction emits ``[i, c, j, d]`` and the reshape that feeds ``linear_out``
# needs ``[i, j, c, d]``. Two buffers of 1,965 MiB, in a trunk whose whole pair
# representation is 246 MiB a copy.
#
# ``chunk_size`` already blocks this, but only when the chunk policy hands it
# one, and upstream's table asks for no chunking at or below 1,024 tokens -- so
# the sizes where this tensor is largest relative to everything else are
# precisely the sizes that build it whole. Every other widening operation here
# already carries a ceiling of its own: the transitions have
# ``_TRANSITION_WIDE_BUDGET_BYTES`` and the triangle projections have
# ``_PROJECTION_BUDGET_BYTES``. This is the same ceiling for the one that was
# missing it, and it does not touch the chunk table: a policy that asks for a
# smaller block still gets it, and ``chunk_size=0`` still means "build it
# whole".
#
# KNOWN BOUNDARY -- this ceiling is inert under a context-parallel mesh, which
# is the one place memory is scarcest, so it is worth saying what that costs.
# Measured on a forced four-device mesh at 1,003 tokens (compile only, sizes
# read off the partitioned shapes at bfloat16):
#
#   ceiling inert, as shipped     492 MiB/device widest buffer,  6 all-gathers
#   ceiling on, guard removed     246 MiB/device widest buffer, 12 all-gathers
#   `a` row-sharded first         492 MiB/device widest buffer,  6 all-gathers
#                                                              + 6 all-to-alls
#
# So the operands arriving replicated does not leave the widened tensor whole:
# the partitioner already splits it without help. Turning the ceiling on under a
# mesh would halve the widest buffer and double the collective traffic, which is
# a trade an interconnect has to settle, not a shape count. Sharding `a` before
# the contraction -- the obvious fix -- buys no memory and adds an all-to-all.
#
# It can only bite between 513 and 1,024 tokens anyway: below that the ceiling
# would not fire, and above it the chunk policy supplies a block size, so the
# mesh already takes the unrolled path below. Context parallelism is for jobs
# far larger than 1,024 tokens, so this band is not why anyone reaches for it.
_OPM_WIDE_BUDGET_BYTES = 512 * 1024**2


def _opm_block_rows(a: jnp.ndarray, requested: int | None) -> int | None:
    """Token rows whose widened outer product fits the budget, or the request."""
    # Both token axes of the outer product come from `a`, so one of its rows
    # costs `n_token * c_hidden ** 2` elements.
    n_token, hidden = a.shape[-2], a.shape[-1]
    per_row = n_token * hidden * hidden * a.dtype.itemsize
    if per_row <= 0 or n_token < 2 or per_row * n_token <= _OPM_WIDE_BUDGET_BYTES:
        return requested
    allowed = max(1, _OPM_WIDE_BUDGET_BYTES // per_row)
    return allowed if requested is None else min(requested, allowed)


def outer_product_mean(
    m: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: OuterProductMeanParams,
    *,
    eps: float = 1e-3,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply Protenix ``OuterProductMean`` in dense inference mode.

    ``chunk_size`` blocks the first token axis, which is what upstream does:
    ``MSABlock.forward`` hands ``outer_product_mean_msa`` the same dynamic chunk
    size it hands the pair stack. Without it the ``[N, N, C, C]`` outer product
    exists at full width before ``linear_out`` narrows it to ``C_z``, which is
    ``C ** 2 / C_z`` times the size of the result -- eight times over at the
    released widths. Projecting inside the block is the part that matters; a
    chunked einsum whose projection stays outside saves nothing.

    When nothing asks for a block size, one is chosen from
    ``_OPM_WIDE_BUDGET_BYTES``; ``chunk_size=0`` opts out and builds it whole.
    """

    if mask is None:
        mask = jnp.ones(m.shape[:-1], dtype=m.dtype)
    mask = mask.astype(m.dtype)

    m_norm = layer_norm(m, params.layer_norm)
    a = linear(m_norm, params.linear_1) * mask[..., None]
    b = linear(m_norm, params.linear_2) * mask[..., None]

    def project(rows: jnp.ndarray) -> jnp.ndarray:
        outer = jnp.einsum("...mic,...mjd->...ijcd", rows, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        return linear(outer, params.linear_out)

    # The token axis is a pure batch axis of the output -- the sum runs over the
    # MSA rows -- so the blocks are independent and reassembling them reproduces
    # the dense result, whichever way they are sequenced.
    n_token = a.shape[-2]
    cp_active = cp_mesh() is not None
    if not cp_active and (chunk_size is None or chunk_size > 0):
        chunk_size = _opm_block_rows(a, chunk_size)
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n_token:
        outer = project(a)
    elif cp_active:
        # A context-parallel mesh keeps the unrolled form it has always had: the
        # loop below writes its blocks at a traced offset, and the partitioner
        # cannot hold a row-sharded carry across that. Under a mesh the pair
        # tensor is split across devices anyway, which is most of what the block
        # was bounding.
        blocks = [
            project(
                jax.lax.dynamic_slice_in_dim(
                    a, start, min(chunk_size, n_token - start), axis=-2
                )
            )
            for start in range(0, n_token, chunk_size)
        ]
        outer = jnp.concatenate(blocks, axis=-3)
    else:
        # Sequenced by a loop rather than emitted as a Python list, and that is
        # the whole saving rather than a style choice. A list hands XLA one
        # independent subgraph per block and its scheduler is free to run every
        # widening contraction before any of the narrowing dots, which keeps
        # every block's widened tensor live at once and bounds nothing.
        #
        # The last block's start is clamped rather than the array padded, so a
        # token count that is not a multiple of the block size recomputes a few
        # rows and writes them again with the same values.
        block = min(chunk_size, n_token)

        def body(index, out):
            start = jnp.minimum(index * block, n_token - block)
            rows = jax.lax.dynamic_slice_in_dim(a, start, block, axis=-2)
            return jax.lax.dynamic_update_slice_in_dim(
                out, project(rows), start, axis=-3
            )

        outer = jax.lax.fori_loop(
            0,
            -(-n_token // block),
            body,
            jnp.zeros(
                a.shape[:-3] + (n_token, n_token, params.linear_out.weight.shape[0]),
                dtype=a.dtype,
            ),
        )

    norm = jnp.einsum("...mi,...mj->...ij", mask, mask)[..., None] + eps
    return outer / norm


def msa_pair_weighted_averaging(
    m: jnp.ndarray,
    z: jnp.ndarray,
    params: MSAPairWeightedAveragingParams,
    pair_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply inference-mode ``MSAPairWeightedAveraging``."""

    m_norm = layer_norm(m, params.layernorm_m)
    num_heads = int(params.linear_z.weight.shape[0])
    v = linear(m_norm, params.linear_mv)
    v = v.reshape(v.shape[:-1] + (num_heads, -1))
    b = linear(layer_norm(z, params.layernorm_z), params.linear_z)
    if pair_mask is not None:
        b = b + jnp.where(
            jnp.asarray(pair_mask).astype(bool)[..., None],
            jnp.asarray(0.0, dtype=b.dtype),
            jnp.asarray(-1.0e10, dtype=b.dtype),
        )
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
    msa_stack_first: bool = False,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    opm_chunk_size: int | None = None,
    triangle_attention_backend: str | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Apply one inference-mode MSA block of the Protenix family.

    ``msa_stack_first`` picks between the two orderings this family uses, and
    they are not interchangeable: whichever runs second reads the other's
    output. Protenix runs the communication first (Algorithm 8 lines 6-13), so
    the outer product mean sees the MSA representation the block was handed and
    the MSA stack sees the pair representation that mean just produced. OpenDDE
    reverses it, deliberately and with a comment saying so -- "Boltz updates MSA
    first, then writes the refreshed MSA state back to z" -- which is also what
    Boltz-2's own module does.

    Both shapes are identical, so nothing fails when this is wrong; the trunk
    simply converges to a different state, and since it recycles that state the
    error compounds. Getting it backwards cost Protenix an order of magnitude on
    `z` per cycle, and cost OpenDDE a pair representation that correlates 0.47
    with upstream's.
    """

    if m is None:
        raise ValueError("MSABlock requires m before the final block output")

    def msa_stack(m_in: jnp.ndarray, z_in: jnp.ndarray) -> jnp.ndarray:
        if params.msa_pair_weighted_averaging is None:
            return m_in
        if params.msa_transition is None:
            raise ValueError("missing MSA transition for non-final MSA block")
        m_out = m_in + msa_pair_weighted_averaging(
            m_in,
            z_in,
            params.msa_pair_weighted_averaging,
            pair_mask=pair_mask,
        )
        if msa_mask is not None:
            m_out = m_out * msa_mask.astype(m_out.dtype)[..., None]
        m_out = m_out + transition(m_out, params.msa_transition)
        if msa_mask is not None:
            m_out = m_out * msa_mask.astype(m_out.dtype)[..., None]
        return m_out

    if msa_stack_first:
        m = msa_stack(m, z)
        z = z + outer_product_mean(
            m, msa_mask, params.outer_product_mean, chunk_size=opm_chunk_size
        )
    else:
        z = z + outer_product_mean(
            m, msa_mask, params.outer_product_mean, chunk_size=opm_chunk_size
        )
        m = msa_stack(m, z)
    if pair_mask is not None:
        z = z * jnp.asarray(pair_mask, dtype=z.dtype)[..., None]
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
    opm_chunk_size: int | None = None,
    triangle_attention_backend: str | None = None,
    use_scan: bool = True,
    msa_stack_first: bool = False,
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
        msa_stack_first=msa_stack_first,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        opm_chunk_size=opm_chunk_size,
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
