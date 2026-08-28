"""Top-level trunk initialization helpers for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._cp import shard_pair_rows
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    ConstraintEmbedderParams,
    RelativePositionParams,
    constraint_embedder,
    relative_position_encoding,
    resolve_relative_position_features,
)
from foldjax.models.protenix.models.trunk_blocks.msa import (
    MSACycleIndexTape,
    MSAModuleParams,
    msa_module,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    PairformerStackParams,
    pairformer_stack,
)
from foldjax.models.protenix.models.trunk_blocks.template import (
    TemplateEmbedderParams,
    template_embedder,
)


class TrunkInitializationParams(NamedTuple):
    """Parameters for Protenix ``s_init`` and ``z_init`` construction."""

    linear_sinit: LinearParams
    linear_zinit1: LinearParams
    linear_zinit2: LinearParams
    relative_position: RelativePositionParams
    linear_token_bond: LinearParams


class RecyclingProjectionParams(NamedTuple):
    """Parameters for the root-level recycling projections."""

    layernorm_z: LayerNormParams
    linear_z: LinearParams
    layernorm_s: LayerNormParams
    linear_s: LinearParams


class TrunkParams(NamedTuple):
    """Root-level Protenix trunk parameters before the Pairformer stack."""

    initial: TrunkInitializationParams
    recycling: RecyclingProjectionParams


class PairformerOutputParams(NamedTuple):
    """Parameters for Protenix ``get_pairformer_output`` after input embedding."""

    trunk: TrunkParams
    constraint: ConstraintEmbedderParams
    template: TemplateEmbedderParams
    msa: MSAModuleParams
    pairformer_stack: PairformerStackParams


def trunk_initial_embeddings(
    s_inputs: jnp.ndarray,
    relp: jnp.ndarray,
    token_bonds: jnp.ndarray,
    params: TrunkInitializationParams,
    *,
    z_constraint: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build ``s_init`` and ``z_init`` as in Protenix ``get_pairformer_output``."""

    s_init = linear(s_inputs, params.linear_sinit)
    z_init = (
        linear(s_init, params.linear_zinit1)[..., :, None, :]
        + linear(
            s_init,
            params.linear_zinit2,
        )[..., None, :, :]
    )
    z_init = z_init + relative_position_encoding(relp, params.relative_position)
    z_init = z_init + linear(token_bonds[..., None], params.linear_token_bond)
    if z_constraint is not None:
        z_init = z_init + z_constraint
    return s_init, z_init


def recycle_embeddings(
    s_init: jnp.ndarray,
    z_init: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: RecyclingProjectionParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Protenix root-level recycling projections without dropout."""

    z_out = z_init + linear(layer_norm(z, params.layernorm_z), params.linear_z)
    s_out = s_init + linear(layer_norm(s, params.layernorm_s), params.linear_s)
    return s_out, z_out


def _parameter_dtype(params) -> jnp.dtype | None:
    """Return the floating dtype these parameters were cast to, if any."""
    for leaf in jax.tree.leaves(params):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            return dtype
    return None


def pairformer_output_from_s_inputs(
    input_feature_dict: dict[str, jnp.ndarray],
    s_inputs: jnp.ndarray,
    params: PairformerOutputParams,
    *,
    num_recycles: int = 1,
    pair_mask: jnp.ndarray | None = None,
    use_pairformer_scan: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    opm_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    single_attention_backend: str = "xla",
    triangle_attention_backend: str | None = None,
    cycle_msa_features: tuple[dict[str, jnp.ndarray], ...] | None = None,
    cycle_msa_index_tape: MSACycleIndexTape | None = None,
    use_cycle_scan: bool = True,
    msa_stack_first: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply Protenix trunk from precomputed ``InputFeatureEmbedder`` output.

    ``use_cycle_scan`` runs recycling as one ``lax.scan`` instead of emitting the
    body per cycle. Same arithmetic; a tenth of the graph at the released depth.
    """

    # Everything the trunk builds keys off `s_inputs`: the MSA one-hot is
    # allocated with its dtype, and that one array decides the dtype of the MSA
    # representation, and through it of the pair carry inside the MSA module's
    # own scan. The embedder that produced it reads integer features, so it
    # comes back float32 whatever the weights were cast to.
    trunk_dtype = _parameter_dtype(params.trunk) or s_inputs.dtype
    s_inputs = s_inputs.astype(trunk_dtype)

    z_constraint = None
    if "constraint_feature" in input_feature_dict:
        z_constraint = constraint_embedder(
            input_feature_dict["constraint_feature"],
            params.constraint,
        )

    relp = resolve_relative_position_features(input_feature_dict)
    # [N, N, 139] and consumed by a linear projection into `z_init`; born
    # sharded, the projection is shard-local instead of transiently whole.
    relp = shard_pair_rows(relp)
    s_init, z_init = trunk_initial_embeddings(
        s_inputs,
        relp,
        input_feature_dict["token_bonds"],
        params.trunk.initial,
        z_constraint=z_constraint,
    )
    # The trunk state follows the dtype the trunk's own weights were cast to.
    # `relp` and the token-bond features are built from integer inputs, so they
    # land in float32 no matter what `--trunk-dtype` asked for, and a single
    # float32 operand promotes `z` and everything downstream of it. Without
    # this, bfloat16 narrowed the weights and left every activation -- the pair
    # representation included, which is the largest buffer in the graph -- in
    # float32. The embedding output cannot be the reference here for exactly
    # the same reason: it is float32 too.
    #
    # AlphaFold 3 does the same thing, casting the recycled state at the trunk
    # boundary (`prev['pair'].astype(pair_activations.dtype)`) rather than
    # trying to keep every internal op from widening.
    s_init = s_init.astype(trunk_dtype)
    # Shard the pair state from its first materialization: `z_init` stays live
    # across every recycle as the recycling residual, so its layout decides
    # whether the carry ever exists whole on one device.
    z_init = shard_pair_rows(z_init.astype(trunk_dtype))

    s = jnp.zeros_like(s_init)
    z = jnp.zeros_like(z_init)
    if cycle_msa_features is not None and cycle_msa_index_tape is not None:
        raise ValueError(
            "cycle_msa_features and cycle_msa_index_tape are mutually exclusive"
        )
    if cycle_msa_features is not None and len(cycle_msa_features) != num_recycles:
        raise ValueError("cycle_msa_features length must equal num_recycles")
    if cycle_msa_index_tape is not None:
        if cycle_msa_index_tape.row_indices.ndim != 2:
            raise ValueError("cycle MSA row indices must have shape [cycle, row]")
        if (
            cycle_msa_index_tape.row_mask.shape
            != cycle_msa_index_tape.row_indices.shape
        ):
            raise ValueError("cycle MSA row mask must match the row-index shape")
        if cycle_msa_index_tape.row_indices.shape[0] != num_recycles:
            raise ValueError("cycle MSA index tape length must equal num_recycles")

    def one_cycle(carry, msa_features):
        s, z = carry
        # ``lax.scan`` with ``xs=None`` hands the body ``None``, which is the right
        # signal that the alignment is not resampled per cycle -- every cycle then
        # reads the same MSA features out of the feature dict. Forwarding the
        # ``None`` instead reaches ``msa_module`` as a missing feature dict.
        if cycle_msa_index_tape is not None:
            msa_features = _materialize_msa_cycle_from_index_tape(
                input_feature_dict, msa_features
            )
        elif msa_features is None:
            msa_features = input_feature_dict
        # Per-cycle alignments are sampled from the raw features, not from the
        # trunk's own (already narrowed) feature dict, so they arrive float32
        # even when the trunk is bfloat16. The MSA module scans over its blocks
        # with the pair representation in the carry, so one float32 row here
        # widens `z` inside that scan and the carry no longer closes.
        msa_features = jax.tree.map(
            lambda value: (
                value.astype(trunk_dtype)
                if hasattr(value, "dtype")
                and jnp.issubdtype(value.dtype, jnp.floating)
                else value
            ),
            msa_features,
        )
        s, z = recycle_embeddings(
            s_init,
            z_init,
            s,
            z,
            params.trunk.recycling,
        )
        z = z + template_embedder(
            input_feature_dict,
            z,
            pair_mask,
            params.template,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            triangle_attention_backend=triangle_attention_backend,
        )
        z = msa_module(
            msa_features,
            z,
            s_inputs,
            pair_mask,
            params.msa,
            msa_stack_first=msa_stack_first,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            opm_chunk_size=opm_chunk_size,
            triangle_attention_backend=triangle_attention_backend,
        )
        if params.pairformer_stack.blocks:
            s, z = pairformer_stack(
                s,
                z,
                pair_mask,
                params.pairformer_stack,
                use_scan=use_pairformer_scan,
                triangle_mul_chunk_size=triangle_mul_chunk_size,
                triangle_att_q_chunk_size=triangle_att_q_chunk_size,
                single_att_q_chunk_size=single_att_q_chunk_size,
                single_attention_backend=single_attention_backend,
                triangle_attention_backend=triangle_attention_backend,
            )
        # The template and MSA embedders read integer features and build their
        # own float32 intermediates, so the state can come back wider than it
        # went in. Under `lax.scan` that is not merely wasteful, it is a type
        # error -- the carry has to close. Cast at the boundary, as AlphaFold 3
        # does, rather than chasing every internal widening.
        return (s.astype(trunk_dtype), shard_pair_rows(z.astype(trunk_dtype))), None

    # Recycling is the same computation every cycle, differing only in the carry and
    # -- when the alignment is resampled per cycle -- in the MSA features. Both are
    # scannable, so the body is emitted once instead of ``num_recycles`` times. That
    # matters at Protenix's released depth: ten cycles over a 48-block Pairformer
    # unroll into a graph ten times larger, and compile time tracks graph size.
    stacked_msa = _stacked_cycle_msa(cycle_msa_features) if use_cycle_scan else None
    if use_cycle_scan and cycle_msa_index_tape is not None and num_recycles > 1:
        (s, z), _ = jax.lax.scan(one_cycle, (s, z), cycle_msa_index_tape)
    elif use_cycle_scan and cycle_msa_features is None and num_recycles > 1:
        (s, z), _ = jax.lax.scan(
            one_cycle, (s, z), xs=None, length=num_recycles
        )
    elif stacked_msa is not None and num_recycles > 1:
        (s, z), _ = jax.lax.scan(one_cycle, (s, z), stacked_msa)
    else:
        for cycle_index in range(num_recycles):
            if cycle_msa_index_tape is not None:
                msa_features = jax.tree.map(
                    lambda value: value[cycle_index],
                    cycle_msa_index_tape,
                )
            else:
                msa_features = (
                    input_feature_dict
                    if cycle_msa_features is None
                    else cycle_msa_features[cycle_index]
                )
            (s, z), _ = one_cycle((s, z), msa_features)
    return s_inputs, s, z


def _materialize_msa_cycle_from_index_tape(
    input_feature_dict: dict[str, jnp.ndarray],
    tape: MSACycleIndexTape,
) -> dict[str, jnp.ndarray]:
    """Gather one recycled MSA view from the single retained raw alignment."""

    required = ("msa", "has_deletion", "deletion_value")
    missing = [name for name in required if name not in input_feature_dict]
    if missing:
        raise ValueError(
            "cycle MSA index tape requires raw features: " + ", ".join(missing)
        )
    row_indices = jnp.asarray(tape.row_indices)
    row_mask = jnp.asarray(tape.row_mask, dtype=bool)
    if row_indices.ndim != 1 or row_mask.shape != row_indices.shape:
        raise ValueError("one cycle MSA index tape must have shape [row]")

    output = {}
    for name in required:
        selected = jnp.take(input_feature_dict[name], row_indices, axis=-2)
        mask_shape = (1,) * (selected.ndim - 2) + (row_mask.shape[0], 1)
        output[name] = jnp.where(
            row_mask.reshape(mask_shape),
            selected,
            jnp.zeros((), dtype=selected.dtype),
        )
    mask_shape = (1,) * (output["msa"].ndim - 2) + (row_mask.shape[0], 1)
    output["msa_mask"] = jnp.broadcast_to(
        row_mask.reshape(mask_shape), output["msa"].shape
    ).astype(jnp.float32)
    return output


def _stacked_cycle_msa(
    cycle_msa_features: tuple[dict[str, jnp.ndarray], ...] | None,
) -> dict[str, jnp.ndarray] | None:
    """Stack per-cycle MSA features on a leading cycle axis, or return ``None``.

    ``sample_msa_cycle_features`` pads every cycle to one row bucket precisely so
    this is possible. ``None`` is returned rather than raising when the shapes do
    not line up, because a caller is free to build the cycles itself; the loop then
    stays unrolled, which is slower to compile but correct.

    The keys are walked in sorted order, and that is load-bearing rather than
    tidy. Iterating a ``set`` of feature names emits these ``stack`` equations in
    an order that depends on the strings' hashes, which CPython randomizes per
    process. The arrays are identical either way, but the traced HLO is not: the
    same prediction lowers to a differently ordered module in every process, so
    its persistent-compile-cache key never repeats and every run recompiles the
    whole program from scratch. Measured on OpenDDE at 132 tokens, that was a
    warm run costing 32-57 s of recompilation against 3.4 s to replay the cached
    executable, while the recycling it was blamed on is 0.24 s per cycle.
    """
    if not cycle_msa_features:
        return None
    keys = sorted(cycle_msa_features[0])
    if any(sorted(cycle) != keys for cycle in cycle_msa_features[1:]):
        return None
    for key in keys:
        shapes = {jnp.shape(cycle[key]) for cycle in cycle_msa_features}
        if len(shapes) != 1:
            return None
    return {
        key: jnp.stack([cycle[key] for cycle in cycle_msa_features]) for key in keys
    }
