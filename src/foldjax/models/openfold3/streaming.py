"""Host-selected MSA recycling with at most two device row buffers.

Only the padded managed path uses this scheduler. Native row selection remains
in ``prepare_msa_cycle_features``; its union and index tape never enter a JIT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._compile_policy import compiler_options
from foldjax.models._cp import (
    context_parallel,
    replicate_tree,
    shard_pair_rows,
    shard_single,
)
from foldjax.models.openfold3 import inference as inf
from foldjax.models.openfold3.data.featurize import (
    _COMPACT_MSA_INDICES,
    _COMPACT_MSA_MARKER,
    _MSA_CYCLE_INDICES,
    MSA_ROW_FEATURES,
)
from foldjax.models.openfold3.models.trunk import initialize_trunk, trunk_cycle

_ROW_KEYS = (*MSA_ROW_FEATURES, _COMPACT_MSA_INDICES)
_MSA_KEYS = frozenset((*_ROW_KEYS, _COMPACT_MSA_MARKER, _MSA_CYCLE_INDICES))


def _stage(tree):
    # replicate_tree is deliberately an identity on a single device. Explicit
    # placement is still needed there to prefetch and retain common inputs once.
    return jax.device_put(replicate_tree(tree))


class HostMSACycles:
    """Keep native selections on the host and materialize only the next cycle."""

    def __init__(self, batch, *, depth: int, cycles: int):
        if depth < 1 or cycles < 1:
            raise ValueError("streamed MSA depth and cycles must be positive")
        rows = batch["msa_mask"].shape[1]
        indices = np.asarray(batch[_MSA_CYCLE_INDICES])
        if (
            indices.ndim != 2
            or indices.shape[0] != cycles
            or not 0 < indices.shape[1] <= depth
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= rows)
        ):
            raise ValueError("invalid streamed MSA cycle indices or padding depth")
        self.indices = indices
        self.depth = depth
        self.rows = {k: np.asarray(batch[k]) for k in _ROW_KEYS if k in batch}
        if any(value.shape[:2] != (1, rows) for value in self.rows.values()):
            raise ValueError("streamed MSA row features disagree on storage shape")
        self.marker = (
            {_COMPACT_MSA_MARKER: np.asarray(batch[_COMPACT_MSA_MARKER])}
            if _COMPACT_MSA_MARKER in batch
            else {}
        )
        self.common = {k: v for k, v in batch.items() if k not in _MSA_KEYS}

    def select(self, cycle: int):
        indices = self.indices[cycle]
        selected = dict(self.marker)
        for name, source in self.rows.items():
            # Category 32 reconstructs a zero vector, matching dense padding.
            fill = 32 if name == _COMPACT_MSA_INDICES else 0
            value = np.full(
                (source.shape[0], self.depth, *source.shape[2:]),
                fill,
                dtype=source.dtype,
            )
            value[:, : indices.size] = source[:, indices]
            selected[name] = value
        return selected


class _StreamedGraph:
    def __init__(self, identity, *, compiled=True):
        config = identity.config

        def inputs(batch, params):
            return inf._predict_inputs(
                inf._restore_ref_atom_category_one_hot(batch), params, config
            )

        def initialize(batch, params):
            initial = initialize_trunk(
                inf._restore_ref_atom_category_one_hot(batch),
                params,
                n_query=config.n_query,
                n_key=config.n_key,
                atom_heads=config.atom_heads,
                n_token=config.n_token,
                max_relative_idx=config.max_relative_idx,
                max_relative_chain=config.max_relative_chain,
                glu_backend=config.glu_backend,
            )
            return initial, (
                shard_single(jnp.zeros_like(initial[1])),
                shard_pair_rows(jnp.zeros_like(initial[2])),
            )

        def cycle(batch, msa, params, initial, carry):
            return trunk_cycle(
                batch,
                msa,
                params,
                initial,
                carry,
                no_heads_msa=config.no_heads_msa,
                no_heads_pair=config.no_heads_pair,
                no_heads_pair_bias=config.no_heads_pair_bias,
                opm_first=config.opm_first,
                chunk_size=config.pair_chunk_size,
                glu_backend=config.glu_backend,
            )

        def finish(key, batch, params, table, output, tape, mask, augmentation):
            return inf._predict_from_trunk(
                key,
                inf._restore_ref_atom_category_one_hot(batch),
                params,
                config,
                table,
                trunk_output=output,
                n_chain=identity.n_chain,
                noise_tape=tape,
                noise_mask=mask,
                augmentation_tape=augmentation,
                augment=identity.augment,
                use_trunk_pair_embedding=identity.use_trunk_pair_embedding,
            )

        # All four stages or none: the promise is about this prediction, and a
        # streamed run that compiled three of them under the policy and one
        # without it is not the run the caller asked for. ``None`` keeps the
        # default stages at exactly the arguments they always had.
        options = compiler_options(deterministic=identity.deterministic)
        policy = {} if options is None else {"compiler_options": options}
        wrap = (lambda fn: jax.jit(fn, **policy)) if compiled else lambda fn: fn
        self.inputs = wrap(inputs)
        self.initialize = wrap(initialize)
        # The previous carry is dead after dispatch. Donation preserves the
        # fused loop's ability to reuse its large pair-state allocation.
        self.cycle = (
            jax.jit(cycle, donate_argnums=(4,), **policy) if compiled else cycle
        )
        self.finish = wrap(finish)
        self.config = config

    def __call__(self, key, batch, params, table, tape, mask, augmentation=None):
        if self.config.stop_after_inputs:
            common = {k: v for k, v in batch.items() if k not in _MSA_KEYS}
            return self.inputs(_stage(common), params.trunk)
        provider = HostMSACycles(
            batch,
            depth=self.config.msa_depth,
            cycles=self.config.num_recycles,
        )
        common = _stage(provider.common)
        initial, carry = self.initialize(common, params.trunk)
        current = _stage(provider.select(0))
        for i in range(self.config.num_recycles):
            carry = self.cycle(common, current, params.trunk, initial, carry)
            # Enqueue one lookahead, then fence before admitting another cycle.
            # This bounds live row buffers even with asynchronous JAX dispatch.
            following = (
                _stage(provider.select(i + 1))
                if i + 1 < self.config.num_recycles
                else None
            )
            jax.block_until_ready(carry)
            current = following
        output = (initial[0], *carry)
        return self.finish(key, common, params, table, output, tape, mask, augmentation)

    def _cache_size(self):
        return sum(
            fn._cache_size()
            for fn in (self.inputs, self.initialize, self.cycle, self.finish)
        )

    def clear_cache(self):
        for fn in (self.inputs, self.initialize, self.cycle, self.finish):
            fn.clear_cache()


class _StreamedPool(inf._CompiledPredictPool):
    @staticmethod
    def _new(identity):
        return _StreamedGraph(identity)


# Three executables per profile, with the same bounded eviction discipline as
# fused inference. No checkpoint or host feature arrays are captured by a JIT.
_compiled_streams = _StreamedPool(limit=3 * inf._MAX_RETAINED_EXECUTABLES)


def compile_streamed_predict(
    config,
    representative_atoms,
    *,
    n_chain=None,
    augment=True,
    use_trunk_pair_embedding=True,
    triangle_kernel=None,
    cache_scope=None,
    compiled=True,
    deterministic=False,
):
    """Bind a host scheduler; only fixed-shape stage functions are compiled.

    ``deterministic`` compiles all four stage functions for reduction orders
    that repeat between runs; see :mod:`foldjax.models._compile_policy`.
    """
    if config.msa_depth is None or config.msa_depth < 1:
        raise ValueError("streamed prediction requires a fixed positive msa_depth")
    if not compiled and config.cp_shards > 1:
        raise ValueError("context parallelism requires compiled streamed prediction")
    if deterministic and not compiled:
        # The uncompiled scheduler dispatches its stages operation by
        # operation and owns no executable to carry the option. Running it
        # anyway would report a deterministic run that was not one.
        raise ValueError(
            "deterministic reductions are carried by the compiled graph; "
            "drop compiled=False or deterministic"
        )
    table = inf._validated_representative_atoms(representative_atoms)
    layout = "1d" if config.cp_shards <= 1 else inf.resolve_cp_layout(config)
    config = config._replace(cp_layout=layout)
    scope = None if cache_scope is None else inf.canonical_cache_scope(str(cache_scope))

    @inf.openfold3_precision
    def run(
        key, batch, params, *, noise_tape=None, noise_mask=None, augmentation_tape=None
    ):
        batch = inf._prepare_ref_atom_category_graph_input(batch)
        augmentation = inf._prediction_augmentation_tape(
            augmentation_tape,
            config=config,
            augment=augment,
        )
        route = inf._rng_route(noise_tape, noise_mask)
        kernel = inf.resolve_triangle_kernel(
            triangle_kernel, cp_shards=config.cp_shards
        )
        with (
            inf.triangle_backend(kernel),
            context_parallel(config.cp_shards, layout=layout) as mesh,
        ):
            identity = inf._PredictGraphIdentity(
                config=config,
                n_chain=n_chain,
                augment=augment,
                use_trunk_pair_embedding=use_trunk_pair_embedding,
                rng_route=route,
                triangle_kernel=kernel,
                cp_topology=inf._cp_topology_identity(mesh, layout=layout),
                cache_scope=scope,
                augmentation_taped=augmentation is not None,
                deterministic=deterministic,
            )
            bounded = inf._persistent_cache_is_bounded(scope)
            token = inf.inspect_cache_scope(scope, repair_atime=bounded)
            if token is not None and token.invalidated:
                _compiled_streams.clear_cache()
            args = (
                _stage(key),
                batch,
                _stage(params),
                _stage(table),
                _stage(noise_tape),
                _stage(noise_mask),
                _stage(augmentation),
            )
            result = (
                _compiled_streams(*args, identity=identity)
                if compiled
                else _StreamedGraph(identity, compiled=False)(*args)
            )
            if compiled:
                inf.observe_cache_scope(
                    scope,
                    token=token,
                    require_payload=inf._persistent_cache_matches(scope),
                    require_atime=bounded,
                )
            return result

    return run
