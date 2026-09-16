"""Context-parallel distribution of OpenFold3's diffusion atom graph.

Fold-CP has sharded OpenFold3's pair trunk since the 1-D/2-D triangle stack
landed; the diffusion atom stream was replicated on every device.  Under a
mesh each device held the whole graph: the ``[N_blocks, N_query, N_key, c]``
atom-pair conditioning, both atom transformer stacks, the ``[N_atom, c_atom]``
single stream, and -- the expensive one -- the projected token-pair tensor,
because ``zij[..., q_tok, k_tok, :]`` is a gather the SPMD partitioner can
only satisfy by collecting its operand.

This module distributes that graph over the pair-row mesh axis.  It differs
from Protenix' equivalent (``models/protenix/models/diffusion/_cp.py``) in the
one place that matters, and the difference decides the mechanism:

**OpenFold3's key windows are not a fixed offset from their query block.**
``atom_blocks.block_indices`` *shifts* a window rather than clipping it, by an
amount derived from ``jnp.sum(atom_mask)``.  A block that would start before
atom 0 slides right; a block that would run past the last real atom slides
left to end there -- so a block lying entirely in the atom padding reads the
last ``n_key`` *real* atoms, however far away those are.  Measured on 96 atoms
with 60 real and a 4/8 window, blocks 15..23 all read atoms 52..59.  A halo of
any static width is wrong for those, and the shift is a traced value, so no
static width exists.  The key side is therefore an *index-driven ring gather*
(:func:`foldjax.models._cp_atom.gather_atom_windows_local`) over rotating atom
shards rather than Boltz-2's and Protenix' halo exchange: nothing bounds where
a key may live, which is the property the shift needs.

The second structural difference is that the atom stage runs inside *one*
``shard_map`` body per encoder/decoder rather than one per arithmetic stage.
OpenFold3 reaches its blocking through four shared primitives
(``single_rep_to_blocks``, ``pair_rep_to_blocks``,
``broadcast_token_feat_to_atoms``, ``aggregate_atom_feat_to_tokens``) called
from five places between them, so a per-stage decomposition would have had to
duplicate the encoder body.  Instead each of those four dispatches on an
:class:`AtomBlockPlan` installed by the enclosing sharded body -- the same
``ContextVar`` device Protenix uses for its window trunk builder, and for the
same reason: a plan holds the body's tracers, so it cannot be an argument
threaded through signatures the input embedder and the trunk also call.

With no plan installed -- every serial run, and every context-parallel run
whose atom graph is deliberately replicated -- all four keep their historical
single path, and the serial lowering is pinned byte-identical
(``tests/models/openfold3/test_atom_context_parallel.py``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import cp_grid, cp_layout, cp_mesh, cp_row_shards, pair_spec
from foldjax.models._cp import single_spec as token_spec
from foldjax.models._cp_atom import (
    atom_axis_name,
    atom_spec,
    gather_atom_windows_local,
    gather_token_pairs_to_windows_local,
    replicate_atoms,
    resolve_atom_graph,
    ring_gather_local,
    window_spec,
)

#: Batch features the diffusion atom encoder/decoder read whose axis ``1`` is
#: the atom axis. Everything token-shaped the bodies touch stays replicated:
#: it is linear in the token count, and the *global* value is what two of the
#: four primitives need (see :meth:`AtomBlockPlan.broadcast_token_feat`).
ATOM_AXIS_FEATURES = (
    "ref_pos",
    "ref_charge",
    "ref_mask",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_mask",
    "atom_to_token_index",
)

#: Token-shaped features handed to a sharded body replicated.
TOKEN_REPLICATED_FEATURES = ("token_mask", "num_atoms_per_token")


# --- alignment -------------------------------------------------------------


def atom_block_misalignment(
    *,
    n_atom: int,
    n_token: int,
    n_query: int,
) -> str | None:
    """Say why this shape cannot carry a distributed atom graph, or ``None``.

    Three requirements, each from an operation rather than from caution: the
    atom axis must split into whole query blocks on every CP row (the local
    reshape in :meth:`AtomBlockPlan.single_rep_to_blocks`), the token axis must
    split over the same rows (the aggregate's ``psum_scatter`` and the token
    gather's ring), and under the square grid the token axis must also split
    over columns (the token-pair tile the atom blocks read).

    ``n_key`` is deliberately absent, unlike the shared halo predicate
    (:func:`foldjax.models._cp_atom.atom_window_misalignment`). A halo needs
    every row to own at least its radius; the ring gather here has no radius,
    so a row owning a single query block is legal -- which is the whole reason
    this port states its own requirements instead of reusing that one.
    """

    if cp_mesh() is None:
        return "no context-parallel mesh is active"
    rows, cols = cp_grid()
    alignment = n_query * rows
    if n_atom % alignment:
        return (
            f"{n_atom} atoms is not a multiple of n_query * cp_rows "
            f"({n_query} * {rows} = {alignment})"
        )
    if n_token % rows:
        return f"{n_token} tokens do not divide {rows} CP rows"
    if cp_layout() == "2d" and n_token % cols:
        return f"{n_token} tokens do not divide {cols} CP columns"
    return None


def resolve_atom_windows(
    *,
    requested: bool,
    n_atom: int,
    n_token: int,
    n_query: int,
) -> bool:
    """The shared decision (:func:`resolve_atom_graph`) under this port's blocks.

    ``inference.py`` resolves the option exactly once, here; the message and
    its pad multiples are the ones every port emits.
    """

    return resolve_atom_graph(
        requested=requested,
        n_query=n_query,
        misalignment=lambda: atom_block_misalignment(
            n_atom=n_atom,
            n_token=n_token,
            n_query=n_query,
        ),
    )


def require_atom_windows(*, n_atom: int, n_token: int, n_query: int) -> None:
    """Fail inside an adapter rather than emit a wrong program."""

    reason = atom_block_misalignment(n_atom=n_atom, n_token=n_token, n_query=n_query)
    if reason is not None:
        raise ValueError(f"distributed atom blocks are not available: {reason}")


# --- global block tables ---------------------------------------------------


def block_gather_tables(
    atom_mask: jnp.ndarray,
    atom_to_token_index: jnp.ndarray,
    *,
    n_query: int,
    n_key: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The four window tables, computed once at global atom positions.

    Each is derived from ``jnp.sum(atom_mask)`` over the whole atom axis. That
    reduction is the reason they cannot be built inside a sharded body: with a
    sharded mask it would silently become a per-shard atom count, and every
    window in the tail of the structure would be placed wrongly -- a defect
    that changes numbers rather than shapes, and so one no shape assertion
    catches.

    Returns ``(indices, invalid, mask_k, key_token_index)``, each
    ``[..., N_blocks, n_key]``: the global atom index of every key slot, the
    slots that fell outside the real atoms, ``atom_mask`` gathered through the
    indices with those slots zeroed, and the owning token of every key slot.
    """

    from foldjax.models.openfold3.models.atom_blocks import block_indices

    indices, invalid = block_indices(atom_mask, n_query=n_query, n_key=n_key)
    mask_k = jnp.take_along_axis(atom_mask[..., None, :], indices, axis=-1)
    mask_k = jnp.where(invalid, 0.0, mask_k)
    key_token_index = jnp.take_along_axis(
        atom_to_token_index[..., None, :].astype(jnp.int32), indices, axis=-1
    )
    return indices, invalid, mask_k, key_token_index


# --- the plan --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AtomBlockPlan:
    """How one device turns its atom shard into the blocks it owns.

    Every table is this device's window-sharded slice of a globally computed
    one; see :func:`block_gather_tables` for why they cannot be rebuilt here.

    ``mask_k`` is built from the same ``atom_mask`` the enclosing body was
    given. All four dispatch sites call their primitive with that mask --
    OpenFold3's diffusion atom transformer always passes one, so the ``None``
    fallback to ``jnp.ones`` is unreachable on this path -- and the only
    freedom left is the element type, which the methods below follow.
    """

    n_query: int
    n_key: int
    #: Mesh axis owning the atom and query-block axes, and its size.
    axis_name: str
    axis_size: int
    #: Number of CP rows the token-pair tile rotates through.
    rows: int
    indices: jnp.ndarray
    invalid: jnp.ndarray
    mask_k: jnp.ndarray
    key_token_index: jnp.ndarray

    # -- helpers --

    def _flat_tables(self, batch: int) -> tuple[jnp.ndarray, ...]:
        shape = (batch,) + tuple(self.indices.shape[-2:])
        return tuple(
            jnp.broadcast_to(table.reshape((-1,) + table.shape[-2:]), shape)
            for table in (
                self.indices,
                self.invalid,
                self.mask_k,
                self.key_token_index,
            )
        )

    def _check(self, n_query: int, n_key: int) -> None:
        if n_query != self.n_query or n_key != self.n_key:
            raise ValueError(
                "the installed atom-block plan was built for "
                f"n_query={self.n_query}, n_key={self.n_key}; the call asks "
                f"for n_query={n_query}, n_key={n_key}"
            )

    def _blocks(self, n_atom_local: int) -> int:
        if n_atom_local % self.n_query:
            raise ValueError(
                f"local atom shard {n_atom_local} is not a multiple of the "
                f"query block {self.n_query}"
            )
        blocks = n_atom_local // self.n_query
        if blocks != int(self.indices.shape[-2]):
            raise ValueError(
                f"the plan covers {int(self.indices.shape[-2])} blocks but the "
                f"shard holds {blocks}"
            )
        return blocks

    def _pair_mask(
        self,
        atom_mask: jnp.ndarray,
        leading: tuple[int, ...],
        n_atom_local: int,
        blocks: int,
    ) -> jnp.ndarray:
        mask = jnp.broadcast_to(atom_mask, (*leading, n_atom_local))
        mask_q = mask.reshape((*leading, blocks, self.n_query))
        mask_k = jnp.broadcast_to(
            self.mask_k.astype(mask_q.dtype), (*leading, blocks, self.n_key)
        )
        return mask_q[..., None] * mask_k[..., None, :]

    # -- the four dispatched primitives --

    def single_rep_to_blocks(
        self,
        ql: jnp.ndarray,
        atom_mask: jnp.ndarray,
        *,
        n_query: int,
        n_key: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """The sharded spelling of ``atom_blocks.single_rep_to_blocks``.

        The query side is a pure reshape -- a sharded atom axis is required to
        be an exact multiple of ``n_query * rows``, so there is no ragged final
        block to pad and crop. The key side rotates the atom shards.
        """

        self._check(n_query, n_key)
        leading = tuple(ql.shape[:-2])
        n_atom_local, channels = ql.shape[-2:]
        blocks = self._blocks(int(n_atom_local))
        query_blocks = ql.reshape((*leading, blocks, self.n_query, channels))

        flat = ql.reshape((-1, int(n_atom_local), int(channels)))
        indices, invalid, _, _ = self._flat_tables(flat.shape[0])
        key_blocks = gather_atom_windows_local(
            flat,
            indices,
            jnp.logical_not(invalid),
            axis_name=self.axis_name,
            axis_size=self.axis_size,
        )
        key_blocks = key_blocks.reshape((*leading, blocks, self.n_key, int(channels)))
        return (
            query_blocks,
            key_blocks,
            self._pair_mask(atom_mask, leading, int(n_atom_local), blocks),
        )

    def pair_rep_to_blocks(
        self,
        zij_tile: jnp.ndarray,
        atom_to_token_index: jnp.ndarray,
        atom_mask: jnp.ndarray,
        *,
        n_query: int,
        n_key: int,
    ) -> jnp.ndarray:
        """The sharded spelling of ``atom_blocks.pair_rep_to_blocks``.

        ``zij_tile`` is this device's pair tile, never the whole token-pair
        tensor: row tiles rotate over CP rows and, under the square grid, a
        final column ``psum`` assembles the atom-block result without
        reconstructing the operand anywhere.

        The query side is handed in all-valid, matching serial: a padded query
        slot reads token ``atom_to_token_index[l]`` in the serial gather too,
        and the pair mask applied at the end is what removes it.
        """

        self._check(n_query, n_key)
        leading = tuple(zij_tile.shape[:-3])
        flat_batch = 1
        for size in leading:
            flat_batch *= int(size)
        n_atom_local = int(atom_to_token_index.shape[-1])
        blocks = self._blocks(n_atom_local)
        _, invalid, _, key_token_index = self._flat_tables(flat_batch)

        query_token = jnp.broadcast_to(
            atom_to_token_index, (*leading, n_atom_local)
        ).reshape((flat_batch, blocks, self.n_query))
        plm = gather_token_pairs_to_windows_local(
            zij_tile.reshape((flat_batch, *zij_tile.shape[-3:])),
            query_token.astype(jnp.int32),
            jnp.ones(query_token.shape, dtype=bool),
            key_token_index,
            jnp.logical_not(invalid),
            rows=self.rows,
            row_axis=self.axis_name,
        )
        plm = plm.reshape((*leading, blocks, self.n_query, self.n_key, plm.shape[-1]))
        pair_mask = self._pair_mask(atom_mask, leading, n_atom_local, blocks)
        return plm * pair_mask[..., None]

    def broadcast_token_feat(
        self,
        token_mask: jnp.ndarray,
        num_atoms_per_token: jnp.ndarray,
        token_feat: jnp.ndarray,
        atom_to_token_index: jnp.ndarray | None,
        *,
        n_atom: int,
    ) -> jnp.ndarray:
        """Gather a token-sharded feature onto this row's atoms.

        ``token_mask`` and ``num_atoms_per_token`` arrive replicated, so the
        prefix-validity threshold ``positions < sum(counts)`` is the same whole
        reduction the serial path performs rather than a per-shard count, and
        the clip bound is the global token count. Only ``token_feat`` is
        sharded; the slice of the mask that multiplies it is cut out of the
        replicated one, which is a dynamic slice rather than a collective.
        """

        if atom_to_token_index is None:
            raise ValueError(
                "a distributed atom graph needs the validated "
                "atom_to_token_index; the count-based fallback derives the "
                "owner table from a cumulative sum over the whole token axis, "
                "which a token-sharded body cannot see"
            )
        if int(atom_to_token_index.shape[-1]) != n_atom:
            raise ValueError(
                "atom_to_token_index length must equal the requested atom count"
            )
        n_token = int(token_mask.shape[-1])
        tokens_local = int(token_feat.shape[-2])
        owner = jax.lax.axis_index(self.axis_name)
        mask_local = jax.lax.dynamic_slice_in_dim(
            jnp.broadcast_to(token_mask, (*token_feat.shape[:-2], n_token)),
            owner * tokens_local,
            tokens_local,
            axis=-1,
        )
        token_feat = token_feat * mask_local[..., None]

        counts = num_atoms_per_token * token_mask
        n_atom_local = int(atom_to_token_index.shape[-1])
        positions = owner * n_atom_local + jnp.arange(n_atom_local)
        valid = positions < jnp.sum(counts, axis=-1, keepdims=True)

        safe = jnp.clip(atom_to_token_index.astype(jnp.int32), 0, n_token - 1)
        leading = tuple(token_feat.shape[:-2])
        flat = token_feat.reshape((-1, tokens_local, int(token_feat.shape[-1])))
        index = jnp.broadcast_to(safe, (*leading, n_atom_local)).reshape(
            (flat.shape[0], n_atom_local)
        )
        gathered = ring_gather_local(
            flat,
            index,
            jnp.ones(index.shape, dtype=bool),
            axis_name=self.axis_name,
            axis_size=self.axis_size,
        )
        gathered = gathered.reshape((*leading, n_atom_local, int(token_feat.shape[-1])))
        return jnp.where(
            jnp.broadcast_to(valid, (*leading, n_atom_local))[..., None],
            gathered,
            0.0,
        )

    def aggregate_atom_feat_to_tokens(
        self,
        atom_feat: jnp.ndarray,
        atom_to_token_index: jnp.ndarray,
        atom_mask: jnp.ndarray,
        *,
        n_token: int,
        aggregate: str,
        eps: float,
    ) -> jnp.ndarray:
        """Scatter this row's atoms into the token slice this row owns.

        The one-hot contraction is kept rather than replaced with a scatter:
        ``atomize.aggregate_atom_feat_to_tokens`` records that a GPU
        scatter-add's summation order is not reproducible and that the
        diffusion rollout amplifies it, and that contract does not change
        because the atoms are now split. The overflow bin masked atoms are
        routed to is dropped *before* the reduce-scatter, because ``n_token +
        1`` does not divide the CP rows.
        """

        if aggregate not in ("mean", "sum"):
            raise ValueError(f"invalid aggregation function: {aggregate}")
        mask = atom_mask.astype(atom_feat.dtype)
        atom_feat = atom_feat * mask[..., None]
        index = (atom_to_token_index * mask + n_token * (1.0 - mask)).astype(jnp.int32)
        membership = (index[..., :, None] == jnp.arange(n_token + 1)).astype(
            atom_feat.dtype
        )
        totals = jnp.einsum("...ac,...at->...tc", atom_feat, membership)
        counts = jnp.einsum("...a,...at->...t", mask, membership)
        totals = jax.lax.psum_scatter(
            totals[..., :n_token, :],
            self.axis_name,
            scatter_dimension=totals.ndim - 2,
            tiled=True,
        )
        counts = jax.lax.psum_scatter(
            counts[..., :n_token],
            self.axis_name,
            scatter_dimension=counts.ndim - 1,
            tiled=True,
        )
        if aggregate == "sum":
            return totals
        return totals / (counts[..., None] + eps)


_ACTIVE: ContextVar[AtomBlockPlan | None] = ContextVar(
    "foldjax_openfold3_atom_block_plan",
    default=None,
)


def atom_block_plan() -> AtomBlockPlan | None:
    """The plan installed by the enclosing sharded body, or ``None``.

    ``None`` is the ordinary case. It includes a context-parallel run whose
    atom graph is deliberately replicated and every call from the input
    embedder, the trunk and the frame builder, so no caller may treat a
    missing plan as an error.
    """

    return _ACTIVE.get()


@contextlib.contextmanager
def use_atom_block_plan(plan: AtomBlockPlan | None) -> Iterator[None]:
    """Install one plan for the duration of a traced sharded body."""

    token = _ACTIVE.set(plan)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


# --- the two sharded bodies ------------------------------------------------


def _feature_specs(batch: Mapping[str, Any], *, n_atom: int) -> dict[str, Any]:
    """Atom axis 1 for the atom features, replication for the token ones."""

    specs: dict[str, Any] = {}
    for name in ATOM_AXIS_FEATURES:
        if name not in batch:
            continue
        value = jnp.asarray(batch[name])
        if value.ndim < 2 or int(value.shape[1]) != n_atom:
            raise ValueError(
                f"{name} must carry the atom axis at position 1 for a "
                f"distributed atom graph; got shape {value.shape}"
            )
        specs[name] = atom_spec(value.ndim, atom_axis=1)
    for name in TOKEN_REPLICATED_FEATURES:
        if name in batch:
            specs[name] = PartitionSpec()
    return specs


def _sharded_features(batch: Mapping[str, Any], specs: Mapping[str, Any]) -> dict:
    return {name: jnp.asarray(batch[name]) for name in specs}


def _plan(
    tables: Sequence[jnp.ndarray],
    *,
    n_query: int,
    n_key: int,
) -> AtomBlockPlan:
    return AtomBlockPlan(
        n_query=n_query,
        n_key=n_key,
        axis_name=atom_axis_name(),
        axis_size=cp_row_shards(),
        rows=cp_grid()[0],
        indices=tables[0],
        invalid=tables[1],
        mask_k=tables[2],
        key_token_index=tables[3],
    )


def atom_attention_encoder_cp(
    batch: Mapping[str, jnp.ndarray],
    params: Any,
    *,
    n_query: int,
    n_key: int,
    no_heads: int,
    n_token: int,
    rl: jnp.ndarray,
    si_trunk: jnp.ndarray,
    zij_trunk: jnp.ndarray,
    inf: float,
    eps: float,
    glu_backend: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run AF3 Algorithm 5's diffusion encoder with its atom graph split.

    ``params`` is closed over rather than passed as a ``shard_map`` operand:
    the parameter NamedTuples carry optional ``None`` members that select code
    paths (``noisy_position_embedder``) and are read by ``can_scan`` before any
    array is touched, and a flattened operand would turn the structure into
    something the first ``if`` cannot read. Closure keeps the arrays
    replicated, which is what they already are.
    """

    from foldjax.models.openfold3.models.atom_features import atom_attention_encoder

    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("atom_attention_encoder_cp requires an active mesh")
    n_atom = int(jnp.asarray(batch["atom_mask"]).shape[1])
    require_atom_windows(n_atom=n_atom, n_token=n_token, n_query=n_query)

    specs = _feature_specs(batch, n_atom=n_atom)
    features = _sharded_features(batch, specs)
    tables = block_gather_tables(
        features["atom_mask"],
        features["atom_to_token_index"],
        n_query=n_query,
        n_key=n_key,
    )
    table_specs = tuple(window_spec(table.ndim, window_axis=-2) for table in tables)

    def local(features_local, rl_local, si_local, zij_local, tables_local):
        with use_atom_block_plan(_plan(tables_local, n_query=n_query, n_key=n_key)):
            return atom_attention_encoder(
                features_local,
                params,
                n_query=n_query,
                n_key=n_key,
                no_heads=no_heads,
                n_token=n_token,
                rl=rl_local,
                si_trunk=si_local,
                zij_trunk=zij_local,
                inf=inf,
                eps=eps,
                glu_backend=glu_backend,
            )

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            specs,
            atom_spec(rl.ndim, atom_axis=-2),
            token_spec(si_trunk.ndim, token_axis=-2),
            pair_spec(zij_trunk.ndim, row_axis=-3, col_axis=-2),
            table_specs,
        ),
        out_specs=(
            token_spec(3, token_axis=-2),
            atom_spec(3, atom_axis=-2),
            atom_spec(3, atom_axis=-2),
            window_spec(5, window_axis=-4),
        ),
    )(features, rl, si_trunk, zij_trunk, tuple(tables))


def atom_attention_decoder_cp(
    batch: Mapping[str, jnp.ndarray],
    ai: jnp.ndarray,
    ql: jnp.ndarray,
    cl: jnp.ndarray,
    plm: jnp.ndarray,
    params: Any,
    *,
    n_query: int,
    n_key: int,
    no_heads: int,
    inf: float,
    eps: float,
    glu_backend: str,
) -> jnp.ndarray:
    """Run AF3 Algorithm 6 on the row-owned atoms, then replicate the update.

    The coordinate update is replicated on the way out: the sampler's state
    and its noise tape are replicated, and the array is ``[samples, atoms, 3]``
    -- linear in the atom count and three channels wide.
    """

    from foldjax.models.openfold3.models.atom_features import atom_attention_decoder

    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("atom_attention_decoder_cp requires an active mesh")
    n_atom = int(jnp.asarray(batch["atom_mask"]).shape[1])
    n_token = int(jnp.asarray(batch["token_mask"]).shape[-1])
    require_atom_windows(n_atom=n_atom, n_token=n_token, n_query=n_query)

    specs = _feature_specs(batch, n_atom=n_atom)
    features = _sharded_features(batch, specs)
    tables = block_gather_tables(
        features["atom_mask"],
        features["atom_to_token_index"],
        n_query=n_query,
        n_key=n_key,
    )
    table_specs = tuple(window_spec(table.ndim, window_axis=-2) for table in tables)

    def local(features_local, ai_local, ql_local, cl_local, plm_local, tables_local):
        with use_atom_block_plan(_plan(tables_local, n_query=n_query, n_key=n_key)):
            return atom_attention_decoder(
                features_local,
                ai_local,
                ql_local,
                cl_local,
                plm_local,
                params,
                n_query=n_query,
                n_key=n_key,
                no_heads=no_heads,
                inf=inf,
                eps=eps,
                glu_backend=glu_backend,
            )

    update = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            specs,
            token_spec(ai.ndim, token_axis=-2),
            atom_spec(ql.ndim, atom_axis=-2),
            atom_spec(cl.ndim, atom_axis=-2),
            window_spec(plm.ndim, window_axis=-4),
            table_specs,
        ),
        out_specs=atom_spec(3, atom_axis=-2),
    )(features, ai, ql, cl, plm, tuple(tables))
    return replicate_atoms(update)


__all__: Sequence[str] = (
    "ATOM_AXIS_FEATURES",
    "TOKEN_REPLICATED_FEATURES",
    "AtomBlockPlan",
    "atom_attention_decoder_cp",
    "atom_attention_encoder_cp",
    "atom_block_misalignment",
    "atom_block_plan",
    "block_gather_tables",
    "require_atom_windows",
    "resolve_atom_windows",
    "use_atom_block_plan",
)
