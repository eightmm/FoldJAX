"""Context-parallel sharding for the AF3-family pair representations.

Fold-CP -- NVIDIA's context-parallel design for biomolecular folding, which
OpenDDE vendors -- shards the ``[N, N, C]`` pair representation across devices
so no single device holds the whole quadratic state. Their torch
implementation hand-writes the whole communication stack because torch has no
automatic partitioner; the JAX form keeps the *layout* and lets XLA's SPMD
partitioner place the collectives wherever a plain sharding constraint
suffices, dropping to ``shard_map`` + ``lax.ppermute`` only where a schedule
has to be spelled out.

Two layouts live here:

``1d``
    Pair rows split across one axis, columns whole. Every device holds
    ``[N/P, N, C]``. Simple, and enough to move a job that does not fit one
    card, but the column axis stays full width, so a per-device buffer is
    still ``O(N^2/P)`` with the constant of a whole row of tiles.

``2d``
    Fold-CP's own layout: a square ``Pr x Pc`` grid with rows on one mesh axis
    and columns on the other, so a device holds ``[N/Pr, N/Pc, C]`` and the
    per-device pair cost is ``O(N^2/P_total)``. The ring algorithms this
    enables (Cannon multiplication, rotating-bias attention) never
    materialise a full-width operand.

State is a module global read at trace time, the same pattern as the
``PROTENIX_TRIANGLE_BACKEND`` environment default. It is safe under ``jit``
because every entry point that activates a mesh also carries the shard count
as a static argument, so a change of mesh is a retrace, never a stale cache
hit.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

#: Mesh axis of the one-dimensional layout.
CP_AXIS = "cp"
#: Mesh axes of the two-dimensional layout: pair rows and pair columns.
CP_ROW_AXIS = "cp_row"
CP_COL_AXIS = "cp_col"

_MESH: Mesh | None = None


def cp_mesh() -> Mesh | None:
    """Return the active context-parallel mesh, or ``None`` outside one."""
    return _MESH


def cp_shards() -> int:
    """Total device count of the active mesh; 1 when none is active."""
    return 1 if _MESH is None else int(_MESH.devices.size)


def cp_row_shards() -> int:
    """Devices along the pair-row axis.

    The row-sharded schedules (the 1-D triangle attention, which stays
    row-only even under a 2-D mesh because its softmax spans whole columns)
    pad to this, not to the total device count.
    """
    return cp_grid()[0]


def cp_layout() -> str | None:
    """``"1d"``, ``"2d"``, or ``None`` when no mesh is active."""
    if _MESH is None:
        return None
    return "2d" if len(_MESH.axis_names) == 2 else "1d"


def cp_grid() -> tuple[int, int]:
    """``(rows, cols)`` of the active grid; ``(P, 1)`` for the 1-D layout."""
    if _MESH is None:
        return (1, 1)
    shape = _MESH.devices.shape
    if len(shape) == 2:
        return (int(shape[0]), int(shape[1]))
    return (int(shape[0]), 1)


@contextlib.contextmanager
def context_parallel(
    n_devices: int,
    *,
    layout: str = "1d",
    devices: list[jax.Device] | None = None,
) -> Iterator[Mesh | None]:
    """Activate a context-parallel mesh over the first ``n_devices`` devices.

    ``layout="2d"`` builds Fold-CP's square grid and therefore requires a
    perfect-square device count; ``"1d"`` splits rows only. ``n_devices <= 1``
    is a null context either way, so callers can pass their shard count
    through unconditionally.
    """
    global _MESH
    if layout not in {"1d", "2d"}:
        raise ValueError(f"unsupported context-parallel layout: {layout!r}")
    if n_devices <= 1:
        yield None
        return
    if _MESH is not None:
        raise RuntimeError("context_parallel does not nest")
    side = math.isqrt(n_devices)
    if layout == "2d" and side * side != n_devices:
        # Checked before the device pool: squareness is a property of the
        # request, so the error should not depend on what machine it runs on.
        raise ValueError(
            "the two-dimensional layout needs a square device count; "
            f"got {n_devices}. Fold-CP's ring schedules assume a square "
            "grid, so a rectangle is refused rather than silently degraded."
        )
    pool = list(jax.devices()) if devices is None else list(devices)
    if len(pool) < n_devices:
        raise ValueError(
            f"context parallelism over {n_devices} devices requested but only "
            f"{len(pool)} JAX device(s) are visible"
        )
    chosen = np.asarray(pool[:n_devices])
    if layout == "2d":
        mesh = Mesh(chosen.reshape(side, side), (CP_ROW_AXIS, CP_COL_AXIS))
    else:
        mesh = Mesh(chosen, (CP_AXIS,))
    _MESH = mesh
    try:
        yield mesh
    finally:
        _MESH = None


def _resolve_axis(axis: int, ndim: int, *, what: str) -> int:
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise ValueError(f"{what} {axis} out of range for rank {ndim}")
    return resolved


def pair_spec(
    ndim: int,
    *,
    row_axis: int = -3,
    col_axis: int | None = None,
) -> PartitionSpec:
    """Partition spec for a pair-shaped tensor under the active layout.

    Under ``1d`` only ``row_axis`` is split. Under ``2d`` the column axis is
    split too; it defaults to the axis just after the rows, which is the
    ``j`` of ``[..., i, j, C]`` and of a channel-less ``[..., i, j]`` mask.
    Tensors whose ``j`` sits elsewhere (attention biases carrying head and
    broadcast axes) pass ``col_axis`` explicitly.
    """
    entries: list[str | None] = [None] * ndim
    row = _resolve_axis(row_axis, ndim, what="row axis")
    if cp_layout() == "2d":
        column = row + 1 if col_axis is None else _resolve_axis(
            col_axis, ndim, what="column axis"
        )
        if column == row:
            raise ValueError("row and column axes must differ")
        entries[row] = CP_ROW_AXIS
        entries[column] = CP_COL_AXIS
    else:
        entries[row] = CP_AXIS
    return PartitionSpec(*entries)


def pair_row_spec(ndim: int, *, row_axis: int = -3) -> PartitionSpec:
    """Spec splitting only ``row_axis``, whatever the layout.

    The ``shard_map`` bodies of the 1-D schedules take whole rows, so they
    need this even when a 2-D mesh names two axes.
    """
    entries: list[str | None] = [None] * ndim
    entries[_resolve_axis(row_axis, ndim, what="row axis")] = (
        CP_ROW_AXIS if cp_layout() == "2d" else CP_AXIS
    )
    return PartitionSpec(*entries)


def single_spec(ndim: int, *, token_axis: int = -2) -> PartitionSpec:
    """Spec for a single (per-token) representation: tokens on the row axis.

    Fold-CP keeps the single stream sharded on the same mesh axis as the pair
    rows and replicated on the other, so a pair-row-local block of the single
    representation is already resident where the pair rows are.
    """
    entries: list[str | None] = [None] * ndim
    entries[_resolve_axis(token_axis, ndim, what="token axis")] = (
        CP_ROW_AXIS if cp_layout() == "2d" else CP_AXIS
    )
    return PartitionSpec(*entries)


def shard_pair_rows(
    x: jax.Array,
    *,
    row_axis: int = -3,
    col_axis: int | None = None,
) -> jax.Array:
    """Constrain a pair-shaped tensor to the active layout.

    Identity when no mesh is active, so call sites need no branching. Under
    the 2-D layout this splits the column axis as well -- the name is kept
    because every existing call site means "lay this pair tensor out the
    standard way".
    """
    if _MESH is None:
        return x
    return jax.lax.with_sharding_constraint(
        x,
        NamedSharding(_MESH, pair_spec(x.ndim, row_axis=row_axis, col_axis=col_axis)),
    )


def shard_single(x: jax.Array, *, token_axis: int = -2) -> jax.Array:
    """Constrain a per-token representation to the single-stream layout."""
    if _MESH is None:
        return x
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(_MESH, single_spec(x.ndim, token_axis=token_axis))
    )


# --- ring primitives -------------------------------------------------------
#
# Every Fold-CP schedule is a *static* permutation of grid coordinates, so all
# of them are one `lax.ppermute` each. These helpers build the permutation
# lists; they are only meaningful inside a `shard_map` whose mesh carries the
# named axes.


def grid_axes() -> tuple[str, str]:
    """The ``(row, column)`` mesh axis names of the 2-D layout."""
    if cp_layout() != "2d":
        raise RuntimeError("grid axes are only defined for the 2-D layout")
    return (CP_ROW_AXIS, CP_COL_AXIS)


def _flat(
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]], side: int
) -> list[tuple[int, int]]:
    """Row-major flatten of coordinate pairs.

    ``lax.ppermute`` over a tuple of mesh axes indexes their product in
    row-major order, not by coordinate, so every schedule below is written in
    coordinates for legibility and flattened here.
    """
    return [(a * side + b, c * side + d) for (a, b), (c, d) in pairs]


def transpose_perm(side: int) -> list[tuple[int, int]]:
    """``(i, j) -> (j, i)``: the grid transpose, an involution."""
    return _flat(
        [((i, j), (j, i)) for i in range(side) for j in range(side)], side
    )


def row_skew_perm(side: int, *, sign: int = -1) -> list[tuple[int, int]]:
    """Cannon's initial LHS alignment: shift row ``i`` along columns by ``i``.

    ``sign=-1`` is the standard left-skew that puts ``A[i, (j + i) % P]`` on
    device ``(i, j)``; ``sign=+1`` is its inverse.
    """
    return _flat(
        [
            ((i, j), (i, (j + sign * i) % side))
            for i in range(side)
            for j in range(side)
        ],
        side,
    )


def col_skew_perm(side: int, *, sign: int = -1) -> list[tuple[int, int]]:
    """Cannon's initial RHS alignment: shift column ``j`` along rows by ``j``."""
    return _flat(
        [
            ((i, j), ((i + sign * j) % side, j))
            for i in range(side)
            for j in range(side)
        ],
        side,
    )


def ring_perm(side: int, *, axis: str, delta: int = 1) -> list[tuple[int, int]]:
    """One ring hop of ``delta`` along ``axis`` of the 2-D grid."""
    if axis == CP_ROW_AXIS:
        pairs = [
            ((i, j), ((i + delta) % side, j))
            for i in range(side)
            for j in range(side)
        ]
    elif axis == CP_COL_AXIS:
        pairs = [
            ((i, j), (i, (j + delta) % side))
            for i in range(side)
            for j in range(side)
        ]
    else:
        raise ValueError(f"unknown grid axis: {axis!r}")
    return _flat(pairs, side)


def permute(x: jax.Array, perm: Sequence) -> jax.Array:
    """Apply a grid permutation to ``x`` inside a ``shard_map`` body."""
    return jax.lax.ppermute(x, axis_name=grid_axes(), perm=list(perm))


def replicate_tree(tree):
    """Place every array leaf of ``tree`` replicated on the active mesh.

    Inputs to a context-parallel program must live on the mesh's device set;
    a checkpoint committed to one device would otherwise fail the jit call's
    device-assignment check. Identity when no mesh is active.
    """
    if _MESH is None:
        return tree
    sharding = NamedSharding(_MESH, PartitionSpec())

    def place(leaf):
        # Trees such as call kwargs and feature dicts mix numeric arrays with
        # strings, flags, and string-dtype numpy arrays; only numeric or bool
        # arrays have a device to move.
        # Excluding by kind rather than including by `np.number`: the custom
        # ml_dtypes (bfloat16 and friends) register as kind 'V', which
        # `issubdtype` refuses, and skipping a bfloat16 checkpoint here would
        # fail the multi-device call's device-assignment check later. JAX
        # extended dtypes (typed PRNG keys) have no `kind` at all and are
        # device-movable, so a missing kind means "place it".
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            kind = getattr(leaf.dtype, "kind", None)
            if kind not in ("U", "S", "O"):
                return jax.device_put(leaf, sharding)
        return leaf

    return jax.tree.map(place, tree)
