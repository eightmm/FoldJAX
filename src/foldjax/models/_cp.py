"""Context-parallel sharding for AF3-family pair representations.

Fold-CP shards the quadratic ``[N, N, C]`` pair state across a JAX mesh.  The
one-dimensional layout splits pair rows; the two-dimensional layout uses a
square row/column grid and supports gather-free Cannon and ring schedules.

The active runtime is context-local rather than module-global.  That matters
for serving and tests: concurrent requests may use different device meshes, and
an ambient process-wide mesh can otherwise leak into an unrelated JIT trace.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

#: Mesh axis of the one-dimensional layout.
CP_AXIS = "cp"
#: Mesh axes of the two-dimensional layout: pair rows and pair columns.
CP_ROW_AXIS = "cp_row"
CP_COL_AXIS = "cp_col"

#: Input feature names whose two token axes may safely enter the program already
#: pair-sharded.  Atom and single-stream features are deliberately absent:
#: those paths are still replicated unless a model supplies an explicit
#: distributed atom-window implementation.
PAIR_FEATURE_NAMES = frozenset(
    {
        "contact_conditioning",
        "contact_threshold",
        "disto_target",
        "pair_mask",
        "relp",
        "token_bonds",
        "token_pair_pad_mask",
        "type_bonds",
    }
)


@dataclass(frozen=True, slots=True)
class CPRuntime:
    """Immutable identity of one active context-parallel program."""

    mesh: Mesh
    layout: str

    @property
    def shards(self) -> int:
        return int(self.mesh.devices.size)

    @property
    def grid(self) -> tuple[int, int]:
        shape = self.mesh.devices.shape
        if len(shape) == 2:
            return (int(shape[0]), int(shape[1]))
        return (int(shape[0]), 1)

    @property
    def identity(self) -> tuple[str, int, tuple[int, int], tuple[str, ...]]:
        """Hashable topology identity suitable for logs and cache profiles."""

        return (
            self.layout,
            self.shards,
            self.grid,
            tuple(str(axis) for axis in self.mesh.axis_names),
        )


_RUNTIME: ContextVar[CPRuntime | None] = ContextVar(
    "foldjax_context_parallel_runtime",
    default=None,
)


def cp_runtime() -> CPRuntime | None:
    """Return the active immutable runtime, or ``None`` outside CP."""

    return _RUNTIME.get()


def cp_mesh() -> Mesh | None:
    """Return the active context-parallel mesh, or ``None`` outside one."""

    runtime = cp_runtime()
    return None if runtime is None else runtime.mesh


def cp_shards() -> int:
    """Total device count of the active mesh; one when none is active."""

    runtime = cp_runtime()
    return 1 if runtime is None else runtime.shards


def cp_row_shards() -> int:
    """Devices along the pair-row axis."""

    return cp_grid()[0]


def cp_layout() -> str | None:
    """Return ``"1d"``, ``"2d"``, or ``None`` when no mesh is active."""

    runtime = cp_runtime()
    return None if runtime is None else runtime.layout


def cp_grid() -> tuple[int, int]:
    """Return ``(rows, cols)``; ``(P, 1)`` for the one-dimensional layout."""

    runtime = cp_runtime()
    return (1, 1) if runtime is None else runtime.grid


def cp_identity() -> tuple[str, int, tuple[int, int], tuple[str, ...]]:
    """Stable identity of the current topology.

    The serial identity is explicit rather than ``None`` so callers can place
    it directly in diagnostics or a JSON-normalised compilation profile.
    """

    runtime = cp_runtime()
    if runtime is None:
        return ("serial", 1, (1, 1), ())
    return runtime.identity


def resolve_cp_layout(layout: str, n_devices: int, *, auto: str = "1d") -> str:
    """Validate and resolve a public CP layout request.

    ``auto`` intentionally defaults to ``"1d"`` until a model has recorded its
    own two-dimensional GPU evidence.  A caller with such evidence may pass
    ``auto="2d"`` explicitly.  Two-dimensional layouts require a non-trivial
    perfect-square device count.
    """

    if isinstance(n_devices, bool) or not isinstance(n_devices, int):
        raise ValueError("context-parallel device count must be an integer")
    if n_devices < 1:
        raise ValueError("context-parallel device count must be positive")
    if layout not in {"auto", "1d", "2d"}:
        raise ValueError(
            f"context-parallel layout must be 'auto', '1d', or '2d'; got {layout!r}"
        )
    if auto not in {"1d", "2d"}:
        raise ValueError(f"auto layout must resolve to '1d' or '2d'; got {auto!r}")
    resolved = auto if layout == "auto" else layout
    side = math.isqrt(n_devices)
    if resolved == "2d" and (n_devices <= 1 or side * side != n_devices):
        raise ValueError(
            "the two-dimensional layout needs a perfect-square device count "
            f"greater than one; got {n_devices}"
        )
    return resolved


@contextlib.contextmanager
def context_parallel(
    n_devices: int,
    *,
    layout: str = "1d",
    devices: list[jax.Device] | None = None,
) -> Iterator[Mesh | None]:
    """Activate a task-local context-parallel mesh.

    ``layout="2d"`` builds Fold-CP's square grid.  ``n_devices == 1`` is a
    serial null context.  Contexts never nest, including a nominal one-device
    context inside a distributed one: allowing that would make the yielded
    value disagree with what :func:`cp_mesh` reports.
    """

    if cp_runtime() is not None:
        raise RuntimeError("context_parallel does not nest")
    resolved_layout = resolve_cp_layout(layout, n_devices)
    if n_devices == 1:
        yield None
        return

    pool = list(jax.devices()) if devices is None else list(devices)
    if len(pool) < n_devices:
        raise ValueError(
            f"context parallelism over {n_devices} devices requested but only "
            f"{len(pool)} JAX device(s) are visible"
        )
    chosen = np.asarray(pool[:n_devices])
    if resolved_layout == "2d":
        side = math.isqrt(n_devices)
        mesh = Mesh(chosen.reshape(side, side), (CP_ROW_AXIS, CP_COL_AXIS))
    else:
        mesh = Mesh(chosen, (CP_AXIS,))

    token = _RUNTIME.set(CPRuntime(mesh=mesh, layout=resolved_layout))
    try:
        yield mesh
    finally:
        _RUNTIME.reset(token)


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
    """Partition spec for a pair-shaped tensor under the active layout."""

    entries: list[str | None] = [None] * ndim
    row = _resolve_axis(row_axis, ndim, what="row axis")
    if cp_layout() == "2d":
        column = (
            row + 1
            if col_axis is None
            else _resolve_axis(col_axis, ndim, what="column axis")
        )
        if column == row:
            raise ValueError("row and column axes must differ")
        if column >= ndim:
            raise ValueError(
                "default pair column axis falls outside the tensor; "
                "pass col_axis explicitly"
            )
        entries[row] = CP_ROW_AXIS
        entries[column] = CP_COL_AXIS
    else:
        entries[row] = CP_AXIS
    return PartitionSpec(*entries)


def pair_row_spec(ndim: int, *, row_axis: int = -3) -> PartitionSpec:
    """Spec splitting only ``row_axis``, whatever the active layout."""

    entries: list[str | None] = [None] * ndim
    entries[_resolve_axis(row_axis, ndim, what="row axis")] = (
        CP_ROW_AXIS if cp_layout() == "2d" else CP_AXIS
    )
    return PartitionSpec(*entries)


def single_spec(ndim: int, *, token_axis: int = -2) -> PartitionSpec:
    """Spec for a per-token representation on the pair-row mesh axis."""

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
    """Constrain a pair tensor to the active layout; identity outside CP."""

    mesh = cp_mesh()
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(
        x,
        NamedSharding(
            mesh,
            pair_spec(x.ndim, row_axis=row_axis, col_axis=col_axis),
        ),
    )


def shard_single(x: jax.Array, *, token_axis: int = -2) -> jax.Array:
    """Constrain a per-token representation; identity outside CP."""

    mesh = cp_mesh()
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(
        x,
        NamedSharding(mesh, single_spec(x.ndim, token_axis=token_axis)),
    )


# --- ring primitives -------------------------------------------------------


def grid_axes() -> tuple[str, str]:
    """The ``(row, column)`` mesh-axis names of the two-dimensional layout."""

    if cp_layout() != "2d":
        raise RuntimeError("grid axes are only defined for the 2-D layout")
    return (CP_ROW_AXIS, CP_COL_AXIS)


def _flat(
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    side: int,
) -> list[tuple[int, int]]:
    """Row-major flatten of coordinate permutations for ``lax.ppermute``."""

    return [(a * side + b, c * side + d) for (a, b), (c, d) in pairs]


def transpose_perm(side: int) -> list[tuple[int, int]]:
    """``(i, j) -> (j, i)``: the grid transpose."""

    return _flat(
        [((i, j), (j, i)) for i in range(side) for j in range(side)],
        side,
    )


def row_skew_perm(side: int, *, sign: int = -1) -> list[tuple[int, int]]:
    """Cannon initial LHS alignment along mesh columns."""

    return _flat(
        [
            ((i, j), (i, (j + sign * i) % side))
            for i in range(side)
            for j in range(side)
        ],
        side,
    )


def col_skew_perm(side: int, *, sign: int = -1) -> list[tuple[int, int]]:
    """Cannon initial RHS alignment along mesh rows."""

    return _flat(
        [
            ((i, j), ((i + sign * j) % side, j))
            for i in range(side)
            for j in range(side)
        ],
        side,
    )


def ring_perm(side: int, *, axis: str, delta: int = 1) -> list[tuple[int, int]]:
    """One ring hop along one axis of the two-dimensional grid."""

    if axis == CP_ROW_AXIS:
        pairs = [
            ((i, j), ((i + delta) % side, j)) for i in range(side) for j in range(side)
        ]
    elif axis == CP_COL_AXIS:
        pairs = [
            ((i, j), (i, (j + delta) % side)) for i in range(side) for j in range(side)
        ]
    else:
        raise ValueError(f"unknown grid axis: {axis!r}")
    return _flat(pairs, side)


def permute(x: jax.Array, perm: Sequence[Any]) -> jax.Array:
    """Apply a grid permutation inside a ``shard_map`` body."""

    return jax.lax.ppermute(x, axis_name=grid_axes(), perm=list(perm))


# --- input placement -------------------------------------------------------


def _is_movable_array(value: Any) -> bool:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return False
    kind = getattr(value.dtype, "kind", None)
    return kind not in ("U", "S", "O")


def _pair_axes_for_feature(name: str, value: Any) -> tuple[int, int] | None:
    """Infer the two square token axes for a whitelisted feature."""

    if name not in PAIR_FEATURE_NAMES or not _is_movable_array(value):
        return None
    shape = tuple(int(size) for size in value.shape)
    if len(shape) >= 3 and shape[-3] == shape[-2] and shape[-3] > 0:
        return (-3, -2)
    if len(shape) >= 2 and shape[-2] == shape[-1] and shape[-2] > 0:
        return (-2, -1)
    return None


def _axes_divide_mesh(
    shape: tuple[int, ...],
    row_axis: int,
    col_axis: int,
) -> bool:
    row = _resolve_axis(row_axis, len(shape), what="row axis")
    col = _resolve_axis(col_axis, len(shape), what="column axis")
    rows, cols = cp_grid()
    if shape[row] % rows:
        return False
    if cp_layout() == "2d" and shape[col] % cols:
        return False
    return True


def feature_spec(name: str, value: Any) -> PartitionSpec | None:
    """Return a safe entry sharding for one known input feature.

    Uneven inputs stay replicated.  Model-level padding may later create a
    divisible pair state, but entry placement must never ask JAX to construct
    unequal device buffers.
    """

    if cp_mesh() is None:
        return None
    axes = _pair_axes_for_feature(name, value)
    if axes is None:
        return None
    shape = tuple(int(size) for size in value.shape)
    if not _axes_divide_mesh(shape, *axes):
        return None
    return pair_spec(value.ndim, row_axis=axes[0], col_axis=axes[1])


def _replicated_sharding() -> NamedSharding:
    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("no context-parallel mesh is active")
    return NamedSharding(mesh, PartitionSpec())


def _place_leaf(value: Any, *, spec: PartitionSpec | None = None) -> Any:
    if not _is_movable_array(value):
        return value
    mesh = cp_mesh()
    if mesh is None:
        return value
    return jax.device_put(
        value,
        NamedSharding(mesh, spec) if spec is not None else _replicated_sharding(),
    )


def _rebuild_mapping(original: Mapping[Any, Any], values: dict[Any, Any]) -> Any:
    if type(original) is dict:
        return values
    try:
        return type(original)(values)
    except (TypeError, ValueError):
        return values


def _place_tree(node: Any, *, shard_pair_features: bool) -> Any:
    if isinstance(node, Mapping):
        placed: dict[Any, Any] = {}
        for key, value in node.items():
            spec = (
                feature_spec(str(key), value)
                if shard_pair_features and isinstance(key, str)
                else None
            )
            if spec is not None:
                placed[key] = _place_leaf(value, spec=spec)
            else:
                placed[key] = _place_tree(
                    value,
                    shard_pair_features=shard_pair_features,
                )
        return _rebuild_mapping(node, placed)

    if isinstance(node, tuple) and hasattr(node, "_fields"):
        return type(node)(
            *(
                _place_tree(value, shard_pair_features=shard_pair_features)
                for value in node
            )
        )
    if isinstance(node, tuple):
        return tuple(
            _place_tree(value, shard_pair_features=shard_pair_features)
            for value in node
        )
    if isinstance(node, list):
        return [
            _place_tree(value, shard_pair_features=shard_pair_features)
            for value in node
        ]

    if _is_movable_array(node):
        return _place_leaf(node)

    # Preserve registered custom pytrees while retaining the historical
    # "replicate every numeric leaf" behaviour for model parameter containers.
    try:
        leaves, treedef = jax.tree.flatten(node)
    except TypeError:
        return node
    if len(leaves) == 1 and leaves[0] is node:
        return node
    return jax.tree.unflatten(
        treedef,
        [_place_leaf(leaf) for leaf in leaves],
    )


def replicate_tree(tree: Any, *, shard_pair_features: bool = True) -> Any:
    """Place a pytree on the active mesh.

    Parameters and all unrecognised inputs are replicated, preserving the old
    contract.  Exact, whitelisted pair features are placed directly on the
    pair mesh when their axes divide evenly.  Atom and single-stream features
    remain replicated; this function never invents an atom-window CP contract.

    Pass ``shard_pair_features=False`` for strict historical replication.
    Identity when no mesh is active.
    """

    if cp_mesh() is None:
        return tree
    return _place_tree(tree, shard_pair_features=shard_pair_features)
