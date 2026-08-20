"""Atom-window context parallelism for AF3-family diffusion modules.

Fold-CP's pair trunk is only half of the memory story.  Atom diffusion owns a
linear atom stream, local key windows and sparse token/pair gathers.  This file
implements those operations with explicit JAX collectives:

* atom/query windows are sharded on the pair-row mesh axis and replicated on
  the pair-column axis;
* neighbouring half-windows are exchanged with a fixed-width halo;
* token-to-atom gathers rotate the linear source shards rather than gathering
  the full source on every device;
* atom-to-token means use ``psum_scatter``;
* token-pair values are looked up by rotating 2-D pair-row tiles and reducing
  only the requested atom-window result over pair columns.

No operation below materialises a full pair representation on one device.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from foldjax.models._cp import (
    CP_AXIS,
    CP_COL_AXIS,
    CP_ROW_AXIS,
    cp_grid,
    cp_layout,
    cp_mesh,
    cp_row_shards,
    pair_spec,
    permute,
    transpose_perm,
)


def _resolve_axis(axis: int, ndim: int, *, name: str) -> int:
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise ValueError(f"{name} {axis} is out of range for rank {ndim}")
    return resolved


def atom_axis_name() -> str:
    """Mesh axis owning atom/query windows in the active CP layout."""

    layout = cp_layout()
    if layout == "2d":
        return CP_ROW_AXIS
    if layout == "1d":
        return CP_AXIS
    raise RuntimeError("atom context parallelism requires an active CP mesh")


def atom_spec(ndim: int, *, atom_axis: int = -2) -> PartitionSpec:
    """Partition one atom axis over CP rows and replicate every other axis."""

    entries: list[str | None] = [None] * ndim
    entries[_resolve_axis(atom_axis, ndim, name="atom axis")] = atom_axis_name()
    return PartitionSpec(*entries)


def window_spec(ndim: int, *, window_axis: int = -3) -> PartitionSpec:
    """Partition a query-window axis over CP rows."""

    entries: list[str | None] = [None] * ndim
    entries[_resolve_axis(window_axis, ndim, name="window axis")] = atom_axis_name()
    return PartitionSpec(*entries)


def shard_atoms(array: jax.Array, *, atom_axis: int = -2) -> jax.Array:
    """Constrain an atom stream to CP-row ownership; identity outside CP."""

    mesh = cp_mesh()
    if mesh is None:
        return array
    return jax.lax.with_sharding_constraint(
        array,
        NamedSharding(mesh, atom_spec(array.ndim, atom_axis=atom_axis)),
    )


def place_atoms(array: jax.Array, *, atom_axis: int = -2) -> jax.Array:
    """Place a host/replicated atom array directly on CP-row shards.

    Unlike :func:`shard_atoms`, which constrains an array inside a traced graph,
    this is an entry-placement operation. It is used for precomputed diffusion
    noise tapes so a large ``[steps, samples, atoms, 3]`` input is never first
    copied in full to every device.
    """

    mesh = cp_mesh()
    if mesh is None:
        return array
    resolved = _resolve_axis(atom_axis, array.ndim, name="atom axis")
    if array.shape[resolved] % cp_row_shards():
        raise ValueError(
            f"atom axis {array.shape[resolved]} is not divisible by "
            f"{cp_row_shards()} CP row shards"
        )
    return jax.device_put(
        array,
        NamedSharding(mesh, atom_spec(array.ndim, atom_axis=atom_axis)),
    )


def replicate_atoms(array: jax.Array) -> jax.Array:
    """Replicate a linear atom result explicitly at a post-CP boundary."""

    mesh = cp_mesh()
    if mesh is None:
        return array
    return jax.lax.with_sharding_constraint(
        array,
        NamedSharding(mesh, PartitionSpec()),
    )


def shard_windows(array: jax.Array, *, window_axis: int = -3) -> jax.Array:
    """Constrain a window-batched tensor to CP-row ownership."""

    mesh = cp_mesh()
    if mesh is None:
        return array
    return jax.lax.with_sharding_constraint(
        array,
        NamedSharding(mesh, window_spec(array.ndim, window_axis=window_axis)),
    )


def _ring_permutation(size: int, delta: int) -> list[tuple[int, int]]:
    return [(source, (source + delta) % size) for source in range(size)]


def _dense_single_to_keys(
    single: jax.Array,
    *,
    query_window: int,
    key_window: int,
) -> jax.Array:
    if single.ndim != 3:
        raise ValueError(f"single_to_keys expects [B, N, C], got {single.shape}")
    if query_window <= 0 or query_window % 2:
        raise ValueError("query_window must be positive and even")
    half = query_window // 2
    if key_window <= 0 or key_window % half:
        raise ValueError("key_window must be divisible by query_window // 2")
    n_atoms = single.shape[1]
    if n_atoms % query_window:
        raise ValueError(
            f"atom count {n_atoms} is not divisible by query window {query_window}"
        )
    n_windows = n_atoms // query_window
    n_half_windows = 2 * n_windows
    half_windows_per_key = key_window // half
    if half_windows_per_key % 2:
        raise ValueError("key window must contain an even number of half-windows")

    halves = single.reshape(
        single.shape[0],
        n_half_windows,
        half,
        single.shape[-1],
    )
    starts = 2 * jnp.arange(n_windows) + 1 - half_windows_per_key // 2
    indices = starts[:, None] + jnp.arange(half_windows_per_key)[None, :]
    valid = (indices >= 0) & (indices < n_half_windows)
    indices = jnp.clip(indices, 0, n_half_windows - 1)
    gathered = jnp.take(halves, indices, axis=1)
    gathered = gathered * valid[None, :, :, None, None].astype(gathered.dtype)
    return gathered.reshape(
        single.shape[0],
        n_windows,
        key_window,
        single.shape[-1],
    )


def single_to_keys_local(
    single: jax.Array,
    *,
    query_window: int,
    key_window: int,
    axis_name: str,
    axis_size: int,
) -> jax.Array:
    """Build local key windows from one atom shard and two fixed halos."""

    batch, local_atoms, channels = single.shape
    if local_atoms % query_window:
        raise ValueError(
            f"local atom count {local_atoms} is not divisible by {query_window}"
        )
    half = query_window // 2
    half_windows_per_key = key_window // half
    radius = half_windows_per_key // 2 - 1
    local_windows = local_atoms // query_window
    local_half_windows = 2 * local_windows
    halves = single.reshape(batch, local_half_windows, half, channels)

    if radius:
        if local_half_windows < radius:
            raise ValueError(
                "each CP row must own at least the atom halo radius: "
                f"local_half_windows={local_half_windows}, radius={radius}"
            )
        rank = jax.lax.axis_index(axis_name)
        left = jax.lax.ppermute(
            halves[:, -radius:],
            axis_name=axis_name,
            perm=_ring_permutation(axis_size, +1),
        )
        right = jax.lax.ppermute(
            halves[:, :radius],
            axis_name=axis_name,
            perm=_ring_permutation(axis_size, -1),
        )
        left = jnp.where(rank > 0, left, jnp.zeros_like(left))
        right = jnp.where(rank + 1 < axis_size, right, jnp.zeros_like(right))
        extended = jnp.concatenate((left, halves, right), axis=1)
    else:
        extended = halves

    # With radius = h/2 - 1, local query i always begins at 2*i in the
    # halo-extended half-window sequence.
    indices = 2 * jnp.arange(local_windows)[:, None]
    indices = indices + jnp.arange(half_windows_per_key)[None, :]
    gathered = jnp.take(extended, indices, axis=1)
    return gathered.reshape(batch, local_windows, key_window, channels)


def single_to_keys_cp(
    single: jax.Array,
    *,
    query_window: int,
    key_window: int,
) -> jax.Array:
    """Convert a sharded atom stream into sharded local key windows.

    Outside CP this is the dense reference operation.  Under CP the logical
    atom axis and resulting query-window axis are both split over CP rows.
    """

    mesh = cp_mesh()
    if mesh is None:
        return _dense_single_to_keys(
            single,
            query_window=query_window,
            key_window=key_window,
        )
    if single.ndim != 3:
        raise ValueError(f"single_to_keys_cp expects [B, N, C], got {single.shape}")
    n_atoms = single.shape[1]
    if n_atoms % query_window:
        raise ValueError(
            f"atom count {n_atoms} is not divisible by query window {query_window}"
        )
    n_windows = n_atoms // query_window
    row_shards = cp_row_shards()
    if n_windows % row_shards:
        raise ValueError(
            "atom query-window count must divide CP rows; pad atoms to a "
            f"multiple of query_window * cp_rows ({query_window * row_shards})"
        )
    axis_name = atom_axis_name()

    def local(single_local):
        return single_to_keys_local(
            single_local,
            query_window=query_window,
            key_window=key_window,
            axis_name=axis_name,
            axis_size=row_shards,
        )

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=atom_spec(single.ndim, atom_axis=1),
        out_specs=window_spec(4, window_axis=1),
    )(single)


def _ring_gather_local(
    source: jax.Array,
    indices: jax.Array,
    valid: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
) -> jax.Array:
    """Gather arbitrary global indices while rotating equal source shards."""

    source_size = source.shape[1]
    owner = jax.lax.axis_index(axis_name)
    result = jnp.zeros(indices.shape + (source.shape[-1],), dtype=source.dtype)
    source_work = source
    for step in range(axis_size):
        local_index = indices - owner * source_size
        owned = (local_index >= 0) & (local_index < source_size) & valid
        clipped = jnp.clip(local_index, 0, source_size - 1)
        gathered = jnp.take_along_axis(source_work, clipped[..., None], axis=1)
        result = result + gathered * owned[..., None].astype(gathered.dtype)
        if step + 1 < axis_size:
            source_work = jax.lax.ppermute(
                source_work,
                axis_name=axis_name,
                perm=_ring_permutation(axis_size, +1),
            )
            owner = (owner - 1) % axis_size
    return result


def gather_tokens_to_atoms_cp(
    token_values: jax.Array,
    token_indices: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Gather token values to CP-row-sharded atoms without a full-token gather."""

    mesh = cp_mesh()
    if mesh is None:
        gathered = jnp.take_along_axis(
            token_values,
            token_indices[..., None],
            axis=1,
        )
        return gathered * valid[..., None].astype(gathered.dtype)
    if token_values.ndim != 3 or token_indices.ndim != 2:
        raise ValueError("token gather expects [B,T,C] values and [B,A] indices")
    rows = cp_row_shards()
    if token_values.shape[1] % rows or token_indices.shape[1] % rows:
        raise ValueError("token and atom axes must divide CP rows")
    axis_name = atom_axis_name()

    def local(values_local, indices_local, valid_local):
        return _ring_gather_local(
            values_local,
            indices_local,
            valid_local,
            axis_name=axis_name,
            axis_size=rows,
        )

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            atom_spec(3, atom_axis=1),
            atom_spec(2, atom_axis=1),
            atom_spec(2, atom_axis=1),
        ),
        out_specs=atom_spec(3, atom_axis=1),
    )(token_values, token_indices, valid)


def gather_atoms_to_tokens_cp(
    atom_values: jax.Array,
    atom_indices: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Gather representative atoms to CP-row-sharded tokens."""

    mesh = cp_mesh()
    if mesh is None:
        gathered = jnp.take_along_axis(
            atom_values,
            atom_indices[..., None],
            axis=1,
        )
        return gathered * valid[..., None].astype(gathered.dtype)
    if atom_values.ndim != 3 or atom_indices.ndim != 2:
        raise ValueError("atom gather expects [B,A,C] values and [B,T] indices")
    rows = cp_row_shards()
    if atom_values.shape[1] % rows or atom_indices.shape[1] % rows:
        raise ValueError("atom and token axes must divide CP rows")
    axis_name = atom_axis_name()

    def local(values_local, indices_local, valid_local):
        return _ring_gather_local(
            values_local,
            indices_local,
            valid_local,
            axis_name=axis_name,
            axis_size=rows,
        )

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            atom_spec(3, atom_axis=1),
            atom_spec(2, atom_axis=1),
            atom_spec(2, atom_axis=1),
        ),
        out_specs=atom_spec(3, atom_axis=1),
    )(atom_values, atom_indices, valid)


def scatter_atoms_to_tokens_mean_cp(
    atom_values: jax.Array,
    token_indices: jax.Array,
    valid: jax.Array,
    *,
    num_tokens: int,
    eps: float = 1e-6,
) -> jax.Array:
    """Scatter atom values to token means with a fused CP-row reduce-scatter."""

    mesh = cp_mesh()
    if mesh is None:

        def one(indices_b, valid_b, values_b):
            values_b = values_b * valid_b[:, None].astype(values_b.dtype)
            sums = jnp.zeros((num_tokens, values_b.shape[-1]), dtype=values_b.dtype)
            counts = jnp.zeros((num_tokens,), dtype=values_b.dtype)
            sums = sums.at[indices_b].add(values_b)
            counts = counts.at[indices_b].add(valid_b.astype(values_b.dtype))
            return sums / (counts[:, None] + eps)

        return jax.vmap(one)(token_indices, valid, atom_values)
    rows = cp_row_shards()
    if num_tokens % rows or atom_values.shape[1] % rows:
        raise ValueError("atom and output token axes must divide CP rows")
    axis_name = atom_axis_name()

    def local(values_local, indices_local, valid_local):
        def one(indices_b, valid_b, values_b):
            values_b = values_b * valid_b[:, None].astype(values_b.dtype)
            sums = jnp.zeros((num_tokens, values_b.shape[-1]), dtype=values_b.dtype)
            counts = jnp.zeros((num_tokens,), dtype=values_b.dtype)
            sums = sums.at[indices_b].add(values_b)
            counts = counts.at[indices_b].add(valid_b.astype(values_b.dtype))
            return sums, counts

        sums, counts = jax.vmap(one)(indices_local, valid_local, values_local)
        sums = jax.lax.psum_scatter(
            sums,
            axis_name,
            scatter_dimension=1,
            tiled=True,
        )
        counts = jax.lax.psum_scatter(
            counts,
            axis_name,
            scatter_dimension=1,
            tiled=True,
        )
        return sums / (counts[..., None] + eps)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            atom_spec(3, atom_axis=1),
            atom_spec(2, atom_axis=1),
            atom_spec(2, atom_axis=1),
        ),
        out_specs=atom_spec(3, atom_axis=1),
    )(atom_values, token_indices, valid)


def gather_token_pairs_to_atom_windows_cp(
    pair_values: jax.Array,
    query_indices: jax.Array,
    query_valid: jax.Array,
    key_indices: jax.Array,
    key_valid: jax.Array,
) -> jax.Array:
    """Gather sparse token-pair values for local atom query/key windows.

    Pair-row tiles rotate over CP rows.  Each pair-column rank contributes only
    keys owned by its local column tile; a final column ``psum`` replicates the
    requested atom-window result without reconstructing the token-pair tensor.
    """

    if cp_mesh() is None:
        batch = pair_values.shape[0]
        batch_index = jnp.arange(batch)[:, None, None, None]
        values = pair_values[
            batch_index,
            query_indices[:, :, :, None],
            key_indices[:, :, None, :],
        ]
        mask = query_valid[:, :, :, None] & key_valid[:, :, None, :]
        return values * mask[..., None].astype(values.dtype)
    if pair_values.ndim != 4:
        raise ValueError(f"pair values must be [B,T,T,C], got {pair_values.shape}")
    if query_indices.ndim != 3 or key_indices.ndim != 3:
        raise ValueError("query/key atom-window indices must have rank three")
    if query_indices.shape[:2] != key_indices.shape[:2]:
        raise ValueError("query and key windows must share batch/window axes")
    rows, cols = cp_grid()
    n_tokens = pair_values.shape[1]
    n_windows = query_indices.shape[1]
    if pair_values.shape[2] != n_tokens:
        raise ValueError("token-pair representation must be square")
    if n_tokens % rows or n_tokens % cols or n_windows % rows:
        raise ValueError("pair token axes and atom windows must divide the CP grid")

    row_axis = atom_axis_name()
    query_spec = window_spec(3, window_axis=1)
    output_spec = window_spec(5, window_axis=1)
    pair_layout = pair_spec(4, row_axis=1, col_axis=2)

    def local(pair_local, q_local, q_valid_local, k_local, k_valid_local):
        row_tile = pair_local.shape[1]
        col_tile = pair_local.shape[2]
        row_owner = jax.lax.axis_index(row_axis)
        col_owner = (
            jax.lax.axis_index(CP_COL_AXIS)
            if cp_layout() == "2d"
            else jnp.asarray(0, dtype=jnp.int32)
        )
        result = jnp.zeros(
            q_local.shape + (k_local.shape[-1], pair_local.shape[-1]),
            dtype=pair_local.dtype,
        )
        pair_work = pair_local
        batch_index = jnp.arange(pair_local.shape[0])[:, None, None, None]

        for step in range(rows):
            q_index = q_local - row_owner * row_tile
            k_index = k_local - col_owner * col_tile
            q_owned = (q_index >= 0) & (q_index < row_tile) & q_valid_local
            k_owned = (k_index >= 0) & (k_index < col_tile) & k_valid_local
            q_take = jnp.clip(q_index, 0, row_tile - 1)
            k_take = jnp.clip(k_index, 0, col_tile - 1)
            values = pair_work[
                batch_index,
                q_take[:, :, :, None],
                k_take[:, :, None, :],
            ]
            owned = q_owned[:, :, :, None] & k_owned[:, :, None, :]
            result = result + values * owned[..., None].astype(values.dtype)
            if step + 1 < rows:
                pair_work = jax.lax.ppermute(
                    pair_work,
                    axis_name=row_axis,
                    perm=_ring_permutation(rows, +1),
                )
                row_owner = (row_owner - 1) % rows

        if cp_layout() == "2d":
            result = jax.lax.psum(result, CP_COL_AXIS)
        return result

    return jax.shard_map(
        local,
        mesh=cp_mesh(),
        in_specs=(
            pair_layout,
            query_spec,
            query_spec,
            query_spec,
            query_spec,
        ),
        out_specs=output_spec,
    )(
        pair_values,
        query_indices,
        query_valid,
        key_indices,
        key_valid,
    )


def pair_bias_attention_2d(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    pair_bias: jax.Array,
    key_mask: jax.Array,
    *,
    scale: float,
    inf: float = 1e6,
) -> jax.Array:
    """Two-dimensional token attention with pair bias and no full-key gather.

    Q stays on pair rows.  A grid transpose changes row-sharded K/V/mask into
    pair-column ownership, then fp32 max/sum reductions over ``cp_col`` perform
    the distributed softmax.
    """

    if cp_layout() != "2d":
        raise RuntimeError("pair_bias_attention_2d requires a 2-D CP mesh")
    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("context-parallel mesh is not active")
    side, cols = cp_grid()
    if side != cols:
        raise ValueError("pair-bias attention requires a square mesh")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key and value shapes must match")
    if query.ndim != 4 or pair_bias.ndim != 4 or key_mask.ndim != 2:
        raise ValueError("expected Q/K/V [B,T,H,D], bias [B,H,T,T], mask [B,T]")

    qkv_spec = atom_spec(4, atom_axis=1)
    mask_spec = atom_spec(2, atom_axis=1)
    bias_spec = PartitionSpec(None, None, CP_ROW_AXIS, CP_COL_AXIS)

    def local(q_local, k_local, v_local, bias_local, mask_local):
        transpose = transpose_perm(side)
        k_local = permute(k_local, transpose)
        v_local = permute(v_local, transpose)
        mask_local = permute(mask_local, transpose)

        logits = jnp.einsum(
            "bqhd,bkhd->bhqk",
            q_local.astype(jnp.float32),
            k_local.astype(jnp.float32),
        )
        logits = logits * jnp.asarray(scale, dtype=jnp.float32)
        logits = logits + bias_local.astype(jnp.float32)
        logits = (
            logits + (1.0 - mask_local[:, None, None, :].astype(jnp.float32)) * -inf
        )

        local_maximum = jnp.max(logits, axis=-1, keepdims=True)
        maximum = jax.lax.pmax(local_maximum, CP_COL_AXIS)
        finite = jnp.isfinite(maximum)
        positive_infinity = jnp.isposinf(maximum)
        shifted = jnp.where(finite, logits - maximum, -jnp.inf)
        probabilities = jnp.where(
            positive_infinity,
            jnp.isposinf(logits).astype(jnp.float32),
            jnp.exp(shifted),
        )
        denominator = jax.lax.psum(
            jnp.sum(probabilities, axis=-1, keepdims=True),
            CP_COL_AXIS,
        )
        numerator = jax.lax.psum(
            jnp.einsum(
                "bhqk,bkhd->bqhd",
                probabilities,
                v_local.astype(jnp.float32),
            ),
            CP_COL_AXIS,
        )
        tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
        denominator_qh = jnp.swapaxes(denominator, 1, 2)
        out = jnp.where(
            denominator_qh > 0,
            numerator / jnp.maximum(denominator_qh, tiny),
            jnp.zeros_like(numerator),
        )
        return out.astype(value.dtype)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(qkv_spec, qkv_spec, qkv_spec, bias_spec, mask_spec),
        out_specs=qkv_spec,
    )(query, key, value, pair_bias, key_mask)
