"""Context-parallel adapters for the Protenix diffusion atom graph.

The pair trunk has been sharded since Fold-CP landed; the diffusion atom
stream was not.  Under a mesh every device held the whole atom graph: the
``[n_windows, n_queries, n_keys, c]`` atom-pair cache, both atom transformer
stacks, and -- the expensive one -- the whole projected token-pair tensor,
because ``z_token[..., idx_q, idx_k, :]`` is a gather whose operand the SPMD
partitioner can only satisfy by collecting it.

This module distributes that graph over the pair-row mesh axis using the
shared atom-window primitives in :mod:`foldjax.models._cp_atom`, and supplies
the three things Protenix needs that Boltz-2's consumer does not:

* Protenix' own scatter-mean arithmetic. The shared helper divides by
  ``counts + eps``; Protenix divides by ``max(counts, 1)`` and accumulates in
  the activation dtype rather than promoting to FP32. Reusing the shared
  helper would have been a quiet arithmetic change inside a parity tolerance.
* windows over an atom axis at ``-2`` with a leading axis that is the
  diffusion sample count, not Boltz-2's fixed batch.
* the sparse token-pair gather fed with all-valid indices, because Protenix
  zero-*pads* its ``atom_to_token_idx`` trunks rather than masking them: the
  padded key positions legitimately read ``z_token[..., 0, 0, :]``, and the
  block-local attention mask is what makes their bias irrelevant. Masking them
  to zero here would compute different numbers from the serial path.

Nothing below runs, or is even imported into a decision, without an active
mesh: every entry point is guarded by ``cp_mesh() is not None`` at its caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import cp_mesh, cp_row_shards

# The alignment decision is not Protenix-specific: OpenDDE diffuses over this
# same atom graph and only names a different padding axis, so the predicate,
# the resolver and the adapter guard live beside the halo primitive whose
# requirement they state. They stay exported from here because that is where
# this port's consumers -- and, until the move, OpenDDE -- import them from.
from foldjax.models._cp_atom import (
    atom_axis_name,
    atom_spec,
    atom_window_misalignment,
    gather_token_pairs_to_atom_windows_cp,
    gather_tokens_to_atoms_cp,
    require_atom_windows,
    resolve_atom_windows,
    window_spec,
)
from foldjax.models.protenix.models.diffusion.transformer import (
    DiffusionTransformerStackParams,
    diffusion_transformer_stack,
)
from foldjax.models.protenix.models.primitives.atom_windows_cp import (
    AtomWindowPlan,
    use_atom_window_plan,
)
from foldjax.models.protenix.models.primitives.primitives import LinearParams, linear


def global_window_mask(
    n_atom: int,
    *,
    n_queries: int,
    n_keys: int,
) -> jnp.ndarray:
    """The ``[n_windows, n_queries, n_keys]`` validity mask, globally.

    The same expression ``_local_qk_trunks`` computes, kept whole so a device
    reads its own windows out of a window-sharded array instead of deriving
    them from its rank. Integer-only and therefore a compile-time constant.
    """

    if n_atom % n_queries:
        raise ValueError(
            f"{n_atom} atoms is not a multiple of the query window {n_queries}"
        )
    n_windows = n_atom // n_queries
    pad_left = (n_keys - n_queries) // 2
    q_abs = jnp.arange(n_windows * n_queries).reshape(n_windows, n_queries)
    k_abs = (
        jnp.arange(n_keys)[None, :]
        + jnp.arange(n_windows)[:, None] * n_queries
        - pad_left
    )
    return (
        (q_abs[..., None] < n_atom)
        & (k_abs[:, None, :] >= 0)
        & (k_abs[:, None, :] < n_atom)
    )


def _flat_leading(x: jnp.ndarray, trailing: int) -> tuple[jnp.ndarray, tuple[int, ...]]:
    """Collapse every axis before the last ``trailing`` into one."""

    leading = tuple(int(size) for size in x.shape[:-trailing])
    return x.reshape((-1,) + x.shape[-trailing:]), leading


def broadcast_token_to_atom_cp(
    x_token: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
) -> jnp.ndarray:
    """``jnp.take(x_token, idx, axis=-2)`` with both axes on CP rows.

    Bit-identical to the serial gather: exactly one CP row owns each token, so
    the ring accumulates one real contribution and zeros.
    """

    if cp_mesh() is None:
        raise RuntimeError("broadcast_token_to_atom_cp requires an active mesh")
    flat, leading = _flat_leading(jnp.asarray(x_token), 2)
    index = jnp.broadcast_to(
        jnp.asarray(atom_to_token_idx).reshape(1, -1),
        (flat.shape[0], int(atom_to_token_idx.shape[-1])),
    )
    gathered = gather_tokens_to_atoms_cp(
        flat,
        index,
        jnp.ones(index.shape, dtype=bool),
    )
    return gathered.reshape(leading + gathered.shape[-2:])


def aggregate_atom_to_token_cp(
    x_atom: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    *,
    n_token: int,
    atom_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Protenix' masked atom->token mean with a CP-row reduce-scatter.

    The mean is ``sum / max(count, 1)`` in the activation dtype -- Protenix'
    formula, not the shared helper's ``sum / (count + eps)`` in FP32. Only the
    order of the summation differs from the serial path, because each device
    scatters its own atoms into a full token buffer and ``psum_scatter``
    reduces those buffers into the row-owned token slice.
    """

    if cp_mesh() is None:
        raise RuntimeError("aggregate_atom_to_token_cp requires an active mesh")
    rows = cp_row_shards()
    if n_token % rows:
        raise ValueError(f"{n_token} tokens do not divide {rows} CP rows")
    if int(x_atom.shape[-2]) % rows:
        raise ValueError(f"{x_atom.shape[-2]} atoms do not divide {rows} CP rows")
    if atom_mask is not None and tuple(atom_mask.shape) != tuple(
        atom_to_token_idx.shape
    ):
        raise ValueError("atom_mask must share shape [N_atom] with atom mapping")
    axis_name = atom_axis_name()

    def local(values, index, weights=None):
        channels = values.shape[-1]
        out = jnp.zeros(values.shape[:-2] + (n_token, channels), dtype=values.dtype)
        counts = jnp.zeros((n_token,), dtype=values.dtype)
        if weights is None:
            out = out.at[..., index, :].add(values)
            counts = counts.at[index].add(jnp.ones((), dtype=values.dtype))
        else:
            shaped = weights.astype(values.dtype).reshape(
                (1,) * (values.ndim - 2) + (-1, 1)
            )
            out = out.at[..., index, :].add(values * shaped)
            counts = counts.at[index].add(weights.astype(values.dtype))
        out = jax.lax.psum_scatter(
            out,
            axis_name,
            scatter_dimension=out.ndim - 2,
            tiled=True,
        )
        counts = jax.lax.psum_scatter(
            counts,
            axis_name,
            scatter_dimension=0,
            tiled=True,
        )
        return out / jnp.maximum(counts[..., None], 1.0)

    operands: list[Any] = [x_atom, atom_to_token_idx]
    in_specs: list[PartitionSpec] = [
        atom_spec(x_atom.ndim, atom_axis=-2),
        atom_spec(1, atom_axis=0),
    ]
    if atom_mask is not None:
        operands.append(atom_mask)
        in_specs.append(atom_spec(1, atom_axis=0))
    return jax.shard_map(
        local,
        mesh=cp_mesh(),
        in_specs=tuple(in_specs),
        out_specs=atom_spec(x_atom.ndim, atom_axis=-2),
    )(*operands)


def broadcast_token_to_local_atom_pair_cp(
    z_token: jnp.ndarray,
    idx_q: jnp.ndarray,
    idx_k: jnp.ndarray,
) -> jnp.ndarray:
    """Gather ``z_token[..., idx_q, idx_k, :]`` without collecting ``z_token``.

    ``idx_q``/``idx_k`` are the zero-padded trunks of ``atom_to_token_idx``, so
    they are handed in with all-true validity: a padded key position reads
    token 0's pair value in the serial path too, and the block-local attention
    mask is what removes it from the softmax.
    """

    if cp_mesh() is None:
        raise RuntimeError("the local atom-pair gather requires an active mesh")
    flat, leading = _flat_leading(jnp.asarray(z_token), 3)
    batch = flat.shape[0]
    query = jnp.broadcast_to(idx_q[None], (batch,) + idx_q.shape)
    key = jnp.broadcast_to(idx_k[None], (batch,) + idx_k.shape)
    gathered = gather_token_pairs_to_atom_windows_cp(
        flat,
        query,
        jnp.ones(query.shape, dtype=bool),
        key,
        jnp.ones(key.shape, dtype=bool),
    )
    return gathered.reshape(leading + gathered.shape[-4:])


def _plan(n_queries: int, n_keys: int, window_mask: jnp.ndarray) -> AtomWindowPlan:
    return AtomWindowPlan(
        n_queries=n_queries,
        n_keys=n_keys,
        axis_name=atom_axis_name(),
        axis_size=cp_row_shards(),
        window_mask=window_mask,
    )


def atom_pair_conditioning_cp(
    p_lm: jnp.ndarray,
    c_l: jnp.ndarray,
    linear_cl: LinearParams,
    linear_cm: LinearParams,
    *,
    small_mlp: Any | None,
    n_queries: int,
    n_keys: int,
    window_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Condition the atom-pair cache on the atom single stream, shard-local.

    The key side of the conditioning reaches ``(n_keys - n_queries) // 2``
    atoms past each shard, so it is the halo exchange rather than a
    constraint. The small MLP rides along: it is channel-wise on the same
    window-sharded cache, and keeping it here means the conditioned cache is
    never assembled anywhere.
    """

    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("atom_pair_conditioning_cp requires an active mesh")
    from foldjax.models.protenix.models.diffusion.atom import atom_pair_small_mlp

    def local(p_l, c_l_local, window_mask_l):
        plan = _plan(n_queries, n_keys, window_mask_l)
        c_l_q, c_l_k, _, _ = plan.qk_trunks(
            c_l_local,
            c_l_local,
            n_queries=n_queries,
            n_keys=n_keys,
        )
        p_l = (
            p_l
            + linear(jax.nn.relu(c_l_q[..., None, :]), linear_cl)
            + linear(jax.nn.relu(c_l_k[..., None, :, :]), linear_cm)
        )
        if small_mlp is not None:
            p_l = p_l + atom_pair_small_mlp(p_l, small_mlp)
        return p_l

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            window_spec(p_lm.ndim, window_axis=-4),
            atom_spec(c_l.ndim, atom_axis=-2),
            window_spec(window_mask.ndim, window_axis=-3),
        ),
        out_specs=window_spec(p_lm.ndim, window_axis=-4),
    )(p_lm, c_l, window_mask)


def atom_transformer_stack_cp(
    q: jnp.ndarray,
    c: jnp.ndarray,
    p: jnp.ndarray,
    params: DiffusionTransformerStackParams,
    *,
    num_heads: int,
    n_queries: int,
    n_keys: int,
    use_scan: bool,
    attention_backend: str,
    glu_backend: str,
    atom_mask: jnp.ndarray | None,
    window_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Run the atom transformer over CP-row-owned query windows.

    ``params`` is closed over rather than passed as a ``shard_map`` operand:
    Protenix' parameter NamedTuples carry Python ``bool`` configuration
    (``has_s``, ``cross_attention_mode``) that a flattened operand would turn
    into a tracer, and the first ``if`` reading it would fail. Closure keeps
    them Python values and the arrays replicated, which is what they already
    are.
    """

    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("atom_transformer_stack_cp requires an active mesh")

    def local(q_l, c_l, p_l, window_mask_l, mask_l=None):
        with use_atom_window_plan(_plan(n_queries, n_keys, window_mask_l)):
            return diffusion_transformer_stack(
                q_l,
                c_l,
                p_l,
                params,
                num_heads=num_heads,
                n_queries=n_queries,
                n_keys=n_keys,
                use_scan=use_scan,
                attention_backend=attention_backend,
                glu_backend=glu_backend,
                sequence_mask=mask_l,
            )

    operands: list[Any] = [q, c, p, window_mask]
    in_specs: list[PartitionSpec] = [
        atom_spec(q.ndim, atom_axis=-2),
        atom_spec(c.ndim, atom_axis=-2),
        window_spec(p.ndim, window_axis=-4),
        window_spec(window_mask.ndim, window_axis=-3),
    ]
    if atom_mask is not None:
        operands.append(atom_mask)
        in_specs.append(atom_spec(1, atom_axis=0))
    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=tuple(in_specs),
        out_specs=atom_spec(q.ndim, atom_axis=-2),
    )(*operands)


__all__: Sequence[str] = (
    "aggregate_atom_to_token_cp",
    "atom_pair_conditioning_cp",
    "atom_transformer_stack_cp",
    "atom_window_misalignment",
    "broadcast_token_to_atom_cp",
    "broadcast_token_to_local_atom_pair_cp",
    "global_window_mask",
    "require_atom_windows",
    "resolve_atom_windows",
)
