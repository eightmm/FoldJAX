"""Pure JAX Transition forward pass."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import cp_grid, cp_mesh, pair_spec, shard_pair_rows
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives.glu_backend import (
    gated_linear_unit,
    reject_fused_glu_under_cp,
)
from foldjax.models.boltz2.models.primitives.native_amp_norm import amp_layer_norm

TransitionParams = Mapping[str, Mapping[str, jnp.ndarray]]


# The transition widens its input before narrowing it again, and the widened
# form is the largest buffer in Boltz-2's graph: the MSA transition alone is
# f32[1, 1024, 490, 512] -- 980 MiB of a 2,072 MiB arena on a 490-residue job,
# because that one call site was the only transition not given a row chunk.
#
# Deriving the block from the tensor rather than from the call site fixes every
# caller at once, and it is exact: each row of axis 1 is independent, so no
# reduction is split.
_WIDE_BUDGET_BYTES = 256 * 1024**2


def _auto_row_chunk(x: jnp.ndarray, params: TransitionParams) -> int | None:
    """Rows of axis 1 whose widened form fits the budget, or None to do it whole."""
    if x.ndim != 4 or x.shape[1] < 2:
        return None
    wide = params["fc1"]["kernel"].shape[-1] + params["fc2"]["kernel"].shape[-1]
    per_row = x.shape[2] * wide * x.dtype.itemsize
    if per_row <= 0 or per_row * x.shape[1] <= _WIDE_BUDGET_BYTES:
        return None
    return max(1, _WIDE_BUDGET_BYTES // per_row)


def transition_forward(
    params: TransitionParams,
    x: jnp.ndarray,
    chunk_size: int | None = None,
    eps: float = 1e-5,
    row_chunk_size: int | None = None,
    glu_backend: str = "xla",
    compute_dtype: jnp.dtype | None = None,
    native_amp_norm: bool = False,
    cp_pair: bool = False,
    cp_msa: bool = False,
) -> jnp.ndarray:
    """Run a Boltz Transition block using mapped PyTorch parameters.

    ``chunk_size`` chunks the hidden (SwiGLU) dimension; the fc2 accumulation
    keeps it bit-exact. ``row_chunk_size`` is an orthogonal outer-row chunk for
    any rank-4 input ``[B, N, ..., C]`` -- the pair tensor and, just as
    importantly, the MSA: it splits axis 1 into independent blocks and
    concatenates. No reduction is split, so it is mathematically exact, though
    not bit-identical under the default matmul precision, where XLA tiles the
    smaller GEMM differently: measured at 1.2e-4 relative, and 2.6e-7 under
    ``float32`` precision. Left as ``None`` it is chosen from the size of the
    widened form; pass ``0`` to force the single-op / hidden-chunk path.

    ``cp_pair`` declares ``x`` a pair tensor ``[B, N, N, C]`` laid out by the
    active context-parallel layout, which is what lets the row block survive
    context parallelism -- see ``_cp_pair_transition``.

    ``cp_msa`` declares ``x`` an MSA tensor ``[B, M, N, C]`` whose axis 1 is
    alignment depth. Under the 1-D layout no layout shards that axis -- the
    MSA carry measures ``PartitionSpec(None, None, "cp")`` -- so the row block
    is shard-aligned as written and needs no ``shard_map``. Under the 2-D
    layout the depth axis is split over the grid's column axis, and the MSA
    module calls this function on the local tile from inside its own
    ``shard_map`` (``trunk_blocks/msa.py``), where ``x`` is again a plain
    array whose axis 1 no mesh axis touches; ``cp_msa`` keeps meaning "block
    axis 1 as written" in both cases.

    An undeclared rank-4 caller keeps the whole local tile, because slicing the
    global row axis of a *sharded* tensor is what the partitioner cannot serve
    without a gather.

    ``glu_backend="tokamax"`` runs the swish GLU through the fused Triton kernel
    (GPU, low precision); ``"xla"`` (default) keeps the bit-exact split-matmul.
    """

    if compute_dtype is None and params["fc1"]["kernel"].dtype in (
        jnp.bfloat16,
        jnp.float16,
    ):
        compute_dtype = params["fc1"]["kernel"].dtype
    mesh = cp_mesh()
    if mesh is not None and cp_pair and x.ndim == 4:
        return _cp_pair_transition(
            params,
            x,
            mesh=mesh,
            chunk_size=chunk_size,
            eps=eps,
            row_chunk_size=row_chunk_size,
            glu_backend=glu_backend,
            compute_dtype=compute_dtype,
            native_amp_norm=native_amp_norm,
        )
    if mesh is not None and not cp_msa:
        # Under context parallelism axis 1 of a pair tensor is the sharded row
        # axis; slicing it block by block would fight the partitioner, and the
        # memory the row chunk exists to bound is already divided across
        # devices. An MSA tensor is the exception -- its axis 1 is alignment
        # depth, unsharded under 1-D and handed in as a local tile under 2-D
        # -- and it says so with ``cp_msa``, because a rank-4 shape alone
        # cannot tell the two apart.
        row_chunk_size = 0
    if row_chunk_size is None:
        row_chunk_size = _auto_row_chunk(x, params)
    return _transition_rows(
        params,
        x,
        chunk_size=chunk_size,
        eps=eps,
        row_chunk_size=row_chunk_size,
        glu_backend=glu_backend,
        compute_dtype=compute_dtype,
        native_amp_norm=native_amp_norm,
    )


def _cp_pair_transition(
    params: TransitionParams,
    x: jnp.ndarray,
    *,
    mesh: object,
    chunk_size: int | None,
    eps: float,
    row_chunk_size: int | None,
    glu_backend: str,
    compute_dtype: jnp.dtype | None,
    native_amp_norm: bool,
) -> jnp.ndarray:
    """Row-block the transition on each device's own tile of a pair tensor.

    Zeroing the block under context parallelism left the widened pre-gate form
    as the largest buffer in the sharded program: at 2,112 tokens on four
    devices it is ``[1, 1056, 1056, 1024]`` per device under the 2-D layout and
    ``[1, 528, 2112, 1024]`` under the 1-D one, together about half the arena,
    against ``[1, 64, 2112, 1024]`` in the serial program. Restoring the block
    on the *global* row axis does not recover that: a 64-row slice of a
    4-way-sharded axis is not shard-aligned, so the partitioner gathers and the
    arena moves 0.2%.

    The block therefore has to be taken inside the shard, the way
    ``triangle/triangle.py`` runs Cannon and ``triangle_attention.py`` runs the
    ring. Every op here is elementwise in the two token axes and contracts only
    over channels, so the local tile is the whole computation for its own rows
    and columns -- no collective, and the arithmetic per element is the
    arithmetic the unblocked sharded program did: outputs are bitwise identical
    to the unblocked context-parallel path in both layouts, in float32 and
    under the shipped bfloat16 compile policy.

    Measured at 2,112 tokens on four CPU devices: the full-tile widened form
    disappears from the partitioned program -- 15 values of
    ``f32[1,1056,1056,1024]`` (4,356 MiB each) under the 2-D layout and 15 of
    ``f32[1,528,2112,1024]`` under the 1-D one go to none -- and the widest
    that remains is ``f32[135168,1024]``, 528 MiB, where XLA merged two
    adjacent block dots back into one; the rest are single blocks
    (``f32[1,64,1056,1024]``, 264 MiB, 96 of them). The CPU temp arena barely
    moves with that (1-D -1.1%, 2-D +0.3%): that backend merges blocks where it
    likes and materialises the ones it keeps rather than fusing them, which is
    the same artefact that puts the serial arm at 92 GiB on CPU against
    18.5 GiB measured on GPU. The per-device shape is the part of a CPU
    partitioning probe that transfers; the arena is not.
    """

    rows, columns = cp_grid()
    n_rows, n_columns = x.shape[1], x.shape[2]
    # `shard_map` needs both sharded axes to divide the grid. The padded region
    # is its own set of rows and columns, and the transition never mixes them
    # with a kept one, so the padded output is sliced away unread.
    row_pad, column_pad = (-n_rows) % rows, (-n_columns) % columns
    if row_pad or column_pad:
        x = jnp.pad(x, ((0, 0), (0, row_pad), (0, column_pad), (0, 0)))
    spec = pair_spec(x.ndim)

    def local(x_local: jnp.ndarray, params_local: TransitionParams) -> jnp.ndarray:
        block = row_chunk_size
        if block is None:
            # The serial rule, read off the local tile rather than the global
            # one: the widened form it bounds is the per-device buffer.
            block = _auto_row_chunk(x_local, params_local)
        return _transition_rows(
            params_local,
            x_local,
            chunk_size=chunk_size,
            eps=eps,
            row_chunk_size=block,
            glu_backend=glu_backend,
            compute_dtype=compute_dtype,
            native_amp_norm=native_amp_norm,
        )

    # Parameters go in as a replicated operand rather than a closure, as in
    # `triangle_attention._triangle_attention_cp`: under a layer scan they are
    # traced values, and an operand keeps the whole tree on the mesh.
    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(spec, PartitionSpec()),
        out_specs=spec,
    )(x, params)
    if row_pad or column_pad:
        # Re-pinning the slice keeps the partitioner from answering the
        # narrower shape with a replicated result.
        out = shard_pair_rows(
            jax.lax.slice(
                out,
                (0, 0, 0, 0),
                (out.shape[0], n_rows, n_columns, out.shape[3]),
            )
        )
    return out


def _transition_rows(
    params: TransitionParams,
    x: jnp.ndarray,
    *,
    chunk_size: int | None,
    eps: float,
    row_chunk_size: int | None,
    glu_backend: str,
    compute_dtype: jnp.dtype | None,
    native_amp_norm: bool,
) -> jnp.ndarray:
    """The transition itself, over whatever rows of axis 1 it is handed.

    ``compute_dtype`` and ``row_chunk_size`` are already resolved, and nothing
    here reads the sharding: the caller decides whether the row block runs on
    the global axis or inside a shard.
    """

    if (
        row_chunk_size is not None
        and row_chunk_size > 0
        and x.ndim == 4
        and x.shape[1] > row_chunk_size
    ):
        n = x.shape[1]
        blocks = []
        for start in range(0, n, row_chunk_size):
            stop = min(start + row_chunk_size, n)
            blocks.append(
                _transition_rows(
                    params,
                    x[:, start:stop],
                    chunk_size=chunk_size,
                    eps=eps,
                    # Already split; 0 stops the slice from splitting again.
                    row_chunk_size=0,
                    glu_backend=glu_backend,
                    compute_dtype=compute_dtype,
                    native_amp_norm=native_amp_norm,
                )
            )
        return jnp.concatenate(blocks, axis=1)

    if compute_dtype is not None:
        # CUDA autocast keeps LayerNorm (including affine) in FP32, then
        # narrows inputs/weights at each Linear boundary.
        x = x.astype(jnp.float32)
    norm = (
        amp_layer_norm
        if native_amp_norm and compute_dtype == jnp.bfloat16
        else _layer_norm
    )
    x = norm(x, params["norm"]["scale"], params["norm"]["bias"], eps)
    fc1_kernel = params["fc1"]["kernel"]
    fc2_kernel = params["fc2"]["kernel"]
    fc3_kernel = params["fc3"]["kernel"]
    if compute_dtype is not None:
        x = x.astype(compute_dtype)
        fc1_kernel = fc1_kernel.astype(compute_dtype)
        fc2_kernel = fc2_kernel.astype(compute_dtype)
        fc3_kernel = fc3_kernel.astype(compute_dtype)

    def silu(value):
        if compute_dtype is not None:
            # Torch's SiLU is one op: do not introduce a bf16 sigmoid rounding
            # before the multiplication inside the activation.
            return jax.nn.silu(value.astype(jnp.float32)).astype(value.dtype)
        return jax.nn.silu(value)

    if glu_backend != "xla":
        reject_fused_glu_under_cp(glu_backend)
        hidden = gated_linear_unit(
            x, fc1_kernel, fc2_kernel, jax.nn.silu, backend=glu_backend
        )
        return hidden @ fc3_kernel

    if chunk_size is None:
        fc12 = x @ jnp.concatenate((fc1_kernel, fc2_kernel), axis=-1)
        fc1, fc2 = jnp.split(fc12, 2, axis=-1)
        hidden = silu(fc1) * fc2
        return hidden @ fc3_kernel

    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)

    out = jnp.zeros((*x.shape[:-1], fc3_kernel.shape[-1]), dtype=x.dtype)
    hidden_dim = fc3_kernel.shape[0]
    for start in range(0, hidden_dim, chunk_size):
        stop = min(start + chunk_size, hidden_dim)
        hidden = silu(x @ fc1_kernel[:, start:stop]) * (x @ fc2_kernel[:, start:stop])
        out = out + hidden @ fc3_kernel[start:stop, :]
    return out
