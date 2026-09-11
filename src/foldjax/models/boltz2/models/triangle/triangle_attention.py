"""Pure JAX triangle attention blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import (
    cp_layout,
    cp_mesh,
    cp_row_shards,
    pair_row_spec,
    shard_pair_rows,
)
from foldjax.models._cp_attention import ring_triangle_attention_2d
from foldjax.models.boltz2.models.primitives._common import (
    layer_norm as _shared_layer_norm,
)
from foldjax.models.boltz2.models.primitives._common import sigmoid as _sigmoid
from foldjax.models.boltz2.models.primitives.native_amp_norm import amp_layer_norm
from foldjax.models.boltz2.models.triangle.triangle import (
    resolve_native_amp as _resolve_native_amp,
)

TriangleAttentionParams = Mapping[
    str, Mapping[str, jnp.ndarray | Mapping[str, jnp.ndarray]]
]


def resolve_triangle_attention_chunk(
    num_tokens: int,
    chunk_size: int,
    triangle_attention_chunk: int | None = None,
) -> int:
    """Return the triangle-attention query chunk.

    Mirrors AF3's long-sequence pair-attention policy: use the general chunk
    size up to 1536 tokens, then reduce attention chunks to 32 for headroom.
    A caller-provided ``triangle_attention_chunk`` always wins.
    """

    if triangle_attention_chunk is not None:
        return triangle_attention_chunk
    return 32 if num_tokens > 1536 else chunk_size


def resolve_triangle_attention_q_chunk(
    num_tokens: int,
    triangle_attention_q_chunk: int | None = None,
) -> int | None:
    """Return the inner query-row chunk for triangle attention.

    The outer triangle chunk splits the independent triangle batch axis. For
    very long sequences that still leaves a dense ``N x N`` attention matrix
    inside each outer block. This optional chunk splits the query rows inside
    that matrix while keeping the full key axis for each softmax row, so it is
    mathematically equivalent and weight-compatible.
    """

    if triangle_attention_q_chunk is not None:
        return triangle_attention_q_chunk
    return 512 if num_tokens > 2048 else None


def resolve_matmul_precision(matmul_precision: str) -> jax.lax.Precision:
    """Map a matmul-precision string to a ``jax.lax.Precision``.

    ``"highest"`` -> ``Precision.HIGHEST`` (fp32 accumulation, the bit-exact
    default). ``"default"``/``"tensorfloat32"`` -> ``Precision.DEFAULT`` (TF32
    tensor-core accumulation on GPU). Callers selecting a relaxed precision
    should also set ``jax.config jax_default_matmul_precision`` to match so that
    the unpinned matmuls use the same path (the trunk entry does this).

    **There is deliberately no ``"high"``**, the spelling the neutral
    ``matmul_precision`` knob uses. This function is reached only by the
    op-level string, which `api.predict` does not set and which therefore
    stays at its signature default while the port's scope ships ``"high"``
    (see `api.MATMUL_PRECISION`). Accepting ``"high"`` here would make a
    future edit that wires the two together compile quietly; refusing it
    makes that edit raise on the first prediction instead. Adding it is part
    of the cost of unifying the two surfaces, not a tidy-up.
    """
    key = matmul_precision.lower()
    if key in ("highest", "float32", "fp32"):
        return jax.lax.Precision.HIGHEST
    if key in ("default", "tensorfloat32", "tf32"):
        return jax.lax.Precision.DEFAULT
    msg = f"Unsupported matmul_precision: {matmul_precision!r}"
    raise ValueError(msg)


def triangle_attention_forward(
    params: TriangleAttentionParams,
    x: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    starting: bool = True,
    inf: float = 1e9,
    eps: float = 1e-5,
    chunk_size: int = 128,
    q_chunk_size: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    native_amp: bool | None = None,
) -> jnp.ndarray:
    """Run Boltz TriangleAttention starting or ending node.

    ``triangle_backend``: ``"xla"`` (default, bit-exact dense/chunked path),
    ``"pallas"`` (opt-in GPU flash kernel), or ``"cueq"`` (Torch-compatible
    cuEquivariance CUDA kernel).

    ``native_amp`` answers the autocast-configuration question for the query
    scale only; ``None`` (default) infers it from the activation width. The
    entry LayerNorm deliberately keeps reading the width instead: upstream has
    its own BF16-input exception there (``boltz/model/layers/
    triangular_attention/primitives.py:138-147``), so a narrowed pair residual
    should take it.
    """

    precision = resolve_matmul_precision(matmul_precision)

    if cp_mesh() is not None:
        # The fused kernels are custom calls the SPMD partitioner cannot
        # split, and the blocked XLA row loop slices the sharded axis; the
        # context-parallel path runs the same row loop under `shard_map` on
        # each device's local rows instead. Same constraint as upstream
        # Fold-CP: plain kernels only in distributed mode.
        if triangle_backend not in ("xla", "cueq"):
            msg = (
                "context-parallel triangle attention supports "
                f"triangle_backend='xla' or 'cueq'; got {triangle_backend!r}."
            )
            raise ValueError(msg)
        return _triangle_attention_cp(
            params,
            x,
            mask,
            starting=starting,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
            q_chunk_size=q_chunk_size,
            precision=precision,
            local_backend=triangle_backend,
            native_amp=native_amp,
        )

    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)

    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    # Native AMP normalizes the FP32 pair residual before its BF16 projections.
    # CUDA Welford/FMA rounding matters at that next cast; leave pure FP32 and
    # the native custom BF16-input LayerNorm exception on their existing paths.
    norm = (
        amp_layer_norm
        if x.dtype == jnp.float32 and params["linear"]["kernel"].dtype == jnp.bfloat16
        else _layer_norm
    )
    x = norm(
        x,
        params["layer_norm"]["scale"],
        params["layer_norm"]["bias"],
        eps,
    )
    mask = mask[..., :, None, None, :]
    # Build the additive mask in fp32 so the large `inf` constant stays finite:
    # in fp16, `inf` (1e9) would saturate to +inf and a fully-masked softmax row
    # would become 0/0 -> NaN. fp32 runtime is unchanged (mask already fp32).
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)

    triangle_bias = _linear(x, params["linear"]["kernel"], precision)
    triangle_bias = jnp.transpose(triangle_bias, (0, 3, 1, 2))
    triangle_bias = jnp.expand_dims(triangle_bias, axis=1)

    x = _attention(
        params["mha"],
        q_x=x,
        kv_x=x,
        tri_bias=triangle_bias,
        mask_bias=mask_bias,
        chunk_size=chunk_size,
        q_chunk_size=q_chunk_size,
        precision=precision,
        triangle_backend=triangle_backend,
        native_amp=native_amp,
    )

    if not starting:
        x = jnp.swapaxes(x, -2, -3)
    return x


def _triangle_attention_cp(
    params: TriangleAttentionParams,
    x: jnp.ndarray,
    mask: jnp.ndarray | None,
    *,
    starting: bool,
    inf: float,
    eps: float,
    chunk_size: int,
    q_chunk_size: int | None,
    precision: jax.lax.Precision,
    local_backend: str = "xla",
    native_amp: bool | None = None,
) -> jnp.ndarray:
    """Triangle attention with the row axis sharded across the ``cp`` mesh.

    Same layout logic as the Protenix port's ``_triangle_attention_cp``: each
    row of the pair representation is an independent attention, so a row shard
    needs no other shard's ``q``/``k``/``v``. The one cross-row input is the
    triangle bias, whose ``[j, k]`` indices read every row; it is projected
    from the sharded tensor (``heads`` channels, its cheapest form) and handed
    to every shard whole. The existing blocked row loop then runs unchanged on
    each device's local rows, which is the routing the SPMD partitioner cannot
    derive on its own -- the loop slices the sharded axis.
    """
    mesh = cp_mesh()
    if x.ndim != 4:
        msg = (
            "context-parallel triangle attention supports the native "
            f"[B, N, N, C] layout, got rank {x.ndim}"
        )
        raise ValueError(msg)
    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)
    # The reshard after the transpose is the Fold-CP "transpose exchange":
    # ending-node attention runs on columns, so the columns become the
    # sharded rows.
    x = shard_pair_rows(x, row_axis=-3)

    x = _layer_norm(
        x,
        params["layer_norm"]["scale"],
        params["layer_norm"]["bias"],
        eps,
    )
    mask = mask[..., :, None, None, :]
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    triangle_bias = _linear(x, params["linear"]["kernel"], precision)
    triangle_bias = jnp.transpose(triangle_bias, (0, 3, 1, 2))
    triangle_bias = jnp.expand_dims(triangle_bias, axis=1)

    if cp_layout() == "2d":
        # Keep both pair axes tiled for the whole softmax. Q stays resident;
        # K/V, mask and triangle bias rotate through the square mesh.
        out = _attention_ring_2d(
            params["mha"],
            q_x=x,
            kv_x=x,
            tri_bias=triangle_bias,
            mask_bias=mask_bias,
            precision=precision,
        )
        if not starting:
            out = jnp.swapaxes(out, -2, -3)
        return shard_pair_rows(out, row_axis=-3)

    # `shard_map` needs the sharded axis to divide evenly; pad the rows and
    # slice them back. Padded rows attend over real columns and are
    # discarded, so their values never reach a kept output.
    n_rows = x.shape[-3]
    pad = (-n_rows) % cp_row_shards()
    if pad:
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        mask_bias = jnp.pad(mask_bias, ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0)))

    def local_rows(x_l, mask_bias_l, bias_l, params_l):
        return _attention(
            params_l["mha"],
            q_x=x_l,
            kv_x=x_l,
            tri_bias=bias_l,
            mask_bias=mask_bias_l,
            chunk_size=chunk_size,
            q_chunk_size=q_chunk_size,
            precision=precision,
            # Inside `shard_map` each device holds whole rows, so the cueq
            # kernel runs unchanged on the local shard.
            triangle_backend=local_backend,
            native_amp=native_amp,
        )

    # Rows only, whichever layout is active. Attention stays row-sharded even
    # under the 2-D grid because each softmax spans a whole column axis, so a
    # column-split operand would need a gather inside every block; asking for
    # whole rows here lets the partitioner do that gather once, outside.
    out = jax.shard_map(
        local_rows,
        mesh=mesh,
        in_specs=(
            pair_row_spec(x.ndim),
            pair_row_spec(mask_bias.ndim, row_axis=1),
            PartitionSpec(),
            PartitionSpec(),
        ),
        out_specs=pair_row_spec(x.ndim),
    )(x, mask_bias, triangle_bias, params)
    if pad:
        # Slicing the padded rows away would otherwise let the partitioner
        # fall back to a replicated result.
        out = shard_pair_rows(
            jax.lax.slice_in_dim(out, 0, n_rows, axis=-3), row_axis=-3
        )
    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return out


def _attention_ring_2d(
    params: Mapping[str, Mapping[str, jnp.ndarray]],
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    *,
    precision: jax.lax.Precision,
) -> jnp.ndarray:
    """Project and run the exact two-dimensional Fold-CP attention ring."""

    no_heads = tri_bias.shape[2]
    c_hidden = params["linear_g"]["kernel"].shape[-1] // no_heads
    qg = _linear(
        q_x,
        jnp.concatenate(
            (params["linear_q"]["kernel"], params["linear_g"]["kernel"]),
            axis=-1,
        ),
        precision,
    )
    q, gate = jnp.split(qg, 2, axis=-1)
    kv = _linear(
        kv_x,
        jnp.concatenate(
            (params["linear_k"]["kernel"], params["linear_v"]["kernel"]),
            axis=-1,
        ),
        precision,
    )
    k, v = jnp.split(kv, 2, axis=-1)
    q = jnp.swapaxes(
        q.reshape(q.shape[:-1] + (no_heads, c_hidden)),
        -2,
        -3,
    )
    k = jnp.swapaxes(
        k.reshape(k.shape[:-1] + (no_heads, c_hidden)),
        -2,
        -3,
    )
    v = jnp.swapaxes(
        v.reshape(v.shape[:-1] + (no_heads, c_hidden)),
        -2,
        -3,
    )
    q = q / jnp.sqrt(jnp.asarray(c_hidden, dtype=q.dtype))
    out = ring_triangle_attention_2d(
        q,
        k,
        v,
        tri_bias,
        mask_bias,
        precision=precision,
    )
    out = jnp.swapaxes(out, -2, -3)
    gate = _sigmoid(gate)
    gate = gate.reshape(gate.shape[:-1] + (no_heads, c_hidden))
    out = out * gate
    out = out.reshape(out.shape[:-2] + (c_hidden * no_heads,))
    return _linear(out, params["linear_o"]["kernel"], precision)


def _attention(
    params: Mapping[str, Mapping[str, jnp.ndarray]],
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    chunk_size: int = 128,
    q_chunk_size: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    triangle_backend: str = "xla",
    native_amp: bool | None = None,
) -> jnp.ndarray:
    no_heads = tri_bias.shape[2]
    c_hidden = params["linear_g"]["kernel"].shape[-1] // no_heads
    # Upstream scales with `q /= math.sqrt(c_hidden)` on a BF16 tensor, which
    # CUDA computes in FP32 and rounds once. Rounding the divisor to BF16
    # first (5.65625 for c_hidden=32) is the FP32-model path, so the question
    # is the precision policy, not the activation width.
    native_amp = _resolve_native_amp(q_x, params["linear_q"]["kernel"], native_amp)

    qg = _linear(
        q_x,
        jnp.concatenate(
            (params["linear_q"]["kernel"], params["linear_g"]["kernel"]), axis=-1
        ),
        precision,
    )
    q, gate = jnp.split(qg, 2, axis=-1)
    kv = _linear(
        kv_x,
        jnp.concatenate(
            (params["linear_k"]["kernel"], params["linear_v"]["kernel"]), axis=-1
        ),
        precision,
    )
    k, v = jnp.split(kv, 2, axis=-1)

    q = q.reshape(q.shape[:-1] + (no_heads, c_hidden))
    k = k.reshape(k.shape[:-1] + (no_heads, c_hidden))
    v = v.reshape(v.shape[:-1] + (no_heads, c_hidden))
    q = jnp.swapaxes(q, -2, -3)
    k = jnp.swapaxes(k, -2, -3)
    v = jnp.swapaxes(v, -2, -3)

    if triangle_backend == "cueq":
        from foldjax.models.boltz2.models.triangle.triangle_cueq import (
            cueq_attention_core,
        )

        # `precision` is passed rather than left to the shared wrapper's
        # default, which derives it from `jax_default_matmul_precision`. This
        # port's two precision surfaces deliberately disagree -- the neutral
        # knob ships "high", the op-level string ships "highest" -- so deriving
        # it here would move the fused kernel from IEEE to TF32 and shift the
        # whole trunk capture. `resolve_matmul_precision` refusing the spelling
        # "high" is the tripwire for the edit that would do it.
        out = cueq_attention_core(
            q,
            k,
            v,
            tri_bias,
            mask_bias,
            scale=float(c_hidden**-0.5),
            precision=precision,
        )
    else:
        if native_amp:
            q = (q.astype(jnp.float32) / float(c_hidden**0.5)).astype(q.dtype)
        else:
            q_scale = jnp.sqrt(jnp.asarray(c_hidden, dtype=q.dtype))
            q = q / q_scale

    if triangle_backend == "pallas":
        from foldjax.models.boltz2.models.triangle.triangle_attention_pallas import (
            pallas_attention_core,
        )

        out = pallas_attention_core(q, k, v, tri_bias, mask_bias)
    elif triangle_backend == "tokamax":
        from foldjax.models.boltz2.models.triangle.triangle_attention_tokamax import (
            tokamax_attention_core,
        )

        out = tokamax_attention_core(q, k, v, tri_bias, mask_bias)
    elif triangle_backend == "xla":
        out = _attention_core(
            q,
            k,
            v,
            tri_bias,
            mask_bias,
            chunk_size,
            q_chunk_size,
            native_amp=native_amp,
        )
    elif triangle_backend != "cueq":
        msg = f"Unsupported triangle_backend: {triangle_backend!r}"
        raise ValueError(msg)
    out = jnp.swapaxes(out, -2, -3)

    gate = _sigmoid(gate)
    gate = gate.reshape(gate.shape[:-1] + (no_heads, c_hidden))
    out = out * gate
    out = out.reshape(out.shape[:-2] + (c_hidden * no_heads,))
    return _linear(out, params["linear_o"]["kernel"], precision)


def _attention_block(
    q_blk: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias_blk: jnp.ndarray,
    q_chunk_size: int | None = None,
    *,
    native_amp: bool = False,
) -> jnp.ndarray:
    """Exact attention for a chunk of query rows (axis=1 already sliced)."""
    n_q = q_blk.shape[-2]
    if q_chunk_size is not None and 0 < q_chunk_size < n_q:
        out = jnp.zeros_like(q_blk)
        for start in range(0, n_q, q_chunk_size):
            size = min(q_chunk_size, n_q - start)
            q_sub = jax.lax.dynamic_slice_in_dim(q_blk, start, size, axis=-2)
            bias_sub = jax.lax.dynamic_slice_in_dim(tri_bias, start, size, axis=-2)
            out = out.at[..., start : start + size, :].set(
                _attention_block(
                    q_sub, k, v, bias_sub, mask_bias_blk, native_amp=native_amp
                )
            )
        return out

    if native_amp:
        scores = jnp.matmul(q_blk, jnp.swapaxes(k, -1, -2))
        # The native non-kernel implementation uses two in-place additions:
        # each write rounds to the BF16 score buffer, even with FP32 masks.
        scores = (
            scores.astype(jnp.float32) + mask_bias_blk.astype(jnp.float32)
        ).astype(q_blk.dtype)
        scores = (scores.astype(jnp.float32) + tri_bias.astype(jnp.float32)).astype(
            q_blk.dtype
        )
        probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
            v.dtype
        )
        return jnp.matmul(probabilities, v)

    scores = jnp.matmul(
        q_blk.astype(jnp.float32),
        jnp.swapaxes(k.astype(jnp.float32), -1, -2),
    )
    scores = scores + mask_bias_blk.astype(jnp.float32) + tri_bias.astype(jnp.float32)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(attn, v.astype(jnp.float32)).astype(v.dtype)


def _attention_core(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    chunk_size: int,
    q_chunk_size: int | None = None,
    *,
    native_amp: bool = False,
) -> jnp.ndarray:
    """Compute attention, chunking over axis=1 (the triangle outer/batch axis).

    After reshape+swapaxes, q/k/v are [b, i, h, j, d]; matmul batches over
    (b, i, h) and contracts the last two axes, so axis=1 (i) is an outer batch
    axis: each i-slice is fully independent. Chunking it is exact (bit-parity):
    nothing is reduced across blocks.

    - q, k, v: slice axis=1 per block.
    - tri_bias: [b, 1, h, N, N] -> axis=1 size 1 broadcasts -> full each block.
    - mask_bias: [b, N, 1, 1, N] -> axis=1 broadcasts over the score rows but
      its first axis aligns with i; slice axis=1 per block.
    """
    n = q.shape[1]
    if chunk_size <= 0 or chunk_size >= n:
        return _attention_block(
            q, k, v, tri_bias, mask_bias, q_chunk_size, native_amp=native_amp
        )

    out = jnp.zeros_like(q)
    for start in range(0, n, chunk_size):
        size = min(chunk_size, n - start)
        q_blk = jax.lax.dynamic_slice_in_dim(q, start, size, axis=1)
        k_blk = jax.lax.dynamic_slice_in_dim(k, start, size, axis=1)
        v_blk = jax.lax.dynamic_slice_in_dim(v, start, size, axis=1)
        mask_blk = jax.lax.dynamic_slice_in_dim(mask_bias, start, size, axis=1)
        out = out.at[:, start : start + size].set(
            _attention_block(
                q_blk,
                k_blk,
                v_blk,
                tri_bias,
                mask_blk,
                q_chunk_size,
                native_amp=native_amp,
            )
        )
    return out


def _linear(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
) -> jnp.ndarray:
    if kernel.dtype == jnp.bfloat16:
        x = x.astype(kernel.dtype)
    return jnp.matmul(x, kernel, precision=precision)


def _layer_norm(x, scale, bias, eps):
    # Native triangle attention has a custom BF16-input exception: autocast
    # is disabled and affine values are narrowed before the fused norm.
    if x.dtype == jnp.bfloat16 and scale.dtype == jnp.float32:
        return _shared_layer_norm(
            x.astype(jnp.float32),
            scale.astype(jnp.bfloat16).astype(jnp.float32),
            bias.astype(jnp.bfloat16).astype(jnp.float32),
            eps,
        ).astype(jnp.bfloat16)
    return _shared_layer_norm(x, scale, bias, eps)
