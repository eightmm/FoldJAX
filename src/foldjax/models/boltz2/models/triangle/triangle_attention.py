"""Pure JAX triangle attention blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm

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
) -> jnp.ndarray:
    """Run Boltz TriangleAttention starting or ending node.

    ``triangle_backend``: ``"xla"`` (default, bit-exact dense/chunked path),
    ``"pallas"`` (opt-in GPU flash kernel), or ``"cueq"`` (Torch-compatible
    cuEquivariance CUDA kernel).
    """

    precision = resolve_matmul_precision(matmul_precision)

    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)

    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = _layer_norm(
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
    )

    if not starting:
        x = jnp.swapaxes(x, -2, -3)
    return x


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
) -> jnp.ndarray:
    no_heads = tri_bias.shape[2]
    c_hidden = params["linear_g"]["kernel"].shape[-1] // no_heads

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
        out = _attention_core(q, k, v, tri_bias, mask_bias, chunk_size, q_chunk_size)
    elif triangle_backend != "cueq":
        msg = f"Unsupported triangle_backend: {triangle_backend!r}"
        raise ValueError(msg)
    out = jnp.swapaxes(out, -2, -3)

    gate = jax.nn.sigmoid(gate)
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
                _attention_block(q_sub, k, v, bias_sub, mask_bias_blk)
            )
        return out

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
        return _attention_block(q, k, v, tri_bias, mask_bias, q_chunk_size)

    out = jnp.zeros_like(q)
    for start in range(0, n, chunk_size):
        size = min(chunk_size, n - start)
        q_blk = jax.lax.dynamic_slice_in_dim(q, start, size, axis=1)
        k_blk = jax.lax.dynamic_slice_in_dim(k, start, size, axis=1)
        v_blk = jax.lax.dynamic_slice_in_dim(v, start, size, axis=1)
        mask_blk = jax.lax.dynamic_slice_in_dim(mask_bias, start, size, axis=1)
        out = out.at[:, start : start + size].set(
            _attention_block(q_blk, k_blk, v_blk, tri_bias, mask_blk, q_chunk_size)
        )
    return out


def _linear(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
) -> jnp.ndarray:
    return jnp.matmul(x, kernel, precision=precision)
