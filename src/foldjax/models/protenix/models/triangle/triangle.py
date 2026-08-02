"""Triangle pair-update blocks for the Protenix JAX port."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.attention import AttentionParams
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
    sigmoid,
)


def _triangle_attention_backend() -> str:
    """Return the configured triangle-attention backend."""
    backend = os.environ.get("PROTENIX_TRIANGLE_BACKEND", "cueq_jit").lower()
    if backend not in {"xla", "xla_jit", "tokamax", "cueq", "cueq_jit"}:
        raise ValueError(f"unsupported triangle attention backend: {backend!r}")
    return backend

TriangleDirection = Literal["outgoing", "incoming"]


class TriangleMultiplicationParams(NamedTuple):
    """Parameters for ``TriangleMultiplicativeUpdate``."""

    layer_norm_in: LayerNormParams
    layer_norm_out: LayerNormParams
    linear_a_p: LinearParams
    linear_a_g: LinearParams
    linear_b_p: LinearParams
    linear_b_g: LinearParams
    linear_z: LinearParams
    linear_g: LinearParams


class TriangleAttentionParams(NamedTuple):
    """Parameters for Protenix ``TriangleAttention``."""

    layer_norm: LayerNormParams
    linear: LinearParams
    attention: AttentionParams


def triangle_multiplication(
    z: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: TriangleMultiplicationParams,
    direction: TriangleDirection,
    *,
    chunk_size: int | None = None,
    use_jit: bool = False,
) -> jnp.ndarray:
    """Apply Protenix triangle multiplication without eval in-place mutation."""

    if use_jit:
        return _compiled_triangle_multiplication(
            z,
            mask,
            params,
            direction,
            chunk_size=chunk_size,
            use_jit=False,
        )
    if mask is None:
        mask = jnp.ones(z.shape[:-1], dtype=z.dtype)
    backend = os.environ.get(
        "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq"
    ).lower()
    cueq_supported = (
        params.linear_a_p.weight.shape[0] == z.shape[-1]
        and z.shape[-1] % 32 == 0
    )
    if backend == "cueq" and cueq_supported:
        from foldjax.models.protenix.models.triangle.triangle_cueq import (
            cueq_triangle_multiplication,
        )

        return cueq_triangle_multiplication(z, mask, params, direction)
    if backend not in {"cueq", "xla"}:
        raise ValueError(f"unsupported triangle multiplication backend: {backend!r}")
    mask = mask.astype(z.dtype)[..., None]

    z_norm = layer_norm(z, params.layer_norm_in)
    b = mask * sigmoid(linear(z_norm, params.linear_b_g))
    b = b * linear(z_norm, params.linear_b_p)

    def project_a(z_slice: jnp.ndarray, mask_slice: jnp.ndarray) -> jnp.ndarray:
        gated = mask_slice * sigmoid(linear(z_slice, params.linear_a_g))
        return gated * linear(z_slice, params.linear_a_p)

    # Keep the projections in the trunk's own dtype and accumulate in float32
    # instead of widening them first. `a` and `b` are each [N, N, c_hidden] --
    # 1,311 MiB apiece on OpenDDE's structural branch -- so casting them up
    # meant a bfloat16 trunk still paid float32 for its two largest buffers.
    # `preferred_element_type` gives the same float32 accumulation without the
    # widened operands, which is what AlphaFold 3's BF16_BF16_F32 algorithm
    # does. With a float32 trunk both forms are identical.
    #
    # `a` is also built one block at a time when the contraction is blocked.
    # Each output row block reduces over the whole of `b` but only its own rows
    # of `a`, so the rest of `a` never has to exist -- the same trade the
    # attention makes with its queries.
    out = _triangle_contract(
        project_a,
        z_norm,
        mask,
        b,
        direction,
        chunk_size,
    )
    out = out.astype(z.dtype)
    out = layer_norm(out, params.layer_norm_out)
    out = linear(out, params.linear_z)
    return out * sigmoid(linear(z_norm, params.linear_g))


_compiled_triangle_multiplication = jax.jit(
    triangle_multiplication,
    static_argnames=("direction", "chunk_size", "use_jit"),
)


def _triangle_contract(
    project_a: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    z_norm: jnp.ndarray,
    mask: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
    chunk_size: int | None,
) -> jnp.ndarray:
    """Contract ``a`` against ``b``, building ``a`` whole or one block at a time.

    ``project_a`` takes a slice of the normalised pair representation and its
    matching mask slice and returns that slice of ``a``. Blocking the output
    rows only ever needs the matching rows of ``a``, so when the contraction is
    blocked the full projection is never materialised.
    """
    n = z_norm.shape[-3]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        return _triangle_contract_block(project_a(z_norm, mask), b, direction)

    # The blocks accumulate in float32 regardless of the operand dtype, so the
    # destination has to be float32 too or every block would be rounded back
    # down on the way in.
    axis = -3 if direction == "outgoing" else -2
    blocks = []
    for start in range(0, n, chunk_size):
        size = min(chunk_size, n - start)
        a_block = project_a(
            jax.lax.dynamic_slice_in_dim(z_norm, start, size, axis=axis),
            jax.lax.dynamic_slice_in_dim(mask, start, size, axis=axis),
        )
        blocks.append(
            _triangle_contract_block(a_block, b, direction).astype(jnp.float32)
        )
    return jnp.concatenate(blocks, axis=-3)


def _triangle_contract_block(
    a: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
) -> jnp.ndarray:
    # NOTE: writing this as permute_final_dims+matmul (to mirror torch's GEMM)
    # is a no-op in JAX — XLA lowers both einsum and matmul to the same
    # dot_general, so the reduction order is XLA's choice regardless. The
    # einsum form is kept for clarity.
    if direction == "outgoing":
        return jnp.einsum(
            "...ikd,...jkd->...ijd", a, b, preferred_element_type=jnp.float32
        )
    if direction == "incoming":
        return jnp.einsum(
            "...kid,...kjd->...ijd", a, b, preferred_element_type=jnp.float32
        )
    raise ValueError(f"unsupported triangle direction: {direction!r}")


_WARNED_UNCHUNKABLE = False


def _warn_unchunkable_backend(q_chunk_size: int) -> None:
    """Say that a requested chunk size cannot be honoured, once.

    The cuEquivariance kernel takes the whole query axis; blocking is only
    implemented in the XLA path. A chunk size that reaches this branch bounds
    nothing, and the [N, h, N, N] score tensor it exists to bound is
    materialised whole. That is worth one line on stderr rather than silence --
    and not an exception, because the automatic chunk policy legitimately emits
    a size without knowing which backend will run.
    """
    global _WARNED_UNCHUNKABLE
    if _WARNED_UNCHUNKABLE:
        return
    _WARNED_UNCHUNKABLE = True
    warnings.warn(
        f"triangle attention q_chunk_size={q_chunk_size} is ignored by the "
        "'cueq' backend, which has no chunked kernel: the full score tensor is "
        "materialised. Pass attention_backend='xla' to block the query axis.",
        RuntimeWarning,
        stacklevel=2,
    )


def triangle_attention(
    x: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: TriangleAttentionParams,
    *,
    num_heads: int,
    starting: bool = True,
    inf: float = 1e9,
    q_chunk_size: int | None = None,
    attention_backend: str | None = None,
) -> jnp.ndarray:
    """Apply Protenix triangle attention in the dense XLA path."""

    backend = (
        _triangle_attention_backend()
        if attention_backend is None
        else attention_backend
    )
    if backend in {"xla_jit", "cueq_jit"}:
        return _compiled_triangle_attention(
            x,
            mask,
            params,
            num_heads=num_heads,
            starting=starting,
            inf=inf,
            q_chunk_size=q_chunk_size,
            attention_backend=backend.removesuffix("_jit"),
        )
    if backend not in {"xla", "tokamax", "cueq"}:
        raise ValueError(f"unsupported triangle attention backend: {backend!r}")
    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = layer_norm(x, params.layer_norm)
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    mask_bias = mask_bias[..., :, None, None, :]
    triangle_bias = linear(x, params.linear)
    triangle_bias = jnp.moveaxis(triangle_bias, -1, -3)
    triangle_bias = jnp.expand_dims(triangle_bias, axis=-4)

    out = _triangle_attention_dense(
        x,
        params,
        num_heads,
        mask_bias,
        triangle_bias,
        q_chunk_size,
        backend,
    )
    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return out


_compiled_triangle_attention = jax.jit(
    triangle_attention,
    static_argnames=(
        "num_heads",
        "starting",
        "inf",
        "q_chunk_size",
        "attention_backend",
    ),
)


def _triangle_attention_dense(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    num_heads: int,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    q_chunk_size: int | None,
    attention_backend: str,
) -> jnp.ndarray:
    k = _project_heads(x, params.attention.linear_k, num_heads)
    v = _project_heads(x, params.attention.linear_v, num_heads)
    scale = float(k.shape[-1] ** -0.5)
    # The chunked path below reads one block of queries at a time, so it
    # projects them per block instead of holding the whole thing. `q` is
    # [N, heads, N, d] -- 1,311 MiB on OpenDDE's structural branch, the same
    # size as `k` and `v`, and unlike them it is never needed whole. Every
    # other path does need it whole.
    chunked = (
        attention_backend not in {"cueq", "tokamax"}
        and q_chunk_size is not None
        and 0 < q_chunk_size < x.shape[-2]
    )
    q = (
        None
        if chunked
        else _project_heads(x, params.attention.linear_q, num_heads)
    )

    if attention_backend == "cueq" and x.shape[-2] > 16:
        # The cuEquivariance kernel takes the whole query axis; there is no
        # chunked entry point. Blocking is only implemented in the XLA path
        # below, so a caller who asked for a chunk size and got this branch
        # would silently get an unchunked, unbounded [N, h, N, N] score tensor
        # -- which is exactly the buffer the chunk size exists to bound. Say so
        # rather than accepting the argument and ignoring it.
        if q_chunk_size is not None and 0 < q_chunk_size < x.shape[-2]:
            _warn_unchunkable_backend(q_chunk_size)
        from foldjax.models.protenix.models.triangle.triangle_cueq import (
            cueq_attention_core,
        )

        out = cueq_attention_core(
            q,
            k,
            v,
            triangle_bias,
            mask_bias,
            scale=scale,
        )
    elif q is not None:
        # The chunked path scales each block as it is projected instead.
        q = q * jnp.asarray(scale, dtype=q.dtype)

    if attention_backend == "tokamax":
        # q/k/v are [..., N_row, h, j, d], triangle_bias [..., 1, h, N, N],
        # mask_bias [..., N, 1, 1, N] — already the layout tokamax expects; q is
        # pre-scaled (scale=1.0 inside). Returns [..., h, i, d] like the block.
        from foldjax.models.protenix.models.triangle.triangle_attention_tokamax import (
            tokamax_attention_core,
        )

        out = tokamax_attention_core(q, k, v, triangle_bias, mask_bias)
    elif attention_backend == "cueq" and x.shape[-2] > 16:
        pass
    elif not chunked:
        out = _triangle_attention_block(q, k, v, mask_bias, triangle_bias)
    else:
        rows = x.shape[-2]
        blocks = []
        for start in range(0, rows, q_chunk_size):
            size = min(q_chunk_size, rows - start)
            # Project this block's queries straight from `x`. Slicing `x` on
            # the query axis and then projecting is the same arithmetic as
            # projecting and then slicing -- the linear contracts over the
            # channel axis -- but it never builds the full `q`.
            q_block = _project_heads(
                jax.lax.dynamic_slice_in_dim(x, start, size, axis=-2),
                params.attention.linear_q,
                num_heads,
            )
            q_block = q_block * jnp.asarray(scale, dtype=q_block.dtype)
            tri_block = jax.lax.dynamic_slice_in_dim(
                triangle_bias,
                start,
                size,
                axis=-2,
            )
            blocks.append(
                _triangle_attention_block(q_block, k, v, mask_bias, tri_block)
            )
        out = jnp.concatenate(blocks, axis=-2)

    out = jnp.swapaxes(out, -2, -3)
    if params.attention.linear_g is not None:
        gate = sigmoid(linear(x, params.attention.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.attention.linear_o)


def _triangle_attention_block(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
) -> jnp.ndarray:
    logits = jnp.einsum("...hid,...hjd->...hij", q, k)
    logits = logits + mask_bias + triangle_bias
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("...hij,...hjd->...hid", probs, v)


def _project_heads(
    x: jnp.ndarray,
    params: LinearParams,
    num_heads: int,
) -> jnp.ndarray:
    y = linear(x, params)
    y = y.reshape(y.shape[:-1] + (num_heads, -1))
    return jnp.swapaxes(y, -2, -3)
