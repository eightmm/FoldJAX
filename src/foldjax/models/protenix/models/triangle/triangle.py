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
    """Return the configured triangle-attention backend.

    Defaults to the XLA path because it is the only one that blocks rows, and
    blocking is what bounds the score tensor. The cuEquivariance kernel was the
    default on the assumption that it fused the score away; measured, it does
    not -- at 490 tokens it and the unblocked XLA path both peak at 6,048 MiB,
    where the blocked XLA path peaks at 4,348. Set
    ``PROTENIX_TRIANGLE_BACKEND=cueq_jit`` to go back.
    """
    backend = os.environ.get("PROTENIX_TRIANGLE_BACKEND", "xla_jit").lower()
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
    # Same reasoning as the attention: the block belongs to the tensor, not to
    # the token count. One row of `a` costs `N * c_hidden * itemsize`, so the
    # chunk policy's 256 means 373 MiB per block on OpenDDE's structural branch
    # and a fifth of that on a narrower trunk. Swept there against the policy's
    # value, the peak went 9,809 -> 9,652 -> 9,628 MiB at 256 -> 128 -> 64 rows,
    # against a 9 MiB run-to-run spread.
    chunk_size = _row_block(
        rows=z.shape[-3],
        per_row=z.shape[-2] * params.linear_a_p.weight.shape[0] * z.dtype.itemsize,
        requested=chunk_size,
        budget=_PROJECTION_BUDGET_BYTES,
    )

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
    """Say that a requested chunk size is not used, once.

    The cuEquivariance kernel takes the whole thing, so a chunk size does not
    reach it -- and it does not make up for that by fusing the score tensor
    away. Measured at 490 tokens, cuEquivariance and the *unblocked* XLA path
    peak identically (6,048 and 6,049 MiB) where the blocked XLA path peaks at
    4,348, so choosing it costs the memory the chunk size would have saved.

    Not an exception, because the automatic chunk policy legitimately emits a
    size without knowing which backend will run.
    """
    global _WARNED_UNCHUNKABLE
    if _WARNED_UNCHUNKABLE:
        return
    _WARNED_UNCHUNKABLE = True
    warnings.warn(
        f"triangle attention q_chunk_size={q_chunk_size} is not used by the "
        "'cueq' backend, which still materialises the score tensor the chunk "
        "size exists to bound. The default 'xla' backend blocks it instead, "
        "and measured lower on every size tried.",
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


# One row of the score tensor costs `heads * N * N * 4` bytes -- the logits are
# float32 whatever the trunk dtype is, because the mask bias is. Blocking rows
# is what bounds it, and both branches were swept independently, at two sizes
# each:
#
#   protenix, 4 heads:   64 rows optimal at 490 tokens and again at 976
#   opendde, 12 heads:   ~25 rows optimal at 948 tokens and again at 1,892
#
# The optimum is the same row count at both sizes for a given branch, so it is
# not a byte budget -- a budget halves the count every time the tokens double,
# and at 1,892 tokens that meant 6 rows, which measured 43,569 MiB against
# 32,641 at 25. What is invariant is `rows * heads`: 64*4 = 256, 25*12 = 300.
# Too few rows pays per-block overhead across hundreds of blocks; too many lets
# the score tensor take over.
#
# Bytes stay on as a ceiling, because a fixed row count grows the score as N^2
# and would run away on a very large complex.
_SCORE_ROWS_TIMES_HEADS = 288
_SCORE_CEILING_BYTES = 8 * 1024**3
_PROJECTION_BUDGET_BYTES = 128 * 1024**2
_MAX_ROWS_PER_BLOCK = 64
_MIN_ROWS_PER_BLOCK = 8


def _row_block(
    *,
    rows: int,
    per_row: int,
    requested: int | None,
    cap: int = _MAX_ROWS_PER_BLOCK,
    budget: int = _SCORE_CEILING_BYTES,
) -> int | None:
    """Rows per block: the caller's request, narrowed to what the budget allows."""
    if per_row <= 0 or rows < 2:
        return requested
    allowed = max(_MIN_ROWS_PER_BLOCK, min(cap, budget // per_row))
    if allowed >= rows:
        return requested
    return allowed if requested is None else min(requested, allowed)


def _score_rows(*, rows: int, cols: int, num_heads: int, requested: int | None):
    """Row block for a triangle attention, from its own head count and width."""
    return _row_block(
        rows=rows,
        per_row=num_heads * cols * cols * 4,
        requested=requested,
        cap=min(_MAX_ROWS_PER_BLOCK, max(1, _SCORE_ROWS_TIMES_HEADS // num_heads)),
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
    # Triangle attention is a batch of independent attentions, one per row of
    # the pair representation: `q`, `k` and `v` are [N_row, heads, N_col, d] and
    # nothing in the softmax crosses a row. Blocking that row axis therefore
    # bounds all three projections *and* the [N_row, heads, N_col, N_col] score
    # tensor at once, for the same block count, the same score budget and the
    # same arithmetic -- each row is projected exactly once either way.
    #
    # Blocking the query axis instead, which is what this did, bounds only the
    # scores and `q`; `k` and `v` stay whole, at 1,311 MiB apiece on OpenDDE's
    # structural branch.
    if attention_backend not in {"cueq", "tokamax"}:
        q_chunk_size = _score_rows(
            rows=x.shape[-3],
            cols=x.shape[-2],
            num_heads=num_heads,
            requested=q_chunk_size,
        )
    chunked = (
        q_chunk_size is not None
        and attention_backend not in {"cueq", "tokamax"}
        and 0 < q_chunk_size < x.shape[-3]
    )
    scale = float(params.attention.linear_k.weight.shape[0] // num_heads) ** -0.5
    if chunked:
        q = k = v = None
    else:
        q = _project_heads(x, params.attention.linear_q, num_heads)
        k = _project_heads(x, params.attention.linear_k, num_heads)
        v = _project_heads(x, params.attention.linear_v, num_heads)

    if attention_backend == "cueq" and x.shape[-2] > 16:
        # The cuEquivariance kernel is fused and takes the whole thing, so a
        # chunk size does not reach it. Nothing is materialised that a chunk
        # size would have bounded; say so once rather than let a knob look
        # like it worked.
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
        rows = x.shape[-3]
        # Written into one preallocated buffer rather than concatenated at the
        # end: at 1,892 structural tokens the row budget picks blocks of six,
        # and holding all 316 results live to concatenate them costs a second
        # copy of the [N_row, heads, N_col, d] output -- 5,244 MiB.
        head_dim = params.attention.linear_q.weight.shape[0] // num_heads
        out = jnp.zeros(
            x.shape[:-3] + (rows, num_heads, x.shape[-2], head_dim),
            dtype=x.dtype,
        )
        for start in range(0, rows, q_chunk_size):
            size = min(q_chunk_size, rows - start)
            # One block of rows, projected straight from `x`. Slicing the row
            # axis and then projecting is the same arithmetic as projecting and
            # then slicing -- the linear contracts over the channel axis -- and
            # each row is still projected exactly once across the whole loop.
            x_rows = jax.lax.dynamic_slice_in_dim(x, start, size, axis=-3)
            q_block = _project_heads(x_rows, params.attention.linear_q, num_heads)
            q_block = q_block * jnp.asarray(scale, dtype=q_block.dtype)
            block = _triangle_attention_block(
                q_block,
                _project_heads(x_rows, params.attention.linear_k, num_heads),
                _project_heads(x_rows, params.attention.linear_v, num_heads),
                # mask_bias is [..., N_row, 1, 1, N_col]; the row axis is -4.
                # triangle_bias is [..., 1, heads, N, N] -- already broadcast
                # over rows, so it is passed through whole.
                jax.lax.dynamic_slice_in_dim(mask_bias, start, size, axis=-4),
                triangle_bias,
            )
            out = out.at[..., start : start + size, :, :, :].set(
                block.astype(out.dtype)
            )

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
