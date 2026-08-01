"""Pallas (Triton GPU) flash kernel for Boltz triangle attention.

Replaces the score -> softmax -> @v block of ``_attention_core`` with a
fused flash-attention kernel that never materializes the O(N^3) score tensor.

Geometry (after the reshape/swapaxes in ``_attention``):
    q, k, v : [b, i, h, j|k, d]   (i is an outer batch axis; j queries, k keys)
    bias[j, k] = tri_bias[b, h, j, k]  (broadcast over i)
               + mask_bias[b, i, k]    (broadcast over j)

The kernel flattens (b, i, h) into a single program grid axis and, per query
block, streams key blocks with an online-softmax (running max/sum), adding the
two bias terms per key tile. fp32 accumulation throughout.

The reduction over keys is re-associated relative to the dense XLA softmax, so
results match XLA to ~1e-4..1e-3 (fp32), not bit-exactly -- hence opt-in.

Import is guarded: on a box without a working Pallas/Triton GPU backend this
module still imports; ``pallas_attention_core`` raises only when actually
called.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

try:  # pragma: no cover - exercised only on GPU
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    _PALLAS_AVAILABLE = True
except Exception:  # pragma: no cover
    pl = None
    plgpu = None
    _PALLAS_AVAILABLE = False


def pallas_available() -> bool:
    """True if the Pallas Triton GPU backend imported."""
    return _PALLAS_AVAILABLE


def _flash_kernel(
    q_ref,  # [block_q, d]
    k_ref,  # [N_k, d]
    v_ref,  # [N_k, d]
    tri_ref,  # [block_q, N_k]   tri_bias[h, j_block, :]
    mask_ref,  # [N_k]           mask_bias[i, :]
    o_ref,  # [block_q, d]
    *,
    block_k: int,
    n_k: int,
):
    """Online-softmax flash attention for one (b,i,h) query block.

    q is pre-scaled by 1/sqrt(d) (matches the existing JAX path). Bias is added
    as tri_bias[j, k] + mask_bias[k] per key tile.
    """
    # Refs carry leading singleton grid dims:
    #   q/k/v/o: [1,1,1,block,d]   tri: [1,1,block_q,n_k]   mask: [1,1,n_k]
    block_q = q_ref.shape[-2]
    d = q_ref.shape[-1]

    q = q_ref[0, 0, 0].astype(jnp.float32)  # [block_q, d]

    m_i = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((block_q,), dtype=jnp.float32)
    acc = jnp.zeros((block_q, d), dtype=jnp.float32)

    num_kblocks = pl.cdiv(n_k, block_k)

    def body(kb, carry):
        m_prev, l_prev, acc_prev = carry
        start = kb * block_k
        k_slice = pl.ds(start, block_k)

        k = k_ref[0, 0, 0, k_slice, :].astype(jnp.float32)  # [block_k, d]
        v = v_ref[0, 0, 0, k_slice, :].astype(jnp.float32)  # [block_k, d]
        tri = tri_ref[0, 0, :, k_slice].astype(jnp.float32)  # [block_q, block_k]
        msk = mask_ref[0, 0, k_slice].astype(jnp.float32)  # [block_k]

        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )  # [block_q, block_k]
        s = s + tri + msk[None, :]

        # mask out-of-range keys (when n_k not a multiple of block_k)
        col = start + jax.lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1)
        s = jnp.where(col < n_k, s, -jnp.inf)

        m_cur = jnp.maximum(m_prev, jnp.max(s, axis=1))
        m_cur = jnp.where(jnp.isinf(m_cur) & (m_cur < 0), 0.0, m_cur)
        p = jnp.exp(s - m_cur[:, None])  # [block_q, block_k]
        alpha = jnp.exp(m_prev - m_cur)

        l_cur = l_prev * alpha + jnp.sum(p, axis=1)
        acc_cur = acc_prev * alpha[:, None] + jax.lax.dot_general(
            p, v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        return m_cur, l_cur, acc_cur

    m_i, l_i, acc = jax.lax.fori_loop(0, num_kblocks, body, (m_i, l_i, acc))

    l_safe = jnp.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    o_ref[0, 0, 0] = out.astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("block_q", "block_k"))
def _flash_attention(
    q: jnp.ndarray,  # [b, i, h, Nq, d]
    k: jnp.ndarray,  # [b, i, h, Nk, d]
    v: jnp.ndarray,  # [b, i, h, Nk, d]
    tri: jnp.ndarray,  # [b, h, Nq, Nk]   (NO i axis)
    mask: jnp.ndarray,  # [b, i, Nk]       (NO h axis)
    block_q: int,
    block_k: int,
) -> jnp.ndarray:
    """Flash kernel over a 3D (b, i, h) grid.

    tri_bias keeps shape [b, h, N, N] and mask_bias [b, i, N]; the BlockSpec
    index maps select the right (b,h) / (b,i) slice per grid cell, so the
    O(N^3) i-broadcast of tri_bias is NEVER materialized in HBM.
    """
    b, i, h, n_q, d = q.shape
    n_k = k.shape[3]

    # Pad key axis to a multiple of block_k (kernel masks padded cols).
    n_k_pad = pl.cdiv(n_k, block_k) * block_k
    if n_k_pad != n_k:
        pad = n_k_pad - n_k
        k = jnp.pad(k, ((0, 0),) * 3 + ((0, pad), (0, 0)))
        v = jnp.pad(v, ((0, 0),) * 3 + ((0, pad), (0, 0)))
        tri = jnp.pad(tri, ((0, 0), (0, 0), (0, 0), (0, pad)))
        mask = jnp.pad(mask, ((0, 0), (0, 0), (0, pad)))

    # Pad query axis to a multiple of block_q.
    n_q_pad = pl.cdiv(n_q, block_q) * block_q
    if n_q_pad != n_q:
        qpad = n_q_pad - n_q
        q = jnp.pad(q, ((0, 0),) * 3 + ((0, qpad), (0, 0)))
        tri = jnp.pad(tri, ((0, 0), (0, 0), (0, qpad), (0, 0)))

    grid = (b, i, h, pl.cdiv(n_q_pad, block_q))
    kernel = functools.partial(_flash_kernel, block_k=block_k, n_k=n_k)

    # grid index = (bb, ii, hh, jj)
    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec(
                (1, 1, 1, block_q, d), lambda b_, i_, h_, j_: (b_, i_, h_, j_, 0)
            ),
            pl.BlockSpec(
                (1, 1, 1, n_k_pad, d), lambda b_, i_, h_, j_: (b_, i_, h_, 0, 0)
            ),
            pl.BlockSpec(
                (1, 1, 1, n_k_pad, d), lambda b_, i_, h_, j_: (b_, i_, h_, 0, 0)
            ),
            # tri: [b, h, Nq, Nk] -> index (b_, h_, j_, 0)
            pl.BlockSpec(
                (1, 1, block_q, n_k_pad), lambda b_, i_, h_, j_: (b_, h_, j_, 0)
            ),
            # mask: [b, i, Nk] -> index (b_, i_, 0)
            pl.BlockSpec((1, 1, n_k_pad), lambda b_, i_, h_, j_: (b_, i_, 0)),
        ],
        out_specs=pl.BlockSpec(
            (1, 1, 1, block_q, d), lambda b_, i_, h_, j_: (b_, i_, h_, j_, 0)
        ),
        out_shape=jax.ShapeDtypeStruct((b, i, h, n_q_pad, d), q.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
    )(q, k, v, tri, mask)
    return out[:, :, :, :n_q, :]


def pallas_attention_core(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    block_q: int = 64,
    block_k: int = 64,
) -> jnp.ndarray:
    """Drop-in replacement for ``_attention_core`` using the Pallas kernel.

    q, k, v   : [b, i, h, j|k, d]
    tri_bias  : [b, 1, h, N, N]  (broadcast over i)
    mask_bias : [b, N, 1, 1, N]  (i on axis 1, broadcast over h and j)
    returns out [b, i, h, j, d]
    """
    if not _PALLAS_AVAILABLE:
        msg = "Pallas/Triton GPU backend unavailable; cannot run pallas attention."
        raise RuntimeError(msg)

    # Strip broadcast singletons WITHOUT materializing the i-broadcast:
    # tri_bias [b,1,h,N,N] -> [b,h,N,N]; mask_bias [b,N,1,1,N] -> [b,N,N].
    tri = tri_bias[:, 0]  # [b, h, Nq, Nk]
    msk = mask_bias[:, :, 0, 0, :]  # [b, i, Nk]

    return _flash_attention(q, k, v, tri, msk, block_q, block_k)
