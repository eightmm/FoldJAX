"""Prototype Pallas-Triton kernels for the pair-stack shootout.

Three kernels, all bf16 operands with f32 accumulation and f32 normalisation:

* ``flash_triangle_attention``: flash attention over the attended axis with the
  shared ``[H, N, N]`` triangle bias (read per (head, query block), never
  broadcast over rows) plus a per-row additive key mask ``[R, N]``.
* ``trimul_k1`` / ``trimul_k2``: the FlashPairformer split of the triangle
  multiplicative update.  K1 = LayerNorm + sigmoid-gated input projections +
  mask, written channel-major ``[H, N*N]`` so the contraction is a plain
  batched GEMM (cuBLAS through XLA's dot_general).  K2 = output LayerNorm +
  output projection + output gate recomputed from ``x``.
* ``fused_transition``: LayerNorm + SwiGLU (``silu(x W1) * (x W2)``) + ``W3``
  in one pass, hidden width processed in chunks so the widened form is never
  written.

Out-of-range rows are handled with masked loads/stores: Pallas-Triton loads are
unmasked unless a mask is given, so every block that can overhang the array is
loaded with one.

``INTERPRET = True`` runs every kernel in Pallas interpret mode (CPU checks).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

INTERPRET = False
_DEFAULT = lax.Precision.DEFAULT


def _pallas_call(kernel, *, num_warps: int, num_stages: int, **kwargs):
    if INTERPRET:
        return pl.pallas_call(kernel, interpret=True, **kwargs)
    return pl.pallas_call(
        kernel,
        compiler_params=plt.CompilerParams(num_warps=num_warps, num_stages=num_stages),
        **kwargs,
    )


def _mm(a, b):
    """[m, k] @ [k, n] with f32 accumulation (bf16 tensor cores for bf16 inputs)."""
    return lax.dot_general(
        a, b, (((1,), (0,)), ((), ())), precision=_DEFAULT,
        preferred_element_type=jnp.float32,
    )


def _mm_nt(a, b):
    """[m, k] @ [n, k]^T with f32 accumulation."""
    return lax.dot_general(
        a, b, (((1,), (1,)), ((), ())), precision=_DEFAULT,
        preferred_element_type=jnp.float32,
    )


# --------------------------------------------------------------------------- #
# Triangle attention
# --------------------------------------------------------------------------- #


def _flash_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, o_ref, *, n, rows, rb, bq, bk,
                  scale):
    """``rb`` pair rows per program share one load of each bias tile.

    The triangle bias is the same for every row, and at f32 it is the largest
    stream the kernel reads (``rows * H * N * N * 4`` bytes over a call); a
    program that walks ``rb`` rows through each key block reads it ``rb``
    times less."""
    r0 = pl.program_id(0) * rb
    j = pl.program_id(2)
    qvalid = (j * bq + jnp.arange(bq)) < n
    rvalid = [(r0 + i) < rows for i in range(rb)]
    qs = [
        plt.load(q_ref.at[jnp.int32(i)], mask=qvalid[:, None] & rvalid[i], other=0.0)
        for i in range(rb)
    ]

    def body(t, carry):
        start = t * bk
        ks = pl.ds(start, bk)
        kvalid = (start + jnp.arange(bk)) < n
        bias = plt.load(
            b_ref.at[:, ks], mask=qvalid[:, None] & kvalid[None, :], other=0.0
        ).astype(jnp.float32)
        new = []
        for i in range(rb):
            m_i, l_i, acc = carry[i]
            kmask = kvalid[:, None] & rvalid[i]
            k = plt.load(k_ref.at[jnp.int32(i), ks, :], mask=kmask, other=0.0)
            s = _mm_nt(qs[i], k) * scale + bias
            s = s + plt.load(
                m_ref.at[jnp.int32(i), ks], mask=kvalid & rvalid[i], other=float("-inf")
            )[None, :]
            m_new = jnp.maximum(m_i, jnp.max(s, axis=1))
            p = jnp.exp(s - m_new[:, None])
            alpha = jnp.exp(m_i - m_new)
            l_new = l_i * alpha + jnp.sum(p, axis=1)
            v = plt.load(v_ref.at[jnp.int32(i), ks, :], mask=kmask, other=0.0)
            new.append((m_new, l_new, acc * alpha[:, None] + _mm(p.astype(v.dtype), v)))
        return tuple(new)

    d = qs[0].shape[-1]
    init = tuple(
        (
            jnp.full((bq,), float("-inf"), jnp.float32),
            jnp.zeros((bq,), jnp.float32),
            jnp.zeros((bq, d), jnp.float32),
        )
        for _ in range(rb)
    )
    final = lax.fori_loop(0, pl.cdiv(n, bk), body, init)
    for i in range(rb):
        _, l_i, acc = final[i]
        out = acc / l_i[:, None]
        plt.store(o_ref.at[jnp.int32(i)], out.astype(o_ref.dtype), mask=qvalid[:, None] & rvalid[i])


@functools.partial(
    jax.jit, static_argnames=("scale", "rb", "bq", "bk", "num_warps", "num_stages")
)
def flash_triangle_attention(
    q, k, v, bias, key_mask, *, scale, rb=1, bq=64, bk=64, num_warps=4, num_stages=2
):
    """q/k/v ``[R, N, H, D]`` (module layout, no transposes), bias ``[H, N, N]``
    (f32 or bf16, added in f32), key_mask ``[R, N]`` f32 additive.  Returns ``[R, N, H, D]``.

    A fully masked row (every key at the mask value) averages its values
    uniformly, as a dense softmax over ``-1e9``-shifted logits does."""
    r, n, h, d = q.shape
    kernel = functools.partial(
        _flash_kernel, n=n, rows=r, rb=rb, bq=bq, bk=bk, scale=scale
    )
    grid = (pl.cdiv(r, rb), h, pl.cdiv(n, bq))
    return _pallas_call(
        kernel,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        in_specs=[
            pl.BlockSpec((rb, bq, None, d), lambda ri, hi, ji: (ri, ji, hi, 0)),
            pl.BlockSpec((rb, n, None, d), lambda ri, hi, ji: (ri, 0, hi, 0)),
            pl.BlockSpec((rb, n, None, d), lambda ri, hi, ji: (ri, 0, hi, 0)),
            pl.BlockSpec((None, bq, n), lambda ri, hi, ji: (hi, ji, 0)),
            pl.BlockSpec((rb, n), lambda ri, hi, ji: (ri, 0)),
        ],
        out_specs=pl.BlockSpec((rb, bq, None, d), lambda ri, hi, ji: (ri, ji, hi, 0)),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
    )(q, k, v, bias, key_mask)


# --------------------------------------------------------------------------- #
# Triangle multiplication
# --------------------------------------------------------------------------- #


def _load_chunks(ref, valid, width, kc):
    """Load a ``[bm, width]`` block as ``width // kc`` f32 column chunks.

    Triton cannot slice a register tensor, so a row that is normalised as a
    whole but contracted in K chunks is kept as a list of chunks from the start.
    """
    return [
        plt.load(ref.at[:, pl.ds(k0, kc)], mask=valid[:, None], other=0.0).astype(jnp.float32)
        for k0 in range(0, width, kc)
    ]


def _ln_chunks(chunks, w_ref, b_ref, eps, kc, dtype):
    """LayerNorm over the concatenation of ``chunks``; returns chunks in ``dtype``."""
    width = sum(c.shape[1] for c in chunks)
    mean = sum(jnp.sum(c, axis=1) for c in chunks) / width
    var = sum(jnp.sum((c - mean[:, None]) ** 2, axis=1) for c in chunks) / width
    rstd = lax.rsqrt(var + eps)
    return [
        ((c - mean[:, None]) * rstd[:, None] * w_ref[pl.ds(i * kc, kc)][None, :]
         + b_ref[pl.ds(i * kc, kc)][None, :]).astype(dtype)
        for i, c in enumerate(chunks)
    ]


def _mm_chunks(xs, w_ref, col0, ncol, kc):
    """``concat(xs) @ W[:, col0:col0 + ncol]`` accumulated over K chunks in f32."""
    acc = None
    for i, xk in enumerate(xs):
        part = _mm(xk, w_ref[pl.ds(i * kc, kc), pl.ds(col0, ncol)])
        acc = part if acc is None else acc + part
    return acc


def _k1_kernel(x_ref, m_ref, lw_ref, lb_ref, wp_ref, wg_ref, a_ref, b_ref, *,
               p_total, bm, hid, eps, kc):
    i = pl.program_id(0)
    valid = (i * bm + jnp.arange(bm)) < p_total
    c = x_ref.shape[-1]
    xn = _ln_chunks(_load_chunks(x_ref, valid, c, kc), lw_ref, lb_ref, eps, kc, wp_ref.dtype)
    msk = plt.load(m_ref, mask=valid, other=0.0).astype(jnp.float32)
    for half, out_ref in ((0, a_ref), (1, b_ref)):
        for c0 in range(0, hid, kc):
            proj = _mm_chunks(xn, wp_ref, half * hid + c0, kc, kc)
            gate = _mm_chunks(xn, wg_ref, half * hid + c0, kc, kc)
            val = jax.nn.sigmoid(gate) * proj * msk[:, None]
            plt.store(out_ref.at[pl.ds(c0, kc), :], val.astype(out_ref.dtype).T,
                      mask=valid[None, :])


@functools.partial(jax.jit, static_argnames=("eps", "bm", "num_warps", "num_stages"))
def trimul_k1(x, mask, ln_w, ln_b, w_p, w_g, *, eps, bm=64, num_warps=4, num_stages=2):
    """x ``[P, C]``, mask ``[P]``, w_p/w_g ``[C, 2H]`` (a half first).

    Returns ``a_t, b_t`` channel-major ``[H, P]``."""
    p_total, c = x.shape
    hid = w_p.shape[1] // 2
    kernel = functools.partial(_k1_kernel, p_total=p_total, bm=bm, hid=hid, eps=eps,
                               kc=min(c, 128))
    full = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))  # noqa: E731
    out = jax.ShapeDtypeStruct((hid, p_total), w_p.dtype)
    return _pallas_call(
        kernel,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=(pl.cdiv(p_total, bm),),
        in_specs=[
            pl.BlockSpec((bm, c), lambda i: (i, 0)),
            pl.BlockSpec((bm,), lambda i: (i,)),
            full((c,)),
            full((c,)),
            full(w_p.shape),
            full(w_g.shape),
        ],
        out_specs=[
            pl.BlockSpec((hid, bm), lambda i: (0, i)),
            pl.BlockSpec((hid, bm), lambda i: (0, i)),
        ],
        out_shape=[out, out],
    )(x, mask, ln_w, ln_b, w_p, w_g)


def _k2_kernel(e_ref, x_ref, liw_ref, lib_ref, low_ref, lob_ref, wpo_ref, wgo_ref,
               o_ref, *, p_total, bm, eps, kc):
    i = pl.program_id(0)
    valid = (i * bm + jnp.arange(bm)) < p_total
    hid = e_ref.shape[0]
    e = [
        plt.load(e_ref.at[pl.ds(k0, kc), :], mask=valid[None, :], other=0.0)
        .astype(jnp.float32).T
        for k0 in range(0, hid, kc)
    ]
    en = _ln_chunks(e, low_ref, lob_ref, eps, kc, wpo_ref.dtype)
    xn = _ln_chunks(_load_chunks(x_ref, valid, x_ref.shape[-1], kc), liw_ref, lib_ref, eps,
                    kc, wgo_ref.dtype)
    for d0 in range(0, o_ref.shape[-1], kc):
        out = _mm_chunks(en, wpo_ref, d0, kc, kc) * jax.nn.sigmoid(
            _mm_chunks(xn, wgo_ref, d0, kc, kc))
        plt.store(o_ref.at[:, pl.ds(d0, kc)], out.astype(o_ref.dtype), mask=valid[:, None])


@functools.partial(jax.jit, static_argnames=("eps", "bm", "num_warps", "num_stages"))
def trimul_k2(e_t, x, ln_in_w, ln_in_b, ln_out_w, ln_out_b, w_po, w_go, *, eps,
              bm=64, num_warps=4, num_stages=2):
    """e_t ``[H, P]`` (the contraction, channel-major), x ``[P, C]``.

    Returns ``[P, D]`` in ``x``'s dtype."""
    hid, p_total = e_t.shape
    c = x.shape[1]
    d = w_po.shape[1]
    kernel = functools.partial(_k2_kernel, p_total=p_total, bm=bm, eps=eps,
                               kc=min(c, hid, d, 128))
    full = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))  # noqa: E731
    return _pallas_call(
        kernel,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=(pl.cdiv(p_total, bm),),
        in_specs=[
            pl.BlockSpec((hid, bm), lambda i: (0, i)),
            pl.BlockSpec((bm, c), lambda i: (i, 0)),
            full((c,)),
            full((c,)),
            full((hid,)),
            full((hid,)),
            full(w_po.shape),
            full(w_go.shape),
        ],
        out_specs=pl.BlockSpec((bm, d), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((p_total, d), x.dtype),
    )(e_t, x, ln_in_w, ln_in_b, ln_out_w, ln_out_b, w_po, w_go)


def triangle_multiplication_k1k2(x, mask, params, *, direction, eps, bm1=64, bm2=64,
                                 warps1=4, warps2=4, stages=2):
    """x ``[N, N, C]`` bf16, mask ``[N, N]``; params in Boltz-2's ``[in, out]``
    layout (``p_in``/``g_in`` ``[C, 2H]``, ``p_out`` ``[H, D]``, ``g_out``
    ``[C, D]``, norms ``scale``/``bias``).  Returns ``[N, N, D]``."""
    n = x.shape[0]
    c = x.shape[-1]
    flat = x.reshape(n * n, c)
    a_t, b_t = trimul_k1(
        flat, mask.reshape(n * n).astype(jnp.float32),
        params["norm_in"]["scale"].astype(jnp.float32),
        params["norm_in"]["bias"].astype(jnp.float32),
        params["p_in"]["kernel"], params["g_in"]["kernel"],
        eps=eps, bm=bm1, num_warps=warps1, num_stages=stages,
    )
    hid = a_t.shape[0]
    a_t = a_t.reshape(hid, n, n)
    b_t = b_t.reshape(hid, n, n)
    if direction == "outgoing":
        dims = (((2,), (2,)), ((0,), (0,)))  # e[c,i,j] = sum_k a[c,i,k] b[c,j,k]
    else:
        dims = (((1,), (1,)), ((0,), (0,)))  # e[c,i,j] = sum_k a[c,k,i] b[c,k,j]
    e_t = lax.dot_general(a_t, b_t, dims, precision=_DEFAULT)
    out = trimul_k2(
        e_t.reshape(hid, n * n), flat,
        params["norm_in"]["scale"].astype(jnp.float32),
        params["norm_in"]["bias"].astype(jnp.float32),
        params["norm_out"]["scale"].astype(jnp.float32),
        params["norm_out"]["bias"].astype(jnp.float32),
        params["p_out"]["kernel"], params["g_out"]["kernel"],
        eps=eps, bm=bm2, num_warps=warps2, num_stages=stages,
    )
    return out.reshape(n, n, -1)


# --------------------------------------------------------------------------- #
# Pair transition
# --------------------------------------------------------------------------- #


def _transition_kernel(x_ref, lw_ref, lb_ref, w1_ref, w2_ref, w3_ref, o_ref, *,
                       p_total, bm, fc, eps, kc):
    i = pl.program_id(0)
    valid = (i * bm + jnp.arange(bm)) < p_total
    dt = w1_ref.dtype
    xn = _ln_chunks(_load_chunks(x_ref, valid, x_ref.shape[-1], kc), lw_ref, lb_ref, eps,
                    kc, dt)
    c_out = o_ref.shape[-1]
    acc = [jnp.zeros((bm, kc), jnp.float32) for _ in range(0, c_out, kc)]
    for f0 in range(0, w1_ref.shape[1], fc):
        h1 = _mm_chunks(xn, w1_ref, f0, fc, kc).astype(dt).astype(jnp.float32)
        h2 = _mm_chunks(xn, w2_ref, f0, fc, kc).astype(dt)
        g = (jax.nn.silu(h1).astype(dt) * h2).astype(dt)
        for j, c0 in enumerate(range(0, c_out, kc)):
            acc[j] = acc[j] + _mm(g, w3_ref[pl.ds(f0, fc), pl.ds(c0, kc)])
    for j, c0 in enumerate(range(0, c_out, kc)):
        plt.store(o_ref.at[:, pl.ds(c0, kc)], acc[j].astype(o_ref.dtype), mask=valid[:, None])


@functools.partial(jax.jit, static_argnames=("eps", "bm", "fc", "num_warps", "num_stages"))
def fused_transition(x, ln_w, ln_b, w1, w2, w3, *, eps, bm=64, fc=128, num_warps=4,
                     num_stages=2):
    """x ``[P, C]``; w1/w2 ``[C, F]``, w3 ``[F, C_out]`` (Boltz-2's fc1/fc2/fc3).

    ``silu(LN(x) W1) * (LN(x) W2) @ W3`` with the operand rounding of the
    autocast path: each projection rounded to the weight dtype, the activation
    evaluated in f32 and rounded, the product rounded before ``W3``."""
    p_total, c = x.shape
    kernel = functools.partial(_transition_kernel, p_total=p_total, bm=bm, fc=fc, eps=eps,
                               kc=min(c, w3.shape[1], 128))
    full = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))  # noqa: E731
    return _pallas_call(
        kernel,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=(pl.cdiv(p_total, bm),),
        in_specs=[
            pl.BlockSpec((bm, c), lambda i: (i, 0)),
            full((c,)),
            full((c,)),
            full(w1.shape),
            full(w2.shape),
            full(w3.shape),
        ],
        out_specs=pl.BlockSpec((bm, w3.shape[1]), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((p_total, w3.shape[1]), x.dtype),
    )(x, ln_w, ln_b, w1, w2, w3)
