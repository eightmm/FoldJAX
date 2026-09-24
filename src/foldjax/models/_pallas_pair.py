"""Pallas-Triton kernels for the pair stack: triangle multiplication and transitions.

Two kernel families replace cuEquivariance's fused triangle multiplication and
the tokamax/XLA SwiGLU transition when a port is asked for the ``pallas``
backend. Both are opt-in; nothing here is a default.

Triangle multiplication is split the way FlashPairformer splits it. ``K1``
normalises the pair, applies the sigmoid-gated input projections and the mask,
and writes ``a`` and ``b`` channel-major (``[H, pixels]``), so the contraction is
``H`` plain batched GEMMs that XLA hands to cuBLAS. ``K2`` normalises the
contraction, applies the output projection and multiplies by the output gate,
which it recomputes from ``x`` rather than storing. Neither the ``[N, N, 2H]``
projection nor the gate is ever written in pixel-major form.

The transition is one kernel. LayerNorm, ``silu(x W1) * (x W2)`` and ``W3`` run
over one block of rows, with the hidden width processed in slices, so the
widened ``[rows, F]`` form never reaches memory. It is used only where it was
measured to win: plain transitions no wider than :data:`TRANSITION_MAX_WIDTH`,
which are the pair and MSA transitions (see :func:`foldjax.models._glu.site_backend`).
A unit-only form for the other GLU sites lost on the card -- 7x tokamax on
Protenix's float32 diffusion GLU, 1.1-1.2x at the 384-wide single transition
(``foldjax-bench/kernel-shootout-20260924/out/slurm-2395.out``) -- and was
removed.

Measured on one RTX PRO 6000 Blackwell (sm_120) against the released paths at
c_z 128, bf16 (``foldjax-bench/kernel-shootout-20260924``, job 2386). Triangle
multiplication ran 0.47-0.65x cuEquivariance's time at 1,003-4,888 tokens, with
the same max-abs error against a float32 reference and a 25-30 % lower peak.
The pair transition ran 0.34-0.49x Boltz-2's row-chunked tokamax GLU, with the
same error, no temp buffer and a 40 % lower peak. At OpenDDE's c_z 384 the
multiplication only ties cuEquivariance and the transition loses; the block
sizes below were pinned at 128.

Rounding follows the autocast path the ports reproduce. LayerNorm runs in
float32. Each projection accumulates in float32 and is rounded to the weight
dtype where a torch Linear would round it. Activations run in float32. The
contraction takes and returns the weight dtype.

Off a GPU the kernels refuse to run instead of falling back.
``INTERPRET = True`` runs them in Pallas interpret mode; that is for tests and
is not a supported execution path.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

#: Pallas interpret mode, for CPU tests only.
INTERPRET = False

#: Block sizes pinned from the shootout (job 2386). Multiplication: 32 pixels
#: and 4 warps, the one configuration that fits 99 KB of shared memory at
#: c_z 384 and within 3 % of the best at 128. Transition: 64 rows, hidden
#: slices of 64, 4 warps. 128 rows with 4 warps spilled registers and ran 12x
#: slower, and 64 rows with 128-wide slices was erratic (0.81-1.04x). Wider
#: operands take half the rows and twice the warps; at c_z 384 that is the only
#: transition configuration that did not spill.
_MUL_ROWS, _MUL_WARPS = 32, 4
_TRANSITION_ROWS, _TRANSITION_SLICE, _TRANSITION_WARPS = 64, 64, 4
_WIDE = 128

#: The widest transition the fused kernel runs. Measured faster at c_z 128
#: (pair) and c_m 64 (MSA); slower than XLA at 384 (OpenDDE's pair, every
#: port's single transition).
TRANSITION_MAX_WIDTH = 128


def _require_gpu(what: str, instead: str) -> None:
    if INTERPRET or jax.default_backend() == "gpu":
        return
    msg = (
        f"{what} runs a Pallas-Triton kernel that needs a CUDA GPU, and this "
        f"process's default JAX backend is {jax.default_backend()!r}; {instead}"
    )
    raise ValueError(msg)


def _pallas_call(kernel, *, num_warps: int, **kwargs):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    if INTERPRET:
        return pl.pallas_call(kernel, interpret=True, **kwargs)
    return pl.pallas_call(
        kernel,
        compiler_params=plt.CompilerParams(num_warps=num_warps, num_stages=2),
        **kwargs,
    )


def _slice_width(*widths: int, dtype=jnp.bfloat16) -> int:
    """The largest power-of-two slice that divides every width.

    Triton tensors must have power-of-two extents. A row is normalised whole
    but contracted in slices of this width, so every width must divide into
    them. The cap is 128 for half-precision weights, which is what the
    shootout measured. It is 64 for float32 weights: each dot stages both
    operand tiles in shared memory for two pipeline stages, and at 128 a
    float32 32-row slice needs 96 KB of this card's 99 KB.
    """

    cap = 128 if jnp.dtype(dtype).itemsize <= 2 else 64
    for size in (128, 64, 32, 16):
        if size <= cap and all(w % size == 0 for w in widths):
            return size
    msg = f"the Pallas pair kernels need widths divisible by 16; got {widths}"
    raise ValueError(msg)


def _precision(dtype) -> lax.Precision:
    """Tensor-core inputs for half precision; the JAX policy for float32."""

    if jnp.dtype(dtype) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        return lax.Precision.DEFAULT
    policy = jax.config.jax_default_matmul_precision
    if policy in ("highest", "float32", "F32_F32_F32"):
        return lax.Precision.HIGHEST
    return lax.Precision.DEFAULT  # TF32 in the Triton lowering


def _mm(a, b, precision):
    return lax.dot_general(
        a,
        b,
        (((1,), (0,)), ((), ())),
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def _load_slices(ref, valid, width, kc):
    """Load a ``[rows, width]`` block as float32 slices of ``kc`` columns.

    Triton cannot slice a tensor held in registers, so a row that is normalised
    whole but contracted in slices is kept as slices from the load onwards.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    return [
        plt.load(ref.at[:, pl.ds(k0, kc)], mask=valid[:, None], other=0.0).astype(
            jnp.float32
        )
        for k0 in range(0, width, kc)
    ]


def _norm_slices(slices, w_ref, b_ref, eps, kc, dtype):
    """LayerNorm over the concatenated slices, returned as slices in ``dtype``."""
    from jax.experimental import pallas as pl

    width = sum(s.shape[1] for s in slices)
    mean = sum(jnp.sum(s, axis=1) for s in slices) / width
    var = sum(jnp.sum((s - mean[:, None]) ** 2, axis=1) for s in slices) / width
    rstd = lax.rsqrt(var + eps)
    return [
        (
            (s - mean[:, None]) * rstd[:, None] * w_ref[pl.ds(i * kc, kc)][None, :]
            + b_ref[pl.ds(i * kc, kc)][None, :]
        ).astype(dtype)
        for i, s in enumerate(slices)
    ]


def _mm_slices(xs, w_ref, col0, ncol, kc, precision):
    """``concat(xs) @ W[:, col0:col0 + ncol]``, accumulated over the slices."""
    from jax.experimental import pallas as pl

    acc = None
    for i, xk in enumerate(xs):
        part = _mm(xk, w_ref[pl.ds(i * kc, kc), pl.ds(col0, ncol)], precision)
        acc = part if acc is None else acc + part
    return acc


def _full(shape):
    from jax.experimental import pallas as pl

    return pl.BlockSpec(shape, lambda i: (0,) * len(shape))


# --------------------------------------------------------------------------- #
# Triangle multiplication
# --------------------------------------------------------------------------- #


def _k1_kernel(
    x_ref,
    m_ref,
    lw_ref,
    lb_ref,
    wp_ref,
    wg_ref,
    a_ref,
    b_ref,
    *,
    pixels,
    rows,
    hidden,
    eps,
    kc,
    precision,
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    valid = (pl.program_id(0) * rows + jnp.arange(rows)) < pixels
    xn = _norm_slices(
        _load_slices(x_ref, valid, x_ref.shape[-1], kc),
        lw_ref,
        lb_ref,
        eps,
        kc,
        wp_ref.dtype,
    )
    mask = plt.load(m_ref, mask=valid, other=0.0).astype(jnp.float32)
    for half, out_ref in ((0, a_ref), (1, b_ref)):
        for c0 in range(0, hidden, kc):
            proj = _mm_slices(xn, wp_ref, half * hidden + c0, kc, kc, precision)
            gate = _mm_slices(xn, wg_ref, half * hidden + c0, kc, kc, precision)
            value = jax.nn.sigmoid(gate) * proj * mask[:, None]
            plt.store(
                out_ref.at[pl.ds(c0, kc), :],
                value.astype(out_ref.dtype).T,
                mask=valid[None, :],
            )


def _k2_kernel(
    e_ref,
    x_ref,
    liw_ref,
    lib_ref,
    low_ref,
    lob_ref,
    wpo_ref,
    wgo_ref,
    o_ref,
    *,
    pixels,
    rows,
    eps,
    kc,
    precision,
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    valid = (pl.program_id(0) * rows + jnp.arange(rows)) < pixels
    e = [
        plt.load(e_ref.at[pl.ds(k0, kc), :], mask=valid[None, :], other=0.0)
        .astype(jnp.float32)
        .T
        for k0 in range(0, e_ref.shape[0], kc)
    ]
    en = _norm_slices(e, low_ref, lob_ref, eps, kc, wpo_ref.dtype)
    xn = _norm_slices(
        _load_slices(x_ref, valid, x_ref.shape[-1], kc),
        liw_ref,
        lib_ref,
        eps,
        kc,
        wgo_ref.dtype,
    )
    for d0 in range(0, o_ref.shape[-1], kc):
        out = _mm_slices(en, wpo_ref, d0, kc, kc, precision) * jax.nn.sigmoid(
            _mm_slices(xn, wgo_ref, d0, kc, kc, precision)
        )
        plt.store(
            o_ref.at[:, pl.ds(d0, kc)], out.astype(o_ref.dtype), mask=valid[:, None]
        )


@functools.partial(jax.jit, static_argnames=("direction", "eps"))
def _triangle_multiplication(
    x, mask, ln_in, w_p, w_g, ln_out, w_po, w_go, *, direction, eps
):
    from jax.experimental import pallas as pl

    *lead, n, _, c = x.shape
    batch = 1
    for size in lead:
        batch *= size
    hidden = w_p.shape[1] // 2
    d_out = w_po.shape[1]
    kc = _slice_width(c, hidden, d_out, dtype=w_p.dtype)
    precision = _precision(w_p.dtype)
    pixels = batch * n * n
    flat = x.reshape(pixels, c)
    grid = (pl.cdiv(pixels, _MUL_ROWS),)
    ab = jax.ShapeDtypeStruct((hidden, pixels), w_p.dtype)
    a_t, b_t = _pallas_call(
        functools.partial(
            _k1_kernel,
            pixels=pixels,
            rows=_MUL_ROWS,
            hidden=hidden,
            eps=eps,
            kc=kc,
            precision=precision,
        ),
        num_warps=_MUL_WARPS,
        grid=grid,
        in_specs=[
            pl.BlockSpec((_MUL_ROWS, c), lambda i: (i, 0)),
            pl.BlockSpec((_MUL_ROWS,), lambda i: (i,)),
            _full((c,)),
            _full((c,)),
            _full(w_p.shape),
            _full(w_g.shape),
        ],
        out_specs=[
            pl.BlockSpec((hidden, _MUL_ROWS), lambda i: (0, i)),
            pl.BlockSpec((hidden, _MUL_ROWS), lambda i: (0, i)),
        ],
        out_shape=[ab, ab],
    )(flat, mask.reshape(pixels).astype(jnp.float32), ln_in[0], ln_in[1], w_p, w_g)
    a_t = a_t.reshape(hidden, batch, n, n)
    b_t = b_t.reshape(hidden, batch, n, n)
    if direction == "outgoing":  # e[c,i,j] = sum_k a[c,i,k] b[c,j,k]
        dims = (((3,), (3,)), ((0, 1), (0, 1)))
    else:  # e[c,i,j] = sum_k a[c,k,i] b[c,k,j]
        dims = (((2,), (2,)), ((0, 1), (0, 1)))
    e_t = lax.dot_general(a_t, b_t, dims, precision=precision)
    out = _pallas_call(
        functools.partial(
            _k2_kernel,
            pixels=pixels,
            rows=_MUL_ROWS,
            eps=eps,
            kc=kc,
            precision=precision,
        ),
        num_warps=_MUL_WARPS,
        grid=grid,
        in_specs=[
            pl.BlockSpec((hidden, _MUL_ROWS), lambda i: (0, i)),
            pl.BlockSpec((_MUL_ROWS, c), lambda i: (i, 0)),
            _full((c,)),
            _full((c,)),
            _full((hidden,)),
            _full((hidden,)),
            _full(w_po.shape),
            _full(w_go.shape),
        ],
        out_specs=pl.BlockSpec((_MUL_ROWS, d_out), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((pixels, d_out), x.dtype),
    )(
        e_t.reshape(hidden, pixels),
        flat,
        ln_in[0],
        ln_in[1],
        ln_out[0],
        ln_out[1],
        w_po,
        w_go,
    )
    return out.reshape(*lead, n, n, d_out)


def triangle_multiplication(
    x: jnp.ndarray,
    *,
    direction: str,
    mask: jnp.ndarray,
    norm_in: tuple[jnp.ndarray, jnp.ndarray | None],
    p_in: tuple[jnp.ndarray, jnp.ndarray | None],
    g_in: tuple[jnp.ndarray, jnp.ndarray | None],
    norm_out: tuple[jnp.ndarray, jnp.ndarray | None],
    p_out: tuple[jnp.ndarray, jnp.ndarray | None],
    g_out: tuple[jnp.ndarray, jnp.ndarray | None],
    eps: float,
) -> jnp.ndarray:
    """The triangle multiplicative update, with the calling convention of
    :func:`foldjax.models._cueq.fused_triangle_multiplication`.

    Weights arrive in the torch ``[out, in]`` layout that function takes
    (``p_in``/``g_in`` stacked ``a`` then ``b``, ``(2H, C)``), so a port switches
    kernels without repacking. ``x`` is ``[..., N, N, C]`` and ``mask`` is
    broadcast to ``x.shape[:-1]``. Leading axes run as one batch of pixels and
    one batched contraction.

    Linear biases are refused rather than dropped. No released checkpoint has
    one, and the kernels have no place for them. The layer norms must be affine,
    as cuEquivariance requires.
    """

    if direction not in ("outgoing", "incoming"):
        msg = f"direction must be 'outgoing' or 'incoming'; got {direction!r}"
        raise ValueError(msg)
    named = {"p_in": p_in, "g_in": g_in, "p_out": p_out, "g_out": g_out}
    biased = [name for name, (_, bias) in named.items() if bias is not None]
    if biased:
        msg = f"the Pallas triangle multiplication is bias-free; {biased} carry a bias"
        raise ValueError(msg)
    if any(t is None for pair in (norm_in, norm_out) for t in pair):
        raise ValueError("the Pallas triangle multiplication needs affine layer norms")
    if x.ndim < 3 or x.shape[-3] != x.shape[-2]:
        raise ValueError("triangle multiplication requires square pair axes")
    _require_gpu(
        "triangle multiplication backend 'pallas'",
        "select 'cueq' or 'xla' for this process",
    )
    f32 = lambda t: t.astype(jnp.float32)  # noqa: E731
    return _triangle_multiplication(
        x,
        jnp.broadcast_to(mask, x.shape[:-1]),
        (f32(norm_in[0]), f32(norm_in[1])),
        p_in[0].T,
        g_in[0].T,
        (f32(norm_out[0]), f32(norm_out[1])),
        p_out[0].T,
        g_out[0].T,
        direction=direction,
        eps=float(eps),
    )


# --------------------------------------------------------------------------- #
# Transition and GLU
# --------------------------------------------------------------------------- #


def _glu_slices(xs, w1_ref, w2_ref, f0, fc, kc, activation, dt, precision):
    """``activation(x W1) * (x W2)`` for one hidden slice, rounded like autocast."""
    h1 = _mm_slices(xs, w1_ref, f0, fc, kc, precision).astype(dt).astype(jnp.float32)
    h2 = _mm_slices(xs, w2_ref, f0, fc, kc, precision).astype(dt)
    return (activation(h1).astype(dt) * h2).astype(dt)


def _transition_kernel(
    x_ref,
    lw_ref,
    lb_ref,
    w1_ref,
    w2_ref,
    w3_ref,
    o_ref,
    *,
    total,
    rows,
    fc,
    eps,
    kc,
    precision,
    update_dtype=None,
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    valid = (pl.program_id(0) * rows + jnp.arange(rows)) < total
    dt = w1_ref.dtype
    xn = _norm_slices(
        _load_slices(x_ref, valid, x_ref.shape[-1], kc), lw_ref, lb_ref, eps, kc, dt
    )
    c_out = o_ref.shape[-1]
    acc = [jnp.zeros((rows, kc), jnp.float32) for _ in range(0, c_out, kc)]
    for f0 in range(0, w1_ref.shape[1], fc):
        g = _glu_slices(xn, w1_ref, w2_ref, f0, fc, kc, jax.nn.silu, dt, precision)
        for j, c0 in enumerate(range(0, c_out, kc)):
            acc[j] = acc[j] + _mm(g, w3_ref[pl.ds(f0, fc), pl.ds(c0, kc)], precision)
    for j, c0 in enumerate(range(0, c_out, kc)):
        if update_dtype is None:
            value = acc[j].astype(o_ref.dtype)
        else:
            # Residual mode: ``x + update`` as the unfused add computes it -- the
            # update rounded to its own dtype, both widened to float32, added
            # once and rounded once to the result. ``x`` is reloaded rather than
            # kept from the norm, so no extra slice stays live across the loop.
            rows_x = plt.load(
                x_ref.at[:, pl.ds(c0, kc)], mask=valid[:, None], other=0.0
            ).astype(jnp.float32)
            update = acc[j].astype(update_dtype).astype(jnp.float32)
            value = (rows_x + update).astype(o_ref.dtype)
        plt.store(o_ref.at[:, pl.ds(c0, kc)], value, mask=valid[:, None])


def _row_config(width: int) -> tuple[int, int]:
    """Rows per program and warps: wider operands take fewer rows, more warps."""
    if width <= _WIDE:
        return _TRANSITION_ROWS, _TRANSITION_WARPS
    return _TRANSITION_ROWS // 2, 2 * _TRANSITION_WARPS


@functools.partial(jax.jit, static_argnames=("eps", "store_dtype", "update_dtype"))
def _transition(x, ln_w, ln_b, w1, w2, w3, *, eps, store_dtype, update_dtype=None):
    from jax.experimental import pallas as pl

    c = x.shape[-1]
    flat = x.reshape(-1, c)
    total = flat.shape[0]
    kc = _slice_width(c, w3.shape[1], dtype=w1.dtype)
    fc = min(_TRANSITION_SLICE, _slice_width(w1.shape[1], dtype=w1.dtype))
    rows, warps = _row_config(c)
    out = _pallas_call(
        functools.partial(
            _transition_kernel,
            total=total,
            rows=rows,
            fc=fc,
            eps=eps,
            kc=kc,
            precision=_precision(w1.dtype),
            update_dtype=update_dtype,
        ),
        num_warps=warps,
        grid=(pl.cdiv(total, rows),),
        in_specs=[
            pl.BlockSpec((rows, c), lambda i: (i, 0)),
            _full((c,)),
            _full((c,)),
            _full(w1.shape),
            _full(w2.shape),
            _full(w3.shape),
        ],
        out_specs=pl.BlockSpec((rows, w3.shape[1]), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((total, w3.shape[1]), store_dtype),
    )(flat, ln_w, ln_b, w1, w2, w3)
    return out.reshape(*x.shape[:-1], w3.shape[1])


def transition(
    x: jnp.ndarray,
    norm: tuple[jnp.ndarray, jnp.ndarray],
    w_gate: jnp.ndarray,
    w_value: jnp.ndarray,
    w_out: jnp.ndarray,
    *,
    eps: float,
    out_dtype=None,
    residual: bool = False,
) -> jnp.ndarray:
    """``(silu(LN(x) W_gate) * (LN(x) W_value)) W_out`` as one kernel.

    Weights are ``[in, out]``. The normalised operand is narrowed to their dtype,
    as an autocast Linear narrows it. The result is returned in ``out_dtype``,
    ``x``'s dtype by default. The norm must be affine and the projections
    bias-free; the caller checks the latter, since only it knows its parameter
    type.

    ``residual=True`` returns ``x + transition(x)`` instead, bit for bit what the
    unfused add computes, in ``result_type(x, out_dtype)``. A custom call's
    output cannot fuse into the add that consumes it, so without this the
    update is a whole buffer of its own beside the residual and the sum:
    +1,138 MiB temp in Boltz-2's MSA layer at 1,003 tokens x 8,808 rows
    (``foldjax-bench/kernel-shootout-20260924``, job 2424), where the released
    row-chunked path never materialises it. The result is a fresh buffer:
    aliasing it to ``x`` made XLA copy the residual first, +3,885 MiB in the
    same layer (job 2435).
    """

    if norm[0] is None or norm[1] is None:
        raise ValueError("the Pallas transition needs an affine layer norm")
    _require_gpu(
        "glu_backend='pallas'",
        "pass the option glu_backend=xla (CLI: --option glu_backend=xla) to run "
        "the XLA unit here",
    )
    f32 = lambda t: t.astype(jnp.float32)  # noqa: E731
    out_dtype = x.dtype if out_dtype is None else jnp.dtype(out_dtype)
    # The kernel rounds its float32 accumulator to ``x``'s dtype, then the
    # caller's. When ``x`` is float32 that is one rounding, so the kernel stores
    # ``out_dtype`` itself: a float32 MSA would otherwise stage a float32 copy
    # of the whole output only to narrow it (+2.1 GiB at 1,003 x 8,808,
    # foldjax-bench/kernel-shootout-20260924 job 2407).
    one_rounding = x.dtype == out_dtype or x.dtype == jnp.float32
    update_dtype = out_dtype if one_rounding else x.dtype
    if residual:
        if w_out.shape[-1] != x.shape[-1]:
            msg = "a residual transition must return x's width"
            raise ValueError(msg)
        return _transition(
            x,
            f32(norm[0]),
            f32(norm[1]),
            w_gate,
            w_value,
            w_out,
            eps=float(eps),
            store_dtype=jnp.result_type(x.dtype, out_dtype),
            update_dtype=update_dtype,
        )
    out = _transition(
        x,
        f32(norm[0]),
        f32(norm[1]),
        w_gate,
        w_value,
        w_out,
        eps=float(eps),
        store_dtype=update_dtype,
    )
    return out.astype(out_dtype)
