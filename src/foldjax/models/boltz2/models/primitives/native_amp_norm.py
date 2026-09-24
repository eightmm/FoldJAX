"""Native CUDA AMP normalization boundaries shared by Boltz and ESMFold2.

The vector-4 Welford/FMA path matches the observed pinned CUDA norm panels.
It is limited to the observed widths; CPU, TPU, context parallelism and other
widths retain the ordinary JAX path. FP32 model inference does not select it.
ESMFold2 reuses the private CUDA helper at width256 with its own fallback;
its captured first norm is bitwise exact, but full-block parity remains open.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_mesh
from foldjax.models.boltz2.models.primitives._common import layer_norm


def amp_layer_norm(x, scale, bias, eps):
    x, scale, bias = (a.astype(jnp.float32) for a in (x, scale, bias))
    if x.shape[-1] not in (16, 64, 128, 256) or cp_mesh() is not None:
        return layer_norm(x, scale, bias, eps)
    return jax.lax.platform_dependent(
        x,
        scale,
        bias,
        cuda=lambda x, s, b: _cuda_layer_norm(x, s, b, eps)[0],
        default=lambda x, s, b: layer_norm(x, s, b, eps),
    )


def amp_affine(x, scale, bias, out_dtype=jnp.float32):
    """Apply the pinned CUDA affine FMA and deliver the result in ``out_dtype``.

    The FMA always runs in FP32, on every path. ``out_dtype`` only chooses the
    width the result is stored at, and narrowing it is bit-exact by
    construction: FP32 storage followed by one round-nearest-even convert and
    direct ``out_dtype`` storage perform the same FMA and the same single RNE
    rounding. Emitting the narrow width from the kernel removes an FP32 buffer
    that can never be fused away, because a ``pallas_call`` output is a real
    buffer and a following dot cannot consume it in place.

    The default stays FP32 so any future caller keeps the historical result.
    ESMFold2 and OpenFold3 share this module but call ``_cuda_layer_norm``, not
    this function. Every path honours ``out_dtype``, including the non-CUDA
    fallbacks, so the delivered width is the same on all platforms.
    """
    if (
        x.shape[-1] not in (16, 128, 256)
        or x.dtype != jnp.float32
        or cp_mesh() is not None
    ):
        return (x * scale + bias).astype(out_dtype)
    scale, bias = scale.astype(jnp.float32), bias.astype(jnp.float32)
    return jax.lax.platform_dependent(
        x,
        scale,
        bias,
        cuda=lambda x, s, b: _cuda_affine(x, s, b, out_dtype),
        default=lambda x, s, b: (x * s + b).astype(out_dtype),
    )


def _cuda_affine(x, scale, bias, out_dtype=jnp.float32):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    width = x.shape[-1]
    rows = math.prod(x.shape[:-1])

    def kernel(x_ref, s_ref, b_ref, out_ref):
        value, s, b = x_ref[0, :], s_ref[:], b_ref[:]
        out_ref[0, :] = pt.elementwise_inline_asm(
            "fma.rn.f32 $0, $1, $2, $3;",
            args=(s, value, b),
            constraints="=f,f,f,f",
            pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct((width,), jnp.float32)],
        )[0].astype(out_dtype)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, width), out_dtype),
        grid=(rows,),
        in_specs=(
            pl.BlockSpec((1, width), lambda i: (i, 0)),
            pl.BlockSpec((width,), lambda i: (0,)),
            pl.BlockSpec((width,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((1, width), lambda i: (i, 0)),
        compiler_params=pt.CompilerParams(num_warps=4),
    )(x.reshape(rows, width), scale, bias)
    return out.reshape(x.shape)


def _cuda_layer_norm(x, scale, bias, eps=1e-5, out_dtype=jnp.float32):
    """The pinned CUDA Welford/FMA layer norm, delivered in ``out_dtype``.

    ``out_dtype`` chooses only the width the normalised result is stored at.
    The reduction and the affine FMA always run in FP32, so narrowing it is
    bit-exact by the argument ``amp_affine`` records: FP32 storage followed by
    one round-nearest-even convert and a direct narrow store perform the same
    FMA and the same single rounding. Emitting the narrow width from the
    kernel is what removes the FP32 buffer rather than shortening its life,
    because a ``pallas_call`` output is a real buffer that a following convert
    cannot be fused into. ``mean`` and ``rstd`` stay FP32; they are a residual
    output, not a stored activation.

    The default keeps every existing caller's result: Boltz-2 and OpenFold3
    ask for FP32 here, and ESMFold2 asks for BF16 at the three pair norms
    whose only consumer rounds to BF16 itself.

    Every call with the same shapes, ``eps`` and ``out_dtype`` shares one
    trace. Pallas keeps no trace cache and the kernel is a fresh closure per
    call, so each call site used to re-trace the ~100 inline-asm body and
    re-lower it to Triton (ESMFold2: ~25 s per warm process). The shared
    function is a ``jit`` inlined back into the caller at trace time, so the
    caller's program holds the same ``pallas_call`` equations as before, now
    with one kernel jaxpr that the lowering cache lowers once.
    """
    width = x.shape[-1]
    if width not in (16, 64, 128, 256) or x.dtype != jnp.float32:
        raise ValueError("native CUDA norm requires FP32 and width 16, 64, 128 or 256")
    try:
        hash(eps)
    except TypeError:
        # A traced or array ``eps`` cannot key a trace; build it in place.
        return _cuda_layer_norm_body(x, scale, bias, eps, out_dtype)
    return _cuda_layer_norm_shared(
        x, scale, bias, eps=eps, out_dtype=jnp.dtype(out_dtype)
    )


@functools.partial(jax.jit, static_argnames=("eps", "out_dtype"), inline=True)
def _cuda_layer_norm_shared(x, scale, bias, *, eps, out_dtype):
    return _cuda_layer_norm_body(x, scale, bias, eps, out_dtype)


#: Elements one kernel program normalises: rows are blocked so each program
#: covers ``_NORM_BLOCK_ELEMENTS // width`` of them (a power of two, at most the
#: row count rounded up). The per-row instruction sequence does not depend on
#: it; see `_cuda_layer_norm_body`.
#:
#: Chosen from the RTX PRO 6000 sweep (job 2384, 64,516 and 1,006,009 rows):
#: 1024 elements beat or matched one row per program at every width with a
#: float32 output (width 256: 1.18 -> 0.80 ms and 5.78 -> 2.63 ms), and
#: larger blocks collapse (width 256 at 4096: up to 16x slower). A bfloat16
#: output at width 256 -- ESMFold2's pair norms -- gained nothing at any
#: block (4 rows: +2-4%), so it keeps one row per program.
_NORM_BLOCK_ELEMENTS = 1024


def _norm_row_block(rows: int, width: int, out_dtype=jnp.float32) -> int:
    if width == 256 and jnp.dtype(out_dtype) != jnp.dtype(jnp.float32):
        return 1
    block = max(1, _NORM_BLOCK_ELEMENTS // width)
    return min(block, 1 << max(0, rows - 1).bit_length())


def _cuda_layer_norm_body(x, scale, bias, eps, out_dtype, block=None):
    """One ``pallas_call`` normalising ``block`` rows per program.

    Every instruction is element-wise inline PTX (``pack=1``) and the Welford
    combine tree runs along the row's own lanes, so adding a leading row axis
    gives each row exactly the instruction sequence the one-row program ran:
    the same operands, in the same order, to the same single rounding. When
    the rows do not fill the last block, rows past the end are loaded as
    zeros under a mask and never stored, which changes no stored row; when
    they do, no mask is emitted. One row per program ran at ~10% of HBM
    bandwidth (ESMFold2 at 1,003 tokens: 25% of all kernel time).
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    width = x.shape[-1]
    shape = x.shape
    rows = math.prod(shape[:-1])
    if block is None:
        block = _norm_row_block(rows, width, out_dtype)
    tail = rows % block != 0

    def kernel(x_ref, scale_ref, bias_ref, eps_ref, out_ref, mean_ref, rstd_ref):
        def op(instruction, *args):
            args = jnp.broadcast_arrays(*[jnp.asarray(a, jnp.float32) for a in args])
            registers = ", ".join(f"${i}" for i in range(len(args) + 1))
            return pt.elementwise_inline_asm(
                f"{instruction} {registers};",
                args=args,
                constraints="=f" + ",f" * len(args),
                pack=1,
                result_shape_dtypes=[jax.ShapeDtypeStruct(args[0].shape, jnp.float32)],
            )[0]

        if tail:
            row = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
            valid = (row < rows)[:, None]
            wide = jnp.broadcast_to(valid, (block, width))
            values = pt.load(x_ref, mask=wide, other=0.0)
        else:
            values = x_ref[...]
        vectors = values.reshape(block, width // 4, 4)
        mean = jnp.zeros((block, width // 4), jnp.float32)
        variance = jnp.zeros_like(mean)
        components = jax.lax.split(vectors, (1, 1, 1, 1), axis=2)
        for index, component in enumerate(components):
            value = component.reshape(block, width // 4)
            delta = op("sub.rn.f32", value, mean)
            updated = op("fma.rn.f32", delta, 1.0 / (index + 1), mean)
            variance = op(
                "fma.rn.f32", delta, op("sub.rn.f32", value, updated), variance
            )
            mean = updated
        count = 4

        def combine(ma, va, mb, vb, count):
            # CUDA cuWelfordCombine(wd, wdB) calls the first operand dataB.
            delta = op("sub.rn.f32", ma, mb)
            combined_mean = op("fma.rn.f32", 0.5, mb, op("mul.rn.f32", 0.5, ma))
            correction = op("mul.rn.f32", op("mul.rn.f32", delta, delta), count)
            combined_var = op("fma.rn.f32", correction, 0.5, op("add.rn.f32", vb, va))
            return combined_mean, combined_var

        warps = 2 if width == 256 else 1
        lanes = min(32, width // 4)
        mean = mean.reshape(block, warps, lanes)
        variance = variance.reshape(block, warps, lanes)
        while lanes > 1:
            offset = lanes // 2
            ma, mb = jax.lax.split(mean, (offset, offset), axis=2)
            va, vb = jax.lax.split(variance, (offset, offset), axis=2)
            mean, variance = combine(ma, va, mb, vb, count)
            count *= 2
            lanes = offset
        if warps == 2:
            ma, mb = jax.lax.split(mean, (1, 1), axis=1)
            va, vb = jax.lax.split(variance, (1, 1), axis=1)
            mean, variance = combine(ma, va, mb, vb, count)
        mean, variance = mean.reshape(block, 1), variance.reshape(block, 1)
        rstd = op(
            "rsqrt.approx.ftz.f32",
            op("add.rn.f32", op("mul.rn.f32", variance, 1.0 / width), eps_ref[0]),
        )
        normed = op("mul.rn.f32", rstd, op("sub.rn.f32", values, mean))
        out = op("fma.rn.f32", scale_ref[:], normed, bias_ref[:]).astype(out_dtype)
        if tail:
            pt.store(out_ref, out, mask=wide)
            pt.store(mean_ref, mean, mask=valid)
            pt.store(rstd_ref, rstd, mask=valid)
        else:
            out_ref[...], mean_ref[...], rstd_ref[...] = out, mean, rstd

    result = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, width), out_dtype),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        ),
        grid=(pl.cdiv(rows, block),),
        in_specs=(
            pl.BlockSpec((block, width), lambda i: (i, 0)),
            pl.BlockSpec((width,), lambda i: (0,)),
            pl.BlockSpec((width,), lambda i: (0,)),
            pl.BlockSpec((1,), lambda i: (0,)),
        ),
        out_specs=(
            pl.BlockSpec((block, width), lambda i: (i, 0)),
            pl.BlockSpec((block, 1), lambda i: (i, 0)),
            pl.BlockSpec((block, 1), lambda i: (i, 0)),
        ),
        compiler_params=pt.CompilerParams(num_warps=4),
    )(x.reshape(rows, width), scale, bias, jnp.asarray(eps, jnp.float32).reshape(1))
    return (
        result[0].reshape(shape),
        result[1].reshape(*shape[:-1], 1),
        result[2].reshape(*shape[:-1], 1),
    )
