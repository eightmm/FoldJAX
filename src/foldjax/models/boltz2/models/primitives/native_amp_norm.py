"""Native CUDA AMP normalization boundaries shared by Boltz and ESMFold2.

The vector-4 Welford/FMA path matches the observed pinned CUDA norm panels.
It is limited to the observed widths; CPU, TPU, context parallelism and other
widths retain the ordinary JAX path. FP32 model inference does not select it.
ESMFold2 reuses the private CUDA helper at width256 with its own fallback;
its captured first norm is bitwise exact, but full-block parity remains open.
"""

from __future__ import annotations

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


def _cuda_layer_norm(x, scale, bias, eps=1e-5):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    width = x.shape[-1]
    if width not in (16, 64, 128, 256) or x.dtype != jnp.float32:
        raise ValueError("native CUDA norm requires FP32 and width 16, 64, 128 or 256")
    shape = x.shape
    rows = math.prod(shape[:-1])

    def kernel(x_ref, scale_ref, bias_ref, eps_ref, out_ref, mean_ref, rstd_ref):
        def op(instruction, *args):
            args = jnp.broadcast_arrays(*[jnp.asarray(a, jnp.float32) for a in args])
            result_shape = args[0].shape
            if not result_shape:
                # Pallas Triton's inline-asm lowering needs IR values; a
                # scalar literal is otherwise forwarded as a Python scalar.
                args = [jnp.broadcast_to(a, (1,)) for a in args]
            registers = ", ".join(f"${i}" for i in range(len(args) + 1))
            return pt.elementwise_inline_asm(
                f"{instruction} {registers};",
                args=args,
                constraints="=f" + ",f" * len(args),
                pack=1,
                result_shape_dtypes=[jax.ShapeDtypeStruct(args[0].shape, jnp.float32)],
            )[0].reshape(result_shape)

        values = x_ref[0, :]
        vectors = values.reshape(width // 4, 4)
        mean = jnp.zeros((width // 4,), jnp.float32)
        variance = jnp.zeros_like(mean)
        components = jax.lax.split(vectors, (1, 1, 1, 1), axis=1)
        for index, component in enumerate(components):
            value = component.reshape(width // 4)
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
        mean, variance = mean.reshape(warps, lanes), variance.reshape(warps, lanes)
        while lanes > 1:
            offset = lanes // 2
            ma, mb = jax.lax.split(mean, (offset, offset), axis=1)
            va, vb = jax.lax.split(variance, (offset, offset), axis=1)
            mean, variance = combine(ma, va, mb, vb, count)
            count *= 2
            lanes = offset
        if warps == 2:
            ma, mb = jax.lax.split(mean, (1, 1), axis=0)
            va, vb = jax.lax.split(variance, (1, 1), axis=0)
            mean, variance = combine(ma, va, mb, vb, count)
        mean, variance = mean.reshape(()), variance.reshape(())
        rstd = op(
            "rsqrt.approx.ftz.f32",
            op("add.rn.f32", op("mul.rn.f32", variance, 1.0 / width), eps_ref[0]),
        )
        normed = op("mul.rn.f32", rstd, op("sub.rn.f32", values, mean))
        out_ref[0, :] = op("fma.rn.f32", scale_ref[:], normed, bias_ref[:])
        mean_ref[0, 0], rstd_ref[0, 0] = mean, rstd

    result = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, width), jnp.float32),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        ),
        grid=(rows,),
        in_specs=(
            pl.BlockSpec((1, width), lambda i: (i, 0)),
            pl.BlockSpec((width,), lambda i: (0,)),
            pl.BlockSpec((width,), lambda i: (0,)),
            pl.BlockSpec((1,), lambda i: (0,)),
        ),
        out_specs=(
            pl.BlockSpec((1, width), lambda i: (i, 0)),
            pl.BlockSpec((1, 1), lambda i: (i, 0)),
            pl.BlockSpec((1, 1), lambda i: (i, 0)),
        ),
        compiler_params=pt.CompilerParams(num_warps=4),
    )(x.reshape(rows, width), scale, bias, jnp.asarray(eps, jnp.float32).reshape(1))
    return (
        result[0].reshape(shape),
        result[1].reshape(*shape[:-1], 1),
        result[2].reshape(*shape[:-1], 1),
    )
