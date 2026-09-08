"""Explicit native cuEq norm arithmetic for the verified 437-token AMP route.

Input and transposed output norms have different reduction orders. The caller
limits inference selection to the measured shape; other profiles are not admitted.
"""

import math

import jax
import jax.numpy as jnp


def native_cueq_norm(x, scale, bias, eps=1e-5, *, _output_profile=False):
    if x.ndim < 1 or x.dtype != jnp.float32 or x.shape[-1] != 128:
        raise ValueError("requires FP32 width 128")
    if math.prod(x.shape[:-1]) == 0:
        raise ValueError("requires at least one row")
    if scale.shape != (128,) or bias.shape != (128,):
        raise ValueError("requires width-128 affine parameters")
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    rows = math.prod(x.shape[:-1])

    def kernel(xr, sr, br, yr, mr, rr):
        def op(instruction, *args):
            args = jnp.broadcast_arrays(*[jnp.asarray(a, jnp.float32) for a in args])
            shape = args[0].shape
            args = [a.reshape(1) if not shape else a for a in args]
            registers = ", ".join(f"${i}" for i in range(len(args) + 1))
            return pt.elementwise_inline_asm(
                f"{instruction} {registers};",
                args=args,
                constraints="=f" + ",f" * len(args),
                pack=1,
                result_shape_dtypes=[jax.ShapeDtypeStruct(args[0].shape, jnp.float32)],
            )[0].reshape(shape)

        def reduce64(v):
            v = v.reshape(16, 4)
            if _output_profile:
                components = jax.lax.split(v, (1,) * 16, axis=0)
                total = components[0].reshape(4)
                for component in components[1:]:
                    total = op("add.rn.f32", total, component.reshape(4))
                for offset in (2, 1):
                    left, right = jax.lax.split(total, (offset, offset), axis=0)
                    total = op("add.rn.f32", left, right)
                return total.reshape(())
            components = jax.lax.split(v, (1, 1, 1, 1), axis=1)
            total = components[0].reshape(16)
            for component in components[1:]:
                total = op("add.rn.f32", total, component.reshape(16))
            for offset in (8, 4, 2, 1):
                left, right = jax.lax.split(total, (offset, offset), axis=0)
                total = op("add.rn.f32", left, right)
            return total.reshape(())

        values = xr[0, :]
        left, right = jax.lax.split(values, (64, 64), axis=0)
        mean = op("div.rn.f32", reduce64(op("add.rn.f32", left, right)), 128.0)
        centered = op("sub.rn.f32", values, mean)
        left, right = jax.lax.split(centered, (64, 64), axis=0)
        squares = op("fma.rn.f32", left, left, 0.0)
        squares = op("fma.rn.f32", right, right, squares)
        variance = op("div.rn.f32", reduce64(squares), 128.0)
        rstd = op("rsqrt.approx.ftz.f32", op("add.rn.f32", variance, eps))
        normalized = op("mul.rn.f32", centered, rstd)
        yr[0, :] = op("fma.rn.f32", normalized, sr[:], br[:])
        mr[0] = mean
        rr[0] = rstd

    output, mean, rstd = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, 128), jnp.float32),
            jax.ShapeDtypeStruct((rows,), jnp.float32),
            jax.ShapeDtypeStruct((rows,), jnp.float32),
        ),
        grid=(rows,),
        in_specs=(
            pl.BlockSpec((1, 128), lambda i: (i, 0)),
            pl.BlockSpec((128,), lambda i: (0,)),
            pl.BlockSpec((128,), lambda i: (0,)),
        ),
        out_specs=(
            pl.BlockSpec((1, 128), lambda i: (i, 0)),
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
        ),
        compiler_params=pt.CompilerParams(num_warps=4),
    )(x.reshape(rows, 128), scale.astype(jnp.float32), bias.astype(jnp.float32))
    return (
        output.reshape(x.shape),
        mean.reshape(x.shape[:-1]),
        rstd.reshape(x.shape[:-1]),
    )


def native_cueq_output_norm(x, scale, bias, eps=1e-5):
    """BF16 dbij->bijd norm with its distinct native reduction profile."""
    if x.ndim != 4 or x.shape[0] != 128 or x.dtype != jnp.bfloat16:
        raise ValueError("requires BF16 [128,batch,row,column] input")
    value = x.transpose(1, 2, 3, 0).astype(jnp.float32)
    output, mean, rstd = native_cueq_norm(value, scale, bias, eps, _output_profile=True)
    return output.astype(jnp.bfloat16), mean, rstd
