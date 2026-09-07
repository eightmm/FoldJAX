"""Diagnostic CUDA vector-4 LayerNorm arithmetic; not a production backend.

The observed native widths need at most two active warps. Explicit FP32 PTX
operations distinguish FMA contraction from an unfused JAX expression.
"""

from __future__ import annotations

import math


def conditioning_override(original):
    """Replace only pairwise and projection-list norms in a diagnostic call."""
    from unittest.mock import patch

    import jax.numpy as jnp

    from foldjax.models.boltz2.models.diffusion import diffusion_conditioning as dc
    from foldjax.models.boltz2.models.primitives import transition as transition_module
    from foldjax.models.boltz2.models.primitives._common import linear
    from foldjax.models.boltz2.models.trunk_blocks import conditioning as pc

    original_pair = dc.pairwise_conditioning_forward

    def norm(x, scale, bias, eps):
        return native_layer_norm(x.astype(jnp.float32), scale, bias, eps)[0]

    def pair(*args, **kwargs):
        with (
            patch.object(pc, "_layer_norm", norm),
            patch.object(transition_module, "_layer_norm", norm),
        ):
            return original_pair(*args, **kwargs)

    def projections(params, x, eps, *, compute_dtype=None):
        if compute_dtype != jnp.bfloat16:
            raise ValueError("conditioning diagnostic requires native BF16 AMP")
        return jnp.concatenate(
            [
                linear(
                    norm(x, p["norm"]["scale"], p["norm"]["bias"], eps),
                    p["linear"]["kernel"],
                    compute_dtype=compute_dtype,
                )
                for p in params
            ],
            axis=-1,
        )

    def run(*args, **kwargs):
        if kwargs.get("lazy_token_trans_bias", False):
            raise ValueError("explicit-FMA control requires materialized token bias")
        with (
            patch.object(dc, "pairwise_conditioning_forward", pair),
            patch.object(dc, "_projection_list_forward", projections),
        ):
            return original(*args, **kwargs)

    return run


def native_layer_norm(x, scale, bias, eps=1e-5):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    width = x.shape[-1]
    if width not in (16, 64, 128, 256) or x.dtype != jnp.float32:
        raise ValueError("diagnostic requires FP32 and width 16, 64, 128 or 256")
    shape = x.shape
    rows = math.prod(shape[:-1])

    def kernel(x_ref, scale_ref, bias_ref, out_ref, mean_ref, rstd_ref):
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
            op("add.rn.f32", op("mul.rn.f32", variance, 1.0 / width), eps),
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
        ),
        out_specs=(
            pl.BlockSpec((1, width), lambda i: (i, 0)),
            pl.BlockSpec((1, 1), lambda i: (i, 0)),
            pl.BlockSpec((1, 1), lambda i: (i, 0)),
        ),
        compiler_params=pt.CompilerParams(num_warps=4),
    )(x.reshape(rows, width), scale, bias)
    return (
        result[0].reshape(shape),
        result[1].reshape(*shape[:-1], 1),
        result[2].reshape(*shape[:-1], 1),
    )
