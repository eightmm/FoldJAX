"""Candidate for OpenBind's Triton triangle-multiplication LayerNorm.

This isolated forward kernel is not selected by production inference. It mirrors
``triangular_multiplicative_update.py::layernorm_kernel/triton_layernorm`` at
OpenFold3 commit c4771653c5d0a3ebb0b3af71b05efd64bc44ee86. In particular, the
variance is the second moment minus the squared mean, and epsilon is a lower
bound on variance, not an additive term. Other OpenBind LayerNorms do not share
this contract.

GPU widths 64/128 explicitly reproduce the reduction/affine FMA boundaries and
the sqrt/division instructions observed in the pinned native CUDA PTX. Other
widths remain experimental formula candidates.
``interpret=True`` exercises the formula/memory program on CPU without those PTX
instructions, not a native GPU arithmetic oracle. The repaired GPU path still
needs comparison.
"""

from __future__ import annotations

import functools
import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt
except ImportError:  # pragma: no cover - optional accelerator implementation
    pl = None
    pt = None


_ROWS_PER_PROGRAM = 8
_MAX_BLOCK_SIZE = 1024
_SUPPORTED_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32)


def _fp32_asm(instruction, *values):
    values = jnp.broadcast_arrays(*values)
    shape = values[0].shape
    # Pallas inline asm needs an IR value, including for a scalar constant.
    if not shape:
        values = [value.reshape(1) for value in values]
    registers = ", ".join(f"${index}" for index in range(len(values) + 1))
    result = pt.elementwise_inline_asm(
        f"{instruction} {registers};",
        args=values,
        constraints="=f" + ",f" * len(values),
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(values[0].shape, jnp.float32)],
    )[0]
    return result.reshape(shape)


def _native_block_moments(values):
    """Observed 64/128-wide native PTX: warp butterfly, then warp leaders.

    The first sum-of-squares butterfly is FMA(x, x, rounded_partner_square).
    Replacing it with two separately rounded squares can change the second
    moment by 8 at inputs near 10000, before variance cancellation. Likewise the
    native variance subtraction is a separate explicit FMA below. The ordinary
    JAX reduction did not retain these contractions on the measured GPU.
    """
    warps = values.shape[0] // 32
    left, right = jax.lax.split(values.reshape(warps, 32), (16, 16), axis=1)
    total = _fp32_asm("add.rn.f32", left, right)
    right_squared = _fp32_asm("mul.rn.f32", right, right)
    squared = _fp32_asm("fma.rn.f32", left, left, right_squared)
    # This is lane 0's xor16/8/4/2/1 order, followed by warp-leader xor2/1
    # (128 values) or xor1 (64), not one flat reduction across all values.
    for offset in (8, 4, 2, 1):
        a, b = jax.lax.split(total, (offset, offset), axis=1)
        total = _fp32_asm("add.rn.f32", a, b)
        a, b = jax.lax.split(squared, (offset, offset), axis=1)
        squared = _fp32_asm("add.rn.f32", a, b)
    while warps > 1:
        warps //= 2
        a, b = jax.lax.split(total, (warps, warps), axis=0)
        total = _fp32_asm("add.rn.f32", a, b)
        a, b = jax.lax.split(squared, (warps, warps), axis=0)
        squared = _fp32_asm("add.rn.f32", a, b)
    return total.reshape(()), squared.reshape(())


def _layer_norm_kernel(
    x_ref, weight_ref, bias_ref, shape_ref, eps_ref, out_ref, *, block_size, interpret
):
    # M and N are runtime scalar arguments in the publisher kernel, including
    # N's use as the divisor. Keep them as loads rather than baking a reciprocal
    # of the static shape into the Pallas arithmetic.
    rows, width = shape_ref[0], shape_ref[1]
    row_start = pl.program_id(0) * _ROWS_PER_PROGRAM
    offsets = jnp.arange(block_size)
    blocks = (width + block_size - 1) // block_size
    native_fma = not interpret and x_ref.shape[-1] in (64, 128)

    def process_row(row_in_program, unused):
        del unused
        row = row_start + row_in_program

        def normalize():
            def moments(block, carry):
                cols = block * block_size + offsets
                x = pt.load(x_ref.at[row, cols], mask=cols < width, other=0.0)
                x = x.astype(jnp.float32)
                total, total_squared = carry
                if native_fma:
                    block_total, block_squared = _native_block_moments(x)
                    return (
                        _fp32_asm("add.rn.f32", total, block_total),
                        _fp32_asm("add.rn.f32", total_squared, block_squared),
                    )
                return total + jnp.sum(x), total_squared + jnp.sum(x * x)

            total, total_squared = jax.lax.fori_loop(
                0, blocks, moments, (jnp.float32(0), jnp.float32(0))
            )
            mean = total / width.astype(jnp.float32)
            second_moment = total_squared / width.astype(jnp.float32)
            if native_fma:
                variance = _fp32_asm("fma.rn.f32", -mean, mean, second_moment)
            else:
                variance = second_moment - mean * mean
            floored_variance = jnp.maximum(variance, eps_ref[0])
            if native_fma:
                # Pinned width-128 PTX SHA256:
                # 7a4fd4265462ba994cc4a57c2d9851598d2ec87633765469cecc868157a41167
                # Width 64 has the same sqrt/div pair. Do not substitute
                # libdevice sqrt, rsqrt, or a fused reciprocal-sqrt operation.
                root = _fp32_asm("sqrt.approx.ftz.f32", floored_variance)
                rstd = _fp32_asm("div.full.f32", jnp.float32(1), root)
            else:
                rstd = jnp.float32(1) / jnp.sqrt(floored_variance)

            def affine(block, unused):
                del unused
                cols = block * block_size + offsets
                mask = cols < width
                x = pt.load(x_ref.at[row, cols], mask=mask, other=0.0)
                weight = pt.load(weight_ref.at[cols], mask=mask, other=1.0)
                bias = pt.load(bias_ref.at[cols], mask=mask, other=0.0)
                x, weight, bias = (
                    value.astype(jnp.float32) for value in (x, weight, bias)
                )
                normalized = (x - mean) * rstd
                if native_fma:
                    # Native PTX rounds normalization before one final affine
                    # FMA; a separate weight multiply rounds a second time.
                    out = _fp32_asm("fma.rn.f32", normalized, weight, bias)
                else:
                    out = normalized * weight + bias
                pt.store(out_ref.at[row, cols], out.astype(out_ref.dtype), mask=mask)

            jax.lax.fori_loop(0, blocks, affine, None)

        jax.lax.cond(row < rows, normalize, lambda: None)

    jax.lax.fori_loop(0, _ROWS_PER_PROGRAM, process_row, None)


def native_layer_norm(x, weight, bias, eps=1e-5, *, interpret=False):
    """Run the standalone, not-yet-admitted OpenBind Triton-norm candidate.

    ``x`` has shape ``[..., N]`` with positive N; affine arrays are both ``[N]``.
    FP16, BF16 and FP32 storage are accepted independently for each array; all
    arithmetic is FP32 and the result has x's shape/dtype. Arrays containing
    nonfinite values retain native propagation rather than being sanitized.
    Empty leading dimensions return an empty result without launching a kernel.

    ``eps`` is a static real scalar, finite and positive after FP32 conversion.
    ``interpret`` is a static boolean. With interpretation disabled this is a
    Triton GPU call, never a silent ordinary-JAX or CPU fallback. Neither storage
    support nor CPU interpretation admits a new precision policy for inference.
    """
    if not isinstance(interpret, bool):
        raise TypeError("interpret must be a static boolean")
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError("eps must be a static real scalar")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        eps32 = np.float32(eps)
    if not np.isfinite(eps32) or eps32 <= 0:
        raise ValueError("eps must be finite and positive in FP32")
    for name, value in (("x", x), ("weight", weight), ("bias", bias)):
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{name} must be an array")
        if value.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have FP16, BF16 or FP32 dtype")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x must have a positive last dimension")
    width = x.shape[-1]
    if weight.shape != (width,) or bias.shape != (width,):
        raise ValueError(f"weight and bias must both have shape ({width},)")
    if pl is None or pt is None:
        raise RuntimeError("Pallas/Triton is required for native OpenBind norm")

    rows = math.prod(x.shape[:-1])
    if rows == 0:
        return jnp.empty_like(x)
    block_size = min(_MAX_BLOCK_SIZE, 1 << (width - 1).bit_length())
    kernel = functools.partial(
        _layer_norm_kernel, block_size=block_size, interpret=interpret
    )
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, width), x.dtype),
        grid=(pl.cdiv(rows, _ROWS_PER_PROGRAM),),
        # The native wrapper leaves these at the pinned CUDA compiler defaults.
        compiler_params=pt.CompilerParams(num_warps=4, num_stages=3),
        interpret=interpret,
        name="openbind_native_triangle_layer_norm",
    )(
        x.reshape(rows, width),
        weight,
        bias,
        jnp.asarray((rows, width), jnp.int32),
        jnp.asarray((eps32,), jnp.float32),
    )
    return result.reshape(x.shape)
