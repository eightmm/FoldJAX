"""CUDA PWA rounding for the observed 437-token, 128-channel BF16 profile.

Full first-layer operands reproduce native outputs with four-lane logits
accumulation and warp-ordered softmax. Other profiles retain ordinary JAX
until independently checked; this is not a universal cuBLAS kernel model.
"""

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_mesh
from foldjax.models.boltz2.models.primitives._common import linear


def four_lane_projection(x, kernel):
    if x.shape[-1] != 128 or kernel.shape != (128, 1):
        raise ValueError("requires a 128-wide single-head projection")
    x = x.astype(jnp.bfloat16).astype(jnp.float32)
    kernel = kernel.astype(jnp.bfloat16).astype(jnp.float32)
    products = jax.lax.optimization_barrier(x * kernel[:, 0])
    products = products.reshape(*x.shape[:-1], 32, 4)
    total = jnp.zeros((*x.shape[:-1], 4), jnp.float32)
    for i in range(32):
        total = jax.lax.optimization_barrier(total + products[..., i, :])
    for offset in (2, 1):
        total = jax.lax.optimization_barrier(
            total + total[..., jnp.arange(4) ^ offset]
        )
    return total[..., :1].astype(jnp.bfloat16)


def _cuda_rn_divide(numerator, denominator):
    """Keep native FP32 division at BF16 softmax rounding boundaries."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    numerator, denominator = jnp.broadcast_arrays(numerator, denominator)
    if numerator.dtype != jnp.float32 or denominator.dtype != jnp.float32:
        raise ValueError("division requires FP32")
    if not numerator.size or numerator.size % 32:
        raise ValueError("division requires nonempty 32-element blocks")
    shape = numerator.shape
    rows = numerator.size // 32

    def kernel(a, b, out):
        out[:] = pt.elementwise_inline_asm(
            "div.rn.f32 $0, $1, $2;", args=(a[:], b[:]),
            constraints="=f,f,f", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct((32,), jnp.float32)],
        )[0]

    spec = pl.BlockSpec((None, 32), lambda i: (i, 0))
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((rows, 32), jnp.float32),
        grid=(rows,), in_specs=(spec, spec), out_specs=spec,
        compiler_params=pt.CompilerParams(num_warps=4),
    )(numerator.reshape(rows, 32), denominator.reshape(rows, 32)).reshape(shape)


def warp_softmax(x):
    if x.shape[-1] not in (128, 437) or x.dtype != jnp.float32:
        raise ValueError("requires 128 or 437 FP32 columns")
    columns = x.shape[-1]
    capacity = 128 if columns == 128 else 512
    iterations = capacity // 32
    padded = jnp.pad(
        x, [(0, 0)] * (x.ndim - 1) + [(0, capacity - columns)],
        constant_values=-jnp.inf,
    )
    lanes = padded.reshape(*x.shape[:-1], iterations, 32)
    maximum = lanes[..., 0, :]
    for i in range(1, iterations):
        maximum = jnp.maximum(maximum, lanes[..., i, :])
    indices = jnp.arange(32)
    for offset in (16, 8, 4, 2, 1):
        maximum = jnp.maximum(maximum, maximum[..., indices ^ offset])
    exponentials = jnp.exp(lanes - maximum[..., None, :])
    total = jnp.zeros_like(maximum)
    for i in range(iterations):
        total = jax.lax.optimization_barrier(total + exponentials[..., i, :])
    for offset in (16, 8, 4, 2, 1):
        total = jax.lax.optimization_barrier(total + total[..., indices ^ offset])
    normalized = jax.lax.platform_dependent(
        exponentials, total[..., None, :], cuda=_cuda_rn_divide,
        default=lambda a, b: a / b,
    )
    return normalized.reshape(*x.shape[:-1], capacity)[..., :columns]


def observed_profile(z, kernel):
    return (
        z.shape == (1, 437, 437, 128)
        and kernel.shape == (128, 1)
        and kernel.dtype == jnp.bfloat16
        and cp_mesh() is None
    )


def pwa_logits(z, kernel):
    if not observed_profile(z, kernel):
        return linear(z, kernel)
    return jax.lax.platform_dependent(
        z, kernel, cuda=four_lane_projection, default=linear
    )


def pwa_softmax(logits):
    if logits.shape != (1, 1, 437, 437) or cp_mesh() is not None:
        return jax.nn.softmax(logits, axis=-1)
    return jax.lax.platform_dependent(
        logits, cuda=warp_softmax, default=jax.nn.softmax
    )
