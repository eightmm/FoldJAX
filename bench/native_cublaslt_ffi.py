"""Development-only FFI wrapper; explicit prebuilt library, no implicit build."""

from __future__ import annotations

import ctypes
import hashlib
from math import prod
from pathlib import Path

_LIBRARIES = {}


def register(library, *, fp32=False):
    import jax

    path = Path(library).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target = "foldjax_bench_native_linear_" + ("f32_" if fp32 else "") + digest
    if target not in _LIBRARIES:
        loaded = ctypes.CDLL(str(path))
        symbol = "FoldjaxBenchNativeLinearF32" if fp32 else "FoldjaxBenchNativeLinear"
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(getattr(loaded, symbol)),
            platform="CUDA",
            api_version=1,
        )
        _LIBRARIES[target] = loaded
    return target


def linear(x, weight, *, target, fp32=False):
    import jax
    import jax.numpy as jnp

    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("native linear requires matching matrix contraction axes")
    m, k, n = prod(x.shape[:-1]), x.shape[-1], weight.shape[0]
    if min(m, k, n) <= 0 or max(m, k, n) > 2**31 - 1:
        raise ValueError("native linear requires positive int32 dimensions")
    dtype = jnp.float32 if fp32 else jnp.bfloat16
    if fp32 and (x.dtype != dtype or weight.dtype != dtype):
        raise ValueError("FP32 native linear requires FP32 operands without conversion")
    result = jax.ffi.ffi_call(
        target,
        [
            jax.ShapeDtypeStruct((m, n), dtype),
            jax.ShapeDtypeStruct((32 * 1024 * 1024,), jnp.uint8),
        ],
        input_layouts=((0, 1), (0, 1)),
        output_layouts=((0, 1), (0,)),
        vmap_method="sequential",
    )(x.astype(dtype).reshape(m, k), weight.astype(dtype))
    return result[0].reshape(*x.shape[:-1], n)
