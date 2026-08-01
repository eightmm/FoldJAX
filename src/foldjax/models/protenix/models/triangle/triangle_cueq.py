"""Torch-compatible cuEquivariance JAX triangle kernels."""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
from jax import lax

from foldjax.models.protenix.models.triangle.triangle import (
    TriangleDirection,
    TriangleMultiplicationParams,
)


def _preload_bundled_nvrtc() -> None:
    """Expose the CUDA 13 wheel's NVRTC library to cuEquivariance."""

    if "cuequivariance_jax" in sys.modules:
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or spec.submodule_search_locations is None:
        return
    for root in spec.submodule_search_locations:
        library = Path(root) / "cu13" / "lib" / "libnvrtc.so.13"
        if library.is_file():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            return


def _load_cueq():
    _preload_bundled_nvrtc()
    try:
        import cuequivariance_jax as cuex
    except (ImportError, OSError) as error:
        msg = (
            "cuEquivariance JAX is required by the CUDA 13 default; "
            "install protenix-jax[cuda13] or explicitly select the XLA backend"
        )
        raise RuntimeError(msg) from error
    return cuex


def cueq_triangle_multiplication(
    z: jnp.ndarray,
    mask: jnp.ndarray,
    params: TriangleMultiplicationParams,
    direction: TriangleDirection,
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply the same fused parameterization used by upstream Protenix Torch."""

    cuex = _load_cueq()
    unbatched = z.ndim == 3
    kernel_z = z[None] if unbatched else z
    kernel_mask = mask[None] if unbatched else mask
    output = cuex.triangle_multiplicative_update(
        x=kernel_z,
        direction=direction,
        mask=kernel_mask,
        norm_in_weight=params.layer_norm_in.weight,
        norm_in_bias=params.layer_norm_in.bias,
        p_in_weight=jnp.concatenate(
            (params.linear_a_p.weight, params.linear_b_p.weight), axis=0
        ),
        g_in_weight=jnp.concatenate(
            (params.linear_a_g.weight, params.linear_b_g.weight), axis=0
        ),
        norm_out_weight=params.layer_norm_out.weight,
        norm_out_bias=params.layer_norm_out.bias,
        p_out_weight=params.linear_z.weight,
        g_out_weight=params.linear_g.weight,
        eps=eps,
        fallback=False,
    )
    return output[0] if unbatched else output


def cueq_attention_core(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    *,
    scale: float,
) -> jnp.ndarray:
    """Apply cuEq attention with upstream Torch mask and scaling semantics."""

    cuex = _load_cueq()
    unbatched = q.ndim == 4
    if unbatched:
        q, k, v = q[None], k[None], v[None]
        triangle_bias = triangle_bias[None]
        mask_bias = mask_bias[None]
    output, _, _ = cuex.triangle_attention(
        q=q,
        k=k,
        v=v,
        bias=triangle_bias,
        mask=mask_bias == 0,
        scale=scale,
        precision=lax.Precision.DEFAULT,
    )
    return output[0] if unbatched else output
