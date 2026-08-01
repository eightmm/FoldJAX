"""cuEquivariance JAX backend for fused triangle multiplication."""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.triangle.triangle import (
    TriangleDirection,
    TriangleMultiplicationParams,
)


def _preload_bundled_nvrtc() -> None:
    """Expose pip-bundled CUDA 13 NVRTC to cuEquivariance's shared library."""

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
            "cuEquivariance JAX is unavailable; install the matching "
            "cuequivariance-jax and cuequivariance-ops-jax CUDA packages"
        )
        raise RuntimeError(msg) from error
    return cuex


def cueq_attention_core(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    *,
    scale: float,
    precision: jax.lax.Precision | None,
) -> jnp.ndarray:
    """Run the Torch-compatible cuEquivariance triangle-attention core."""

    cuex = _load_cueq()
    output, _, _ = cuex.triangle_attention(
        q=q,
        k=k,
        v=v,
        bias=tri_bias,
        mask=mask_bias == 0,
        scale=scale,
        precision=precision,
    )
    return output


def cueq_triangle_multiplication_forward(
    params: TriangleMultiplicationParams,
    x: jnp.ndarray,
    mask: jnp.ndarray,
    direction: TriangleDirection,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run the fused cuEquivariance triangle multiplicative update.

    Boltz-JAX stores dense kernels as ``[in, out]`` while cuEquivariance follows
    the PyTorch ``[out, in]`` layout, so every learned projection is transposed.
    Fallback is disabled to keep benchmark and production behavior explicit.
    """

    cuex = _load_cueq()

    return cuex.triangle_multiplicative_update(
        x=x,
        direction=direction,
        mask=mask,
        norm_in_weight=params["norm_in"]["scale"],
        norm_in_bias=params["norm_in"]["bias"],
        p_in_weight=params["p_in"]["kernel"].T,
        g_in_weight=params["g_in"]["kernel"].T,
        norm_out_weight=params["norm_out"]["scale"],
        norm_out_bias=params["norm_out"]["bias"],
        p_out_weight=params["p_out"]["kernel"].T,
        g_out_weight=params["g_out"]["kernel"].T,
        eps=eps,
        fallback=False,
    )
