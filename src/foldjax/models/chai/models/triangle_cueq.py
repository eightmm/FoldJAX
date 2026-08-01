"""Optional cuEquivariance kernels for Chai triangle attention."""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp


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
            "cuEquivariance JAX is unavailable; install chai-jax[cueq-cu13] "
            "or select CHAI_JAX_TRIANGLE_ATTENTION_BACKEND=xla"
        )
        raise RuntimeError(msg) from error
    return cuex


def cueq_available() -> bool:
    """Return whether the optional cuEquivariance runtime can be loaded."""

    try:
        _load_cueq()
    except RuntimeError:
        return False
    return True


def cueq_attention_core(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    scale: float,
) -> jnp.ndarray:
    """Apply cuEq attention while preserving Chai's Torch mask semantics."""

    cuex = _load_cueq()
    output, _, _ = cuex.triangle_attention(
        q=q,
        k=k,
        v=v,
        bias=bias,
        mask=mask,
        scale=scale,
        precision=jax.lax.Precision.HIGHEST,
    )
    return output
