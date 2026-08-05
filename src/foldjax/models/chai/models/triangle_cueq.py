"""cuEquivariance kernels for Chai's triangle attention and multiplication."""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from foldjax.models.chai.models.pairformer import (
        FusedTriangleMultiplicationParams,
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
            "cuEquivariance JAX is required by the Chai triangle default; "
            "install foldjax[cuda13], or name the blocked path explicitly with "
            "CHAI_JAX_TRIANGLE_ATTENTION_BACKEND=xla and "
            "CHAI_JAX_TRIANGLE_MULTIPLICATION_BACKEND=xla"
        )
        raise RuntimeError(msg) from error
    return cuex


def cueq_triangle_multiplication_direction(
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: FusedTriangleMultiplicationParams,
    *,
    incoming: bool,
    eps: float = 1e-5,
    kernel_dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
    """Return one direction's full contribution to Chai's fused triangle update.

    Chai's module is not two triangle updates; it is one, with a single output
    projection over the *sum* of the two normalized products::

        sigmoid(g_out . norm) * W_out . (LN(outgoing) + LN(incoming))

    The kernel only computes one direction and bakes its own output projection
    and gate into that same call, so at first reading the two do not compose.
    They do, because the projection is linear and the gate is shared::

        g * W . (A + B)  ==  g * W . A  +  g * W . B

    so passing `linear_z_out` as `p_out_weight` and the shared output gate as
    `g_out_weight` to *both* calls makes their sum exactly Chai's module. The
    output LayerNorm carries no learned weight here (Chai calls `layer_norm`
    with neither weight nor bias), so ones and zeros are passed explicitly
    rather than left to the kernel's initializer, which wants a PRNG key.
    """

    cuex = _load_cueq()
    c_z = z.shape[-1]
    unbatched = z.ndim == 3
    kernel_z = z[None] if unbatched else z
    # The blocked path is handed `z` in fp32 and runs every matmul through
    # `linear_bf16`, so all six of its intermediates are bf16 and it returns
    # bf16. Passing the fp32 tensor into the kernel instead runs the whole
    # module in fp32 -- twice the bytes and several times the arithmetic of the
    # path it replaces, which is exactly what the first measurement showed:
    # 970 tokens went from 17,185 MiB / 179.6 s to 18,205 MiB / 202.2 s, a
    # regression on both axes. The only fp32 step Chai actually keeps here is
    # the input LayerNorm's statistics, and the kernel computes those in fp32
    # internally; what changes is that they are taken over a bf16-rounded z.
    #
    # `kernel_dtype` exists so the mapping can be checked in fp32, where the
    # two-call identity against the blocked module holds to 2e-5 instead of
    # bf16's 2e-2. It is not a backend switch: nothing on the prediction path
    # passes it.
    kernel_z = kernel_z.astype(kernel_dtype)
    kernel_mask = pair_mask[None] if unbatched else pair_mask
    if incoming:
        # Chai masks the incoming projections with the transposed pair mask.
        # The kernel applies whatever mask it is handed to both projections
        # without transposing, so the transpose happens here. Every trunk call
        # site builds the mask as an outer product of one token mask and is
        # therefore symmetric, but relying on that would make an asymmetric
        # mask silently wrong instead of merely slower.
        kernel_mask = jnp.swapaxes(kernel_mask, -1, -2)
    # Float, not bool: the kernel is free to either multiply by the mask or
    # select on it, and 1.0/0.0 is correct under both readings.
    kernel_mask = kernel_mask.astype(kernel_z.dtype)
    offset = 2 if incoming else 0
    p_weight = params.merged_linear_p_weight[offset * c_z : (offset + 2) * c_z]
    g_weight = params.merged_linear_g_weight[offset * c_z : (offset + 2) * c_z]
    output = cuex.triangle_multiplicative_update(
        x=kernel_z,
        direction="incoming" if incoming else "outgoing",
        mask=kernel_mask,
        norm_in_weight=params.layer_norm_weight,
        norm_in_bias=params.layer_norm_bias,
        p_in_weight=p_weight,
        g_in_weight=g_weight,
        norm_out_weight=jnp.ones(c_z, dtype=jnp.float32),
        norm_out_bias=jnp.zeros(c_z, dtype=jnp.float32),
        p_out_weight=params.linear_z_out_weight,
        g_out_weight=params.merged_linear_g_weight[4 * c_z : 5 * c_z],
        eps=eps,
        fallback=False,
    )
    return output[0] if unbatched else output


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
