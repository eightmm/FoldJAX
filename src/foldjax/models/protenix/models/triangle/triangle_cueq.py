"""Torch-compatible cuEquivariance JAX triangle kernels."""

from __future__ import annotations

import jax.numpy as jnp

from foldjax.models._cueq import cueq_attention_core, load_cueq
from foldjax.models.protenix.models.triangle.triangle import (
    TriangleDirection,
    TriangleMultiplicationParams,
)


def cueq_triangle_multiplication(
    z: jnp.ndarray,
    mask: jnp.ndarray,
    params: TriangleMultiplicationParams,
    direction: TriangleDirection,
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply the same fused parameterization used by upstream Protenix Torch."""

    cuex = load_cueq()
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


# Re-exported: the kernel wrapper moved to `models/_cueq.py` when OpenFold3
# needed it too. Kept importable from here so existing call sites do not move.
__all__ = ["cueq_triangle_multiplication", "cueq_attention_core"]
