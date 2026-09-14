"""Torch-compatible cuEquivariance JAX triangle kernels."""

from __future__ import annotations

import jax.numpy as jnp

from foldjax.models._cueq import (
    cueq_attention_core,
    fused_triangle_multiplication,
)
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

    unbatched = z.ndim == 3
    kernel_z = z[None] if unbatched else z
    kernel_mask = mask[None] if unbatched else mask
    output = fused_triangle_multiplication(
        kernel_z,
        direction=direction,
        mask=kernel_mask,
        norm_in=(params.layer_norm_in.weight, params.layer_norm_in.bias),
        p_in=(
            jnp.concatenate(
                (params.linear_a_p.weight, params.linear_b_p.weight), axis=0
            ),
            None,
        ),
        g_in=(
            jnp.concatenate(
                (params.linear_a_g.weight, params.linear_b_g.weight), axis=0
            ),
            None,
        ),
        norm_out=(params.layer_norm_out.weight, params.layer_norm_out.bias),
        p_out=(params.linear_z.weight, None),
        g_out=(params.linear_g.weight, None),
        eps=eps,
    )
    return output[0] if unbatched else output


# Re-exported: the kernel wrapper moved to `models/_cueq.py` when OpenFold3
# needed it too. Kept importable from here so existing call sites do not move.
__all__ = ["cueq_triangle_multiplication", "cueq_attention_core"]
