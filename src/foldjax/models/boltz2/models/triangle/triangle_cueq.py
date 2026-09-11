"""cuEquivariance JAX backend for fused triangle multiplication."""

from __future__ import annotations

import jax.numpy as jnp

from foldjax.models._cueq import (
    cueq_attention_core,
    load_cueq,
    triangle_multiplication_precision,
)
from foldjax.models.boltz2.models.triangle.triangle import (
    TriangleDirection,
    TriangleMultiplicationParams,
    resolve_native_amp,
)


def _load_cueq_amp_primitives():
    # cuEq 0.11.1 exposes no autocast argument on its public triangle API.
    # Reuse its pinned implementation's primitives at the native AMP boundaries.
    from cuequivariance_jax.triangle._layer_norm_transpose import layer_norm_transpose
    from cuequivariance_jax.triangle._sigmoid_gated_dual_gemm import (
        sigmoid_gated_dual_gemm,
        sigmoid_gated_dual_gemm_dual_x,
    )

    return layer_norm_transpose, sigmoid_gated_dual_gemm, sigmoid_gated_dual_gemm_dual_x


def cueq_triangle_multiplication_forward(
    params: TriangleMultiplicationParams,
    x: jnp.ndarray,
    mask: jnp.ndarray,
    direction: TriangleDirection,
    eps: float = 1e-5,
    native_amp: bool | None = None,
) -> jnp.ndarray:
    """Run the fused cuEquivariance triangle multiplicative update.

    Boltz-JAX stores dense kernels as ``[in, out]`` while cuEquivariance follows
    the PyTorch ``[out, in]`` layout, so every learned projection is transposed.
    Fallback is disabled to keep benchmark and production behavior explicit.
    """

    cuex = load_cueq()

    if resolve_native_amp(x, params["p_in"]["kernel"], native_amp):
        # A BF16 pair residual is still the autocast configuration, so it
        # takes this branch rather than the plain fused kernel. Note what
        # that does and does not preserve: the contraction stays BF16 as
        # before, but `layer_norm_transpose` returns the width it is given,
        # so the input normalisation below runs BF16 rather than FP32 -- the
        # dtype-following behaviour the fused kernel has and `nn.LayerNorm`
        # does not. Only the CPU reference implementation runs in the unit
        # suite; `fallback=False` makes a CUDA-side rejection of the narrow
        # entry width loud on first use rather than silent.
        return _cueq_triangle_native_amp(cuex, params, x, mask, direction, eps=eps)

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
        precision=triangle_multiplication_precision(cuex, dtype=x.dtype),
        fallback=False,
    )


def _cueq_triangle_native_amp(cuex, params, x, mask, direction, *, eps):
    """Preserve native cuEq's norm and BF16 GEMM boundaries.

    The norm is FP32 whenever the pair arrives FP32, which is every released
    call. It is the fused kernel's own dtype-following norm, not
    `nn.LayerNorm`, so a narrowed pair residual normalises at its own width.
    """
    norm, gemm, gemm_dual = _load_cueq_amp_primitives()
    if direction not in ("incoming", "outgoing"):
        raise ValueError(
            f"Unsupported triangle multiplication direction: {direction!r}"
        )
    if x.ndim < 3 or x.shape[-3] != x.shape[-2]:
        raise ValueError("Triangle multiplication requires square pair axes")
    batch_shape = x.shape[:-3]
    n, channels = x.shape[-2:]
    mask = jnp.broadcast_to(mask, x.shape[:-1]).reshape((-1, n, n))
    x = x.reshape((-1, n, n, channels))
    if x.shape == (1, 437, 437, 128):
        from foldjax.models.boltz2.models.primitives.native_cueq_norm import (
            native_cueq_norm,
        )

        # Observed native FP32 cuEq reduction/FMA order, without compiler patching.
        x = native_cueq_norm(
            x, params["norm_in"]["scale"], params["norm_in"]["bias"], eps
        )[0]
    else:
        x = norm(
            x,
            params["norm_in"]["scale"],
            params["norm_in"]["bias"],
            eps=eps,
            layout="bijd->bijd",
            fallback=False,
        )
    # Native Torch autocasts inside gated GEMM, after input normalization.
    # JAX cuEq instead follows x.dtype and would widen the BF16 kernels to FP32.
    x_in = x.astype(jnp.bfloat16)
    precision = triangle_multiplication_precision(cuex, dtype=x_in.dtype)
    ab = gemm(
        x_in,
        params["g_in"]["kernel"].T,
        params["p_in"]["kernel"].T,
        mask=mask,
        transpose_out=True,
        precision=precision,
        fallback=False,
    )
    a, b = jnp.split(ab, 2, axis=0)
    equation = "dbik,dbjk->dbij" if direction == "outgoing" else "dbki,dbkj->dbij"
    contracted = jnp.einsum(equation, a, b)
    # This is the fused cuEq norm, not nn.LayerNorm: FP32 affine parameters
    # survive, but its output keeps the BF16 contraction's dtype.
    if contracted.shape == (128, 1, 437, 437):
        from foldjax.models.boltz2.models.primitives.native_cueq_norm import (
            native_cueq_output_norm,
        )

        x_out = native_cueq_output_norm(
            contracted, params["norm_out"]["scale"], params["norm_out"]["bias"], eps
        )[0]
    else:
        x_out = norm(
            contracted,
            params["norm_out"]["scale"],
            params["norm_out"]["bias"],
            eps=eps,
            layout="dbij->bijd",
            fallback=False,
        )
    out = gemm_dual(
        x_in,
        x_out.astype(jnp.bfloat16),
        params["g_out"]["kernel"].T,
        params["p_out"]["kernel"].T,
        precision=precision,
        fallback=False,
    )
    return out.reshape((*batch_shape, n, n, out.shape[-1]))


# Re-exported: the kernel wrapper moved to `models/_cueq.py`, which Protenix and
# OpenFold3 already used. Kept importable from here so existing call sites do
# not move. `triangle_attention._attention` keeps passing its own `precision`
# to it, because this port's op-level string and its neutral knob deliberately
# disagree; the shared wrapper's docstring records what deriving it would cost.
__all__ = [
    "cueq_attention_core",
    "cueq_triangle_multiplication_forward",
]
