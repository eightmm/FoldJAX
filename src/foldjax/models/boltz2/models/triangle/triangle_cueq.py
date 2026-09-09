"""cuEquivariance JAX backend for fused triangle multiplication."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from foldjax.models._cueq import triangle_multiplication_precision
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


def _load_cueq_amp_primitives():
    # cuEq 0.11.1 exposes no autocast argument on its public triangle API.
    # Reuse its pinned implementation's primitives at the native AMP boundaries.
    from cuequivariance_jax.triangle._layer_norm_transpose import layer_norm_transpose
    from cuequivariance_jax.triangle._sigmoid_gated_dual_gemm import (
        sigmoid_gated_dual_gemm,
        sigmoid_gated_dual_gemm_dual_x,
    )

    return layer_norm_transpose, sigmoid_gated_dual_gemm, sigmoid_gated_dual_gemm_dual_x


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


def triangle_amp_granularity() -> str:
    """How to reach cuEquivariance when autocast boundaries are in play.

    Upstream makes a single fused ``triangle_multiplicative_update`` call and
    lets ``torch.autocast`` place the boundaries. This port instead decomposes
    the operator into ``norm``, ``gemm`` and ``gemm_dual`` and places them by
    hand, which is what ``"decomposed"`` selects and what ships.

    ``"fused"`` takes upstream's own granularity. Measured 2026-09-09 against a
    native 5SAK capture, teacher-forced on native's own pair input with
    bitwise-identical MSA features: the fused call gives a first-cycle
    ``delta_z`` RMSE of 3.212102e-02 against the decomposed path's 3.233621e-02
    -- 0.67% closer to native, against a repeated-baseline spread of 0.02% and a
    control that moves the result twenty-six fold. The same library computes both;
    only the number of calls it is reached through differs, and a fused kernel
    does not accumulate the way three primitive calls do.

    It is not the default. That is a stage measurement, not an end-to-end one,
    and this repository has twice this month watched a stage-level improvement
    fail to predict a coordinate one. Changing what ships needs a native
    comparison on final coordinates.
    """

    value = os.environ.get("BOLTZ_JAX_TRIANGLE_AMP_GRANULARITY", "decomposed").lower()
    if value not in ("decomposed", "fused"):
        msg = (
            "BOLTZ_JAX_TRIANGLE_AMP_GRANULARITY must be 'decomposed' or "
            f"'fused'; got {value!r}. A misspelling must not quietly select "
            "the other path: two machines would run two different programs "
            "under one setting."
        )
        raise ValueError(msg)
    return value


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

    compute_dtype = params["p_in"]["kernel"].dtype
    if (
        x.dtype == jnp.float32
        and compute_dtype == jnp.bfloat16
        and triangle_amp_granularity() == "decomposed"
    ):
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
    """Preserve native cuEq's FP32 norm and BF16 GEMM boundaries."""
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
