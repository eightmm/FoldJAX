"""The cuEquivariance triangle-attention kernel, shared by every port that has one.

This started inside the Protenix port and moved here when OpenFold3 needed it. The
kernel is model-agnostic -- it takes ``q``/``k``/``v``, a per-head bias and a mask
-- and the four AF3-family ports build all five in the same layout, so the wrapper
belongs beside them rather than inside one of them.

What it buys is the score tensor. The XLA path materialises ``[rows, heads, N, N]``
and writes it to HBM; the fused kernel never forms it. Blocking the rows bounds
that tensor but still pays for it, which is why the fused path wins on both axes
at a real length: measured on Protenix at 1531 tokens, 172.6 s / 21,475 MiB against
254.5 s / 24,764 MiB for the blocked XLA path.

It is not Triton. ``cuequivariance_ops_jax`` ships a precompiled ``libcue_ops_jax.so``
called through ``jax.ffi``, so there is no kernel generation or autotuning at
runtime, and unlike the usual flash kernels it takes float32 -- which is what makes
it usable in the ports whose upstream runs fp32.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
from jax import lax


def preload_bundled_nvrtc() -> None:
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


def load_cueq():
    """Import ``cuequivariance_jax``, or say why it could not be loaded.

    There is deliberately no silent fallback to XLA. A backend that quietly
    changes kernel when a wheel is missing means two machines running two
    different programs under one setting, and a memory figure that cannot be
    reproduced.
    """
    preload_bundled_nvrtc()
    try:
        import cuequivariance_jax as cuex
    except (ImportError, OSError) as error:
        msg = (
            "cuEquivariance JAX is required by this backend; install the "
            "cuda13 extra or select the XLA triangle-attention backend"
        )
        raise RuntimeError(msg) from error
    return cuex


def cueq_attention_core(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    *,
    scale: float,
) -> jnp.ndarray:
    """Apply cuEq attention with upstream Torch mask and scaling semantics.

    Layouts, which every caller must already be in:

    * ``q``/``k``/``v``: ``[B, N_row, H, N_col, D]``
    * ``triangle_bias``: ``[B, 1, H, N_col, N_col]`` -- the row axis is 1 because
      the bias is shared across rows, and the kernel broadcasts it rather than
      materialising a copy per row. That is the property the whole saving rests on.
    * ``mask_bias``: ``[B, N_row, 1, 1, N_col]`` additive, converted to the
      kernel's boolean convention here.

    ``scale`` is applied inside the kernel, so the caller must *not* pre-divide
    the queries the way the XLA path does.

    Leading axes beyond those are folded into ``B`` and restored afterwards. The
    kernel takes exactly five dimensions and unpacks them positionally, so a
    sixth -- which the confidence head produces, one pair representation per
    diffusion sample -- reaches it as ``too many values to unpack`` rather than
    as anything that names the problem.
    """

    cuex = load_cueq()
    # Everything before the (N_row, H, N_col, D) suffix is batch. The three
    # operands carry it in different amounts (the bias has a 1 where q has rows),
    # so broadcast the leads to a common shape before folding them together.
    lead = jnp.broadcast_shapes(
        q.shape[:-4], triangle_bias.shape[:-4], mask_bias.shape[:-4]
    )
    q, k, v = (
        jnp.broadcast_to(t, (*lead, *t.shape[-4:])) for t in (q, k, v)
    )
    triangle_bias = jnp.broadcast_to(
        triangle_bias, (*lead, *triangle_bias.shape[-4:])
    )
    mask_bias = jnp.broadcast_to(mask_bias, (*lead, *mask_bias.shape[-4:]))

    flat = lambda t: t.reshape((-1, *t.shape[-4:]))  # noqa: E731
    output, _, _ = cuex.triangle_attention(
        q=flat(q),
        k=flat(k),
        v=flat(v),
        bias=flat(triangle_bias),
        mask=flat(mask_bias) == 0,
        scale=scale,
        precision=lax.Precision.DEFAULT,
    )
    return output.reshape((*lead, *output.shape[-4:]))
