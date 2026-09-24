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
from typing import Any

import jax
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


def triangle_multiplication_precision(cuex, *, dtype):
    """Translate JAX's float32 policy for the FFI, which cannot inherit it."""
    policy = jax.config.jax_default_matmul_precision
    modes = {
        None: "DEFAULT",
        "default": "DEFAULT",
        "bfloat16": "DEFAULT",
        "high": "TF32",
        "tensorfloat32": "TF32",
        "highest": "IEEE",
        "float32": "IEEE",
        "TF32_TF32_F32": "TF32",
        "TF32_TF32_F32_X3": "TF32x3",
        "F32_F32_F32": "IEEE",
    }
    if policy not in modes:
        raise ValueError(
            f"cuEquivariance multiplication cannot represent JAX precision {policy!r}; "
            "select XLA multiplication for this policy"
        )
    # Native cuEq overrides float32 policies for half-precision operands.
    # Its TF32 branch uses FP32-only conversion instructions: forwarding
    # BF16 with TF32 can silently turn every gated projection into zero.
    if jnp.dtype(dtype) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        return cuex.TriMulPrecision.DEFAULT
    return getattr(cuex.TriMulPrecision, modes[policy])


def triangle_attention_precision():
    """Pass the active float32 policy across the attention FFI boundary."""
    policy = jax.config.jax_default_matmul_precision
    modes = {
        None: lax.Precision.DEFAULT,
        "default": lax.Precision.DEFAULT,
        "bfloat16": lax.Precision.DEFAULT,
        "high": lax.Precision.HIGH,
        "tensorfloat32": lax.Precision.HIGH,
        "highest": lax.Precision.HIGHEST,
        "float32": lax.Precision.HIGHEST,
        "TF32_TF32_F32": lax.Precision.HIGH,
        "F32_F32_F32": lax.Precision.HIGHEST,
    }
    if policy not in modes:
        raise ValueError(
            f"cuEquivariance attention cannot represent JAX precision {policy!r}; "
            "select XLA attention for this policy"
        )
    return modes[policy]


def cueq_attention_arguments(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
) -> tuple[tuple[int, ...], dict[str, jnp.ndarray]]:
    """The five arrays :func:`cueq_attention_core` hands the kernel, and the lead.

    Split out so the one mask-polarity conversion in this repository -- the
    additive ``0``/``-1e9`` bias becomes the kernel's boolean, whose installed
    contract (``cuequivariance_jax 0.11.1``, ``triangle_attention``) reads
    "boolean, True means valid" -- can be checked without the kernel: a CPU
    host cannot load ``libcue_ops.so``. Nothing here imports cuEquivariance.

    The query extent and the key extent are independent, as the wheel's
    ``[B, N, H, S_qo, D]`` / ``[B, N, H, S_kv, D]`` contract allows: the
    2-D grid's gather path calls it with the local query columns against every
    key (``_cp_attention.gather_triangle_attention_2d_from_pair``).
    """

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
    return lead, {
        "q": flat(q),
        "k": flat(k),
        "v": flat(v),
        "bias": flat(triangle_bias),
        "mask": flat(mask_bias) == 0,
    }


def cueq_attention_core(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    *,
    scale: float,
    precision: lax.Precision | None = None,
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

    ``precision`` is the FFI's float32 strategy. ``None``, the default, derives
    it from the active JAX policy, which is what Protenix and OpenFold3 want:
    their trunks have one precision surface. Boltz-2 has two that deliberately
    disagree -- a neutral ``jax_default_matmul_precision`` of ``"high"`` and an
    op-level ``matmul_precision`` string of ``"highest"``, the latter being what
    its triangle attention ships and what its capture was taken under -- so it
    passes its own resolved value here. Deriving it instead would move that
    kernel from IEEE to TF32 and shift the whole trunk; the difference belongs
    in the call, not in a second implementation of this function.

    Leading axes beyond those are folded into ``B`` and restored afterwards. The
    kernel takes exactly five dimensions and unpacks them positionally, so a
    sixth -- which the confidence head produces, one pair representation per
    diffusion sample -- reaches it as ``too many values to unpack`` rather than
    as anything that names the problem.
    """

    if precision is None:
        precision = triangle_attention_precision()
    cuex = load_cueq()
    lead, arguments = cueq_attention_arguments(q, k, v, triangle_bias, mask_bias)
    queries = arguments["q"].shape[-2]
    if pads_attention_extents():
        arguments = align_attention_arguments(arguments)
    output, _, _ = cuex.triangle_attention(
        **arguments,
        scale=scale,
        precision=precision,
    )
    output = output[..., :queries, :]
    return output.reshape((*lead, *output.shape[-4:]))


#: The attended extents cuEquivariance's half-precision triangle attention is
#: fast at. Measured on the installed wheel (``cuequivariance_jax 0.11.1``, RTX
#: PRO 6000 Blackwell, bf16, 4 heads x 32): an extent that is not a multiple of
#: 8 takes a path 1.7x slower at 2,100, 2.2x at 3,012 and 6-8x at 4,100 than
#: its aligned neighbours, while 16- and 32-alignment buy nothing over 8
#: (``foldjax-bench/align-20260924``). Float32 operands are not affected and
#: are left alone.
ATTENTION_ALIGNMENT = 8


def pads_attention_extents() -> bool:
    """Whether this process's triangle attention runs the CUDA kernel.

    Only that kernel has the slow unaligned path. On any other backend the
    wheel runs its reference body, where padding buys nothing and a longer key
    reduction would only move the CPU parity replays. Decided at trace time
    from the default backend -- a ``lax.platform_dependent`` would trace the
    kernel call twice, doubling what a dispatch census counts.
    """

    return jax.default_backend() == "gpu"


def align_attention_arguments(
    arguments: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Pad the query and key extents of half-precision operands to the alignment.

    Padded keys are masked invalid, so they never enter a softmax; padded query
    rows are computed and dropped by the caller, which slices the output back
    to the original query extent. The aligned kernel is the one an aligned
    input already runs -- 2,096 or 4,096 tokens take it unpadded -- so this
    changes which of the wheel's two paths an unaligned input takes, not the
    arithmetic contract. Measured at 1,003-4,100 with DEFAULT, HIGH and
    HIGHEST: valid rows land as far from an f32 reference as an aligned
    input's do (0.031-0.040 on outputs of magnitude ~3, against 0.008-0.016 on
    the unaligned path), and on the wheel's reference body, which has one
    path, they are bit for bit the unpadded result. A row with no valid key
    follows the kernel's own convention on either path and is not preserved
    across them. :func:`cueq_attention_core` applies it on CUDA only
    (:func:`pads_attention_extents`).
    """

    if jnp.dtype(arguments["q"].dtype) not in (
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float16),
    ):
        return arguments

    def pad(array: jnp.ndarray, axes: tuple[int, ...], value) -> jnp.ndarray:
        widths = [(0, 0)] * array.ndim
        for axis in axes:
            widths[axis] = (0, -array.shape[axis] % ATTENTION_ALIGNMENT)
        if not any(extra for _, extra in widths):
            return array
        return jnp.pad(array, widths, constant_values=value)

    return {
        "q": pad(arguments["q"], (-2,), 0),
        "k": pad(arguments["k"], (-2,), 0),
        "v": pad(arguments["v"], (-2,), 0),
        "bias": pad(arguments["bias"], (-2, -1), 0),
        "mask": pad(arguments["mask"], (-1,), False),
    }


#: Why the fused triangle multiplication cannot always run, in one place.
#:
#: `cuex.triangle_multiplicative_update` takes `p_in_weight` as
#: `(2*D_in, D_in)` and its wrapper refuses a hidden dimension that is not a
#: multiple of 32, so the projection has to be square and the width has to be
#: aligned. That is the kernel's contract rather than any port's policy:
#: upstream carries the same guard and falls back the same way, which Protenix
#: reads off `triangular.py:491`. Spelling it once is what stops two ports
#: from drifting when the wheel changes its mind -- it was written twice, the
#: same rule with different variable names, which is the shape a silent drift
#: takes.
def fused_multiplication_fits(
    *, pair_width: int, hidden_width: int, has_affine: bool
) -> bool:
    """Whether cuEquivariance's fused update can take these widths at all.

    `has_affine` is the third condition and it is not a width: the kernel
    computes both layer norms itself, so a stack whose triangle
    multiplication omits a scale or an offset has nothing to hand it.
    """

    return has_affine and hidden_width == pair_width and pair_width % 32 == 0


def fused_triangle_multiplication(
    x: jnp.ndarray,
    *,
    direction: str,
    mask: jnp.ndarray,
    norm_in: tuple[jnp.ndarray, jnp.ndarray | None],
    p_in: tuple[jnp.ndarray, jnp.ndarray | None],
    g_in: tuple[jnp.ndarray, jnp.ndarray | None],
    norm_out: tuple[jnp.ndarray, jnp.ndarray | None],
    p_out: tuple[jnp.ndarray, jnp.ndarray | None],
    g_out: tuple[jnp.ndarray, jnp.ndarray | None],
    eps: float,
) -> jnp.ndarray:
    """One call into cuEquivariance's fused triangle multiplication.

    Each parameter is a ``(weight, bias)`` pair in the kernel's own layout --
    ``p_in`` and ``g_in`` are the stacked ``(2*D_in, D_in)`` projections, the
    others ``(D_out, D_in)`` -- and it is the caller's job to get there from its
    own parameter tree. That is the only thing the three ports differed in:
    Protenix concatenates its ``a``/``b`` halves and adds a batch axis, Boltz-2
    transposes ``kernel`` layouts, OpenFold3 carries biases the other two do
    not. What they must not differ in is here: the precision selected from the
    operand dtype and ``fallback=False``, so a card that cannot run the kernel
    says so rather than running XLA under this name.

    A ``None`` bias is passed as ``None``, which is what OpenFold3 already did
    for a bias-free stack and what its packing test pins; the kernel treats it
    as absent.
    """

    cuex = load_cueq()
    kwargs: dict[str, Any] = {
        "x": x,
        "direction": direction,
        "mask": mask,
        "eps": eps,
        "precision": triangle_multiplication_precision(cuex, dtype=x.dtype),
        "fallback": False,
    }
    for name, (weight, bias) in (
        ("norm_in", norm_in), ("p_in", p_in), ("g_in", g_in),
        ("norm_out", norm_out), ("p_out", p_out), ("g_out", g_out),
    ):
        kwargs[f"{name}_weight"] = weight
        kwargs[f"{name}_bias"] = bias
    return cuex.triangle_multiplicative_update(**kwargs)
