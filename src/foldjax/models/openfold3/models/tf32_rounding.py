"""Match the operand rounding upstream's TF32 GEMMs actually use.

Both sides ask for the same thing by name -- upstream calls
``torch.set_float32_matmul_precision("high")`` and the port pins JAX's
``high`` -- and both then round FP32 operands onto the 10-bit TF32 mantissa
grid.  They do not round them the same way.  Measured 2026-09-08 against the
saved native LayerNorm output and the unchanged projection weights, quantizing
both operands in NumPy FP64 and casting back:

===========================  ==========================
Operand rounding             Maximum error vs native
===========================  ==========================
nearest, ties to even (RNE)  ``1.34e-5``
nearest, ties away (RNA)     ``~1.2e-3``
toward zero (RTZ)            ``2.4e-2``
===========================  ==========================

RNE reproduces native and RNA reproduces JAX's ``high``, across all 15
case/projection combinations at three token counts.  A full-shape GPU control
then prequantized every projection operand to the RNE grid and ran the
*unchanged* ``high`` GEMMs: 5/5 projections passed at each shape and the
largest one matched bitwise.  Prequantizing to RNE is therefore idempotent
under the ties-away rounding that follows it, which is what makes this a
correction rather than a second approximation.

**This contract does not generalise across kernels.** Upstream dispatches its
triangle *multiplication* linear through a custom Triton kernel whose
same-operand controls established truncation, not RNE, and its attention core
is a separate kernel again.  Applying this to those paths would replace one
mismatch with another, so the port applies it only where an ordinary GEMM is
what upstream runs.  See :mod:`foldjax.models.openfold3.models.triangle_attention`
for the call sites that qualify and why they stop there.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

#: Environment switch, read the way ``OPENFOLD3_TRIANGLE_BACKEND`` is: once,
#: at the call site, so it reaches every qualifying projection without six
#: signatures growing an argument.
_ENVIRONMENT_VARIABLE = "OPENFOLD3_TF32_ROUNDING"

#: Mantissa bits an FP32 value keeps on the TF32 grid; the low 13 are dropped.
_TF32_MASK = 0xFFFFE000
_HALF = 0xFFF


def tf32_operand_rounding() -> str:
    """Return ``"none"`` (default) or ``"rne"``.

    The default is deliberately inert: an unasked run stays bit-for-bit the run
    it was before this module existed, on every backend.  That matters more
    than it looks, because the correction is only correct on hardware that
    actually rounds to TF32 -- on CPU, where ``high`` is plain FP32, applying
    it would *introduce* the error it exists to remove.
    """

    return os.environ.get(_ENVIRONMENT_VARIABLE, "none").lower()


def round_to_tf32_rne(x: jnp.ndarray) -> jnp.ndarray:
    """Round FP32 to the TF32 grid, nearest, ties to even.

    Non-FP32 inputs pass through: a BF16 activation is already coarser than
    this grid, and rounding it would be meaningless rather than conservative.

    Non-finite inputs pass through as themselves.  The bit arithmetic below
    would carry an infinity's exponent into a NaN and rewrite a NaN's payload,
    and neither is a rounding.  Signed zero survives the arithmetic unaided --
    ``0x80000000 + 0xfff`` masks back to ``0x80000000`` -- and is covered by a
    test rather than left to that observation.

    A finite value within half a grid step of the FP32 maximum rounds up to
    infinity, exactly as the hardware grid does.  That is kept rather than
    clamped: a clamp would be a third rounding rule, and the operands this is
    applied to are activations and weights, not sentinels.
    """

    if x.dtype != jnp.float32:
        return x
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    # Add half a step, plus one more when the retained mantissa is odd, so an
    # exact tie lands on the even neighbour instead of always going up.
    tie_to_even = (bits >> jnp.uint32(13)) & jnp.uint32(1)
    rounded = (bits + jnp.uint32(_HALF) + tie_to_even) & jnp.uint32(_TF32_MASK)
    return jnp.where(
        jnp.isfinite(x), jax.lax.bitcast_convert_type(rounded, jnp.float32), x
    )


def round_operands(*operands: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """Apply the active policy to a GEMM's operands.

    Returns them unchanged unless the policy asks for ``rne``, so a call site
    can be written once and cost nothing when the policy is off.
    """

    if tf32_operand_rounding() != "rne":
        return operands
    return tuple(round_to_tf32_rne(operand) for operand in operands)
