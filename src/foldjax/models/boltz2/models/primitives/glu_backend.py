"""Gated-linear-unit backend helper.

Mirrors AF3's use of ``tokamax.gated_linear_unit`` for the Transition MLP
(swish GLU) and TriangleMultiplication projection+gate (sigmoid GLU). The
default ``"xla"`` path uses split matmuls, preserving native single-operator
rounding for low-precision activations. The opt-in ``"tokamax"`` path runs the
fused Triton GLU kernel; it only pays off in low precision (fp16/bf16) on a
supported GPU and is verified numerically against the xla path on GPU.

The implementation moved to :mod:`foldjax.models._glu` when the other four
ports gained the same option. This module keeps the name Boltz-2's call sites
import, and the arithmetic is unchanged: the shared function is this one,
lifted.
"""

from __future__ import annotations

from foldjax.models._glu import gated_linear_unit

__all__ = ["gated_linear_unit"]
