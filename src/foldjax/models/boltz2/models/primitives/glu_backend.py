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

from foldjax.models._cp import cp_mesh
from foldjax.models._glu import gated_linear_unit

__all__ = ["gated_linear_unit", "reject_fused_glu_under_cp"]


def reject_fused_glu_under_cp(glu_backend: str) -> None:
    """Refuse the fused kernel under a context-parallel mesh, at the site.

    ``api.predict`` resolves the released ``tokamax`` default to ``xla`` when
    ``cp_devices > 1`` and ``Boltz2Backend.validate_native_options`` refuses an
    option dict that names it, but neither is the only door: a library caller
    can open a mesh with ``context_parallel`` and call a transition directly.
    The fused kernel is one Pallas/Triton call over the whole operand, so it
    declares its outputs with no ``manual_axis_type`` and a checked
    ``shard_map`` -- the one the pair transition and the MSA module wrap that
    site in -- refuses it. Saying so here beats a lowering error several frames
    down, and it is the same argument, and the same message, as Protenix's
    ``reject_fused_glu_under_cp``.

    The transition is the only caller. Boltz-2's other fused GLU is the
    triangle multiplication's projection gate, and
    ``triangle_multiplication_forward`` already refuses the combination at the
    top of its own body, before the parameters are read.
    """

    if glu_backend != "xla" and cp_mesh() is not None:
        raise ValueError(
            "context parallelism requires glu_backend='xla'; a fused GLU "
            "cannot be partitioned"
        )
