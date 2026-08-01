"""Gated-linear-unit backend helper.

Mirrors AF3's use of ``tokamax.gated_linear_unit`` for the Transition MLP
(swish GLU) and TriangleMultiplication projection+gate (sigmoid GLU). The
default ``"xla"`` path is the plain split-matmul-then-gate our modules have
always used (bit-exact, CPU-friendly). The opt-in ``"tokamax"`` path runs the
fused Triton GLU kernel; it only pays off in low precision (fp16/bf16) on a
supported GPU and is verified numerically against the xla path on GPU.

tokamax weight layout is ``[K, 2, N]``: index 0 is the activated (gate) branch,
index 1 the linear (value) branch, so the kernel computes
``activation(x @ w[:, 0]) * (x @ w[:, 1])``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def gated_linear_unit(
    x: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_value: jnp.ndarray,
    activation: Callable[[jax.Array], jax.Array],
    *,
    backend: str = "xla",
) -> jnp.ndarray:
    """Return ``activation(x @ w_gate) * (x @ w_value)``.

    ``w_gate`` / ``w_value`` are ``[K, N]`` kernels. ``backend="xla"`` keeps the
    plain elementwise gate; ``backend="tokamax"`` fuses via the Triton kernel.
    """
    if backend == "xla":
        return activation(x @ w_gate) * (x @ w_value)
    if backend == "tokamax":
        import tokamax
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["foldjax.models.boltz2"], known_only=True)
        weights = jnp.stack([w_gate, w_value], axis=1)  # [K, 2, N]
        return tokamax.gated_linear_unit(
            x=x, weights=weights, activation=activation, implementation="triton"
        )
    msg = f"Unsupported glu backend: {backend!r}"
    raise ValueError(msg)
