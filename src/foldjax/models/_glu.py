"""One fused gated-linear-unit path, shared by every port that has a SwiGLU.

The transitions in these models all compute ``activation(x @ w_gate) * (x @
w_value)``. Spelled as two matmuls and an elementwise product, XLA
materialises the widened gate and value tensors before multiplying them;
``tokamax.gated_linear_unit`` runs the same arithmetic in one Triton kernel
and never writes them out. The saving is the intermediate, not the precision,
which is why this is the one fused kernel worth offering on an fp32 port.

Two layouts reach the same kernel. ``gated_linear_unit`` takes the two
``[K, N]`` kernels separately and stacks them; ``gated_linear_unit_packed``
takes the ``[K, 2N]`` form some ports store, whose first ``N`` columns are the
gate branch, and reshapes it without a copy. tokamax wants ``[K, 2, N]`` with
index 0 the activated branch.

The two backends do not round identically on low-precision activations. The
``xla`` path evaluates the activation in float32 and casts back, which is what
the torch models it ports from do; the fused kernel is handed the activation
and applies it at the kernel's own width. On float32 activations the question
does not arise. Treat a backend change as a numerics change and read it
against the port's own rerun floor.

There is no fallback. ``implementation`` is pinned to Triton so a card that
cannot run the kernel says so, instead of running XLA under a name that claims
otherwise and producing a measurement that compares XLA with XLA.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

#: Values every port's ``glu_backend`` option accepts.
GLU_BACKENDS = ("xla", "tokamax")


def _check(backend: str) -> None:
    if backend not in GLU_BACKENDS:
        msg = f"glu backend must be one of {GLU_BACKENDS}; got {backend!r}"
        raise ValueError(msg)


def _fused(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    activation: Callable[[jax.Array], jax.Array],
) -> jnp.ndarray:
    import tokamax
    from absl import flags

    # tokamax reads absl flags for its autotuning policy and raises if the
    # process never parsed any. A library caller has no argv to give it.
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["foldjax.models"], known_only=True)
    return tokamax.gated_linear_unit(
        x=x, weights=weights, activation=activation, implementation="triton"
    )


def gated_linear_unit(
    x: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_value: jnp.ndarray,
    activation: Callable[[jax.Array], jax.Array],
    *,
    backend: str = "xla",
) -> jnp.ndarray:
    """Return ``activation(x @ w_gate) * (x @ w_value)``.

    ``w_gate`` and ``w_value`` are ``[K, N]``. The operand is narrowed to the
    weights' dtype first, so a low-precision weight selects the arithmetic the
    way an autocast projection would.
    """

    _check(backend)
    if w_gate.dtype in (jnp.bfloat16, jnp.float16):
        x = x.astype(w_gate.dtype)
    if backend == "tokamax":
        return _fused(x, jnp.stack([w_gate, w_value], axis=1), activation)
    gate = x @ w_gate
    if gate.dtype in (jnp.bfloat16, jnp.float16):
        activated = activation(gate.astype(jnp.float32)).astype(gate.dtype)
    else:
        activated = activation(gate)
    return activated * (x @ w_value)


def gated_linear_unit_packed(
    x: jnp.ndarray,
    w_packed: jnp.ndarray,
    activation: Callable[[jax.Array], jax.Array],
    *,
    backend: str = "xla",
) -> jnp.ndarray:
    """The same unit from one ``[K, 2N]`` kernel, gate branch first.

    The reshape to ``[K, 2, N]`` is free: C order puts columns ``0..N-1`` at
    index 0, which is the branch tokamax activates. The ``xla`` path keeps the
    single widened matmul the packed layout exists for, so the two backends
    differ here in one more place than they do above -- the packed path does
    one ``[K, 2N]`` matmul where the fused kernel does two ``[K, N]`` halves.
    """

    _check(backend)
    if w_packed.dtype in (jnp.bfloat16, jnp.float16):
        x = x.astype(w_packed.dtype)
    if w_packed.shape[-1] % 2:
        msg = f"packed GLU kernel must have an even width; got {w_packed.shape}"
        raise ValueError(msg)
    half = w_packed.shape[-1] // 2
    if backend == "tokamax":
        weights = w_packed.reshape(w_packed.shape[:-1] + (2, half))
        return _fused(x, weights, activation)
    packed = x @ w_packed
    gate, value = packed[..., :half], packed[..., half:]
    if gate.dtype in (jnp.bfloat16, jnp.float16):
        gate = activation(gate.astype(jnp.float32)).astype(packed.dtype)
    else:
        gate = activation(gate)
    return gate * value
