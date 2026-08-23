"""Small JAX primitives matching Protenix inference modules."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_identity


class LinearParams(NamedTuple):
    """PyTorch-compatible linear parameters.

    PyTorch stores linear weights as [out_features, in_features]. JAX matmul
    uses the final input dimension, so the forward pass multiplies by
    ``weight.T``.
    """

    weight: jnp.ndarray
    bias: jnp.ndarray | None = None


class LayerNormParams(NamedTuple):
    """Parameters for Protenix/OpenFold layer norm."""

    weight: jnp.ndarray | None = None
    bias: jnp.ndarray | None = None


class TransitionParams(NamedTuple):
    """Parameters for ``protenix.model.modules.primitives.Transition``."""

    layer_norm: LayerNormParams
    linear_a: LinearParams
    linear_b: LinearParams
    linear_out: LinearParams


class AdaptiveLayerNormParams(NamedTuple):
    """Parameters for ``AdaptiveLayerNorm``."""

    layernorm_a: LayerNormParams
    layernorm_s: LayerNormParams
    linear_s: LinearParams
    linear_no_bias_s: LinearParams


def linear(x: jnp.ndarray, params: LinearParams) -> jnp.ndarray:
    """Apply a PyTorch-layout linear projection."""

    y = jnp.matmul(x, jnp.swapaxes(params.weight, -1, -2))
    if params.bias is not None:
        y = y + params.bias
    return y


def layer_norm(
    x: jnp.ndarray,
    params: LayerNormParams,
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply layer norm over the final dimension."""

    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax_reciprocal_sqrt(var + eps)
    if params.weight is not None:
        y = y * params.weight
    if params.bias is not None:
        y = y + params.bias
    return y


def silu(x: jnp.ndarray) -> jnp.ndarray:
    """SiLU activation matching ``torch.nn.functional.silu``."""

    return x * jnp.reciprocal(1.0 + jnp.exp(-x))


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation matching PyTorch."""

    return jnp.reciprocal(1.0 + jnp.exp(-x))


# The transition widens its input before narrowing it again, and holds three
# copies of the wide form at once -- `a`, `b`, and `silu(a) * b`. On OpenDDE's
# structural pair representation that is three f32[946, 946, 768] buffers,
# 2,622 MiB each: 7,866 MiB of a 10,914 MiB temp arena, for an operation that
# is elementwise over every axis but the last.
#
# Blocking the leading axis is mathematically exact: layer norm reduces over
# the channel axis and the linears contract over it, so no reduction crosses a
# block boundary. It is not bit-identical, because XLA picks a different GEMM
# tiling for the blocked shape -- measured at 2e-4 relative under the default
# TF32 precision and 3e-7 under `float32` precision, i.e. the same order as any
# other shape change. There is no knob to trade here, only kernel launches, and
# only on tensors already large enough for that to be cheap.
_TRANSITION_WIDE_BUDGET_BYTES = 512 * 1024**2


def _transition_chunk_rows(x: jnp.ndarray, params: TransitionParams) -> int | None:
    """Rows of ``x`` whose widened form fits the budget, or None to do it whole."""
    rows = x.shape[0]
    if rows < 2:
        return None
    hidden = params.linear_a.weight.shape[0]
    per_row = hidden * x.dtype.itemsize
    for size in x.shape[1:-1]:
        per_row *= size
    if per_row <= 0 or per_row * rows <= _TRANSITION_WIDE_BUDGET_BYTES:
        return None
    return max(1, _TRANSITION_WIDE_BUDGET_BYTES // per_row)


def _transition_block(x: jnp.ndarray, params: TransitionParams) -> jnp.ndarray:
    y = layer_norm(x, params.layer_norm)
    a = linear(y, params.linear_a)
    b = linear(y, params.linear_b)
    return linear(silu(a) * b, params.linear_out)


def _transition_runtime_identity() -> tuple[tuple, str]:
    """Static execution identity for the independently compiled transition."""

    return cp_identity(), jax.default_backend()


def _transition_for_runtime(
    x: jnp.ndarray,
    params: TransitionParams,
    *,
    chunk_size: int | None,
    runtime_identity: tuple[tuple, str],
) -> jnp.ndarray:
    """Apply one transition using the proven execution route for this runtime."""

    if chunk_size is None:
        chunk_size = _transition_chunk_rows(x, params)
    if chunk_size is None or chunk_size <= 0 or chunk_size >= x.shape[0]:
        return _transition_block(x, params)

    cp_topology, platform = runtime_identity
    n_chunks = -(-x.shape[0] // chunk_size)
    if (
        cp_topology[0] == "serial"
        and platform == "cpu"
        and x.dtype == jnp.float32
        and n_chunks >= 4
    ):
        # `batch_size` vectorizes one fixed-width transition block inside a
        # scan. JAX evaluates a non-divisible tail through a separate vmap, so
        # it retains the historical short final block rather than padding or
        # recomputing rows. This keeps the compiled graph and live widened
        # buffers bounded when there are enough blocks for the loop to pay.
        #
        # Keep this route serial and CPU-only. Mapping the global leading axis
        # under either CP layout introduces a full-width all-gather, while GPU
        # latency and memory have not been measured for this schedule.
        return jax.lax.map(
            lambda row: _transition_block(row, params),
            x,
            batch_size=chunk_size,
        )

    # Historical route for GPU/TPU, reduced precision, context parallelism,
    # and short block lists.
    return jnp.concatenate(
        [
            _transition_block(x[start : start + chunk_size], params)
            for start in range(0, x.shape[0], chunk_size)
        ],
        axis=0,
    )


def transition(
    x: jnp.ndarray,
    params: TransitionParams,
    *,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply the Protenix transition block, blocking the widened intermediates.

    ``chunk_size`` overrides the automatic block size; ``0`` disables blocking
    and materialises the wide form whole.
    """

    return _transition_for_runtime(
        x,
        params,
        chunk_size=chunk_size,
        runtime_identity=_transition_runtime_identity(),
    )


@partial(
    jax.jit,
    static_argnames=("chunk_size", "runtime_identity"),
)
def _compiled_transition(
    x: jnp.ndarray,
    params: TransitionParams,
    *,
    chunk_size: int | None,
    runtime_identity: tuple[tuple, str],
) -> jnp.ndarray:
    return _transition_for_runtime(
        x,
        params,
        chunk_size=chunk_size,
        runtime_identity=runtime_identity,
    )


def compiled_transition(
    x: jnp.ndarray,
    params: TransitionParams,
    *,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply the transition through a topology- and platform-keyed JIT."""

    return _compiled_transition(
        x,
        params,
        chunk_size=chunk_size,
        runtime_identity=_transition_runtime_identity(),
    )


def adaptive_layer_norm(
    a: jnp.ndarray,
    s: jnp.ndarray,
    params: AdaptiveLayerNormParams,
) -> jnp.ndarray:
    """Apply Protenix adaptive layer norm."""

    a_norm = layer_norm(a, params.layernorm_a)
    s_norm = layer_norm(s, params.layernorm_s)
    return sigmoid(linear(s_norm, params.linear_s)) * a_norm + linear(
        s_norm,
        params.linear_no_bias_s,
    )


def jax_reciprocal_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    """Small helper kept separate for testable numerical parity."""

    return jnp.reciprocal(jnp.sqrt(x))
