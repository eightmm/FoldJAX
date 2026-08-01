"""Small JAX primitives matching Protenix inference modules."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


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


def transition(x: jnp.ndarray, params: TransitionParams) -> jnp.ndarray:
    """Apply the Protenix transition block without eval chunking."""

    y = layer_norm(x, params.layer_norm)
    a = linear(y, params.linear_a)
    b = linear(y, params.linear_b)
    return linear(silu(a) * b, params.linear_out)


compiled_transition = jax.jit(transition)


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
