"""JAX primitives matching OpenFold3 inference modules.

Each parameter container mirrors one ``torch.nn.Module``'s ``state_dict`` layout
so the mapping from a checkpoint stays explicit and greppable. Forward passes are
pure functions over arrays.

References are to ``openfold3/core/model/primitives`` and
``openfold3/core/model/layers/transition.py`` in the upstream checkout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class LinearParams(NamedTuple):
    """Parameters for ``openfold3.core.model.primitives.Linear``.

    OpenFold3's ``Linear`` subclasses ``torch.nn.Linear`` and only customizes
    initialization, so the inference math is a plain affine map. PyTorch stores
    the weight as ``[out_features, in_features]``; JAX contracts over the final
    input axis, so the forward pass multiplies by ``weight.T``.
    """

    weight: jnp.ndarray
    bias: jnp.ndarray | None = None


class LayerNormParams(NamedTuple):
    """Parameters for ``openfold3.core.model.primitives.LayerNorm``.

    Both the scale and the offset are optional upstream (``create_scale`` /
    ``create_offset``), and AdaLN relies on that: it normalizes ``a`` with
    neither and ``s`` with a scale only.
    """

    weight: jnp.ndarray | None = None
    bias: jnp.ndarray | None = None


class SwiGLUParams(NamedTuple):
    """Parameters for ``openfold3.core.model.primitives.SwiGLU``.

    Both projections are bias-free upstream (``swiglu_init``).
    """

    linear_a: LinearParams
    linear_b: LinearParams


class AdaLNParams(NamedTuple):
    """Parameters for ``openfold3.core.model.primitives.AdaLN`` (AF3 Alg. 26).

    ``linear_g`` carries a bias and ``linear_s`` does not (``ada_ln_init``).
    """

    layer_norm_a: LayerNormParams
    layer_norm_s: LayerNormParams
    linear_g: LinearParams
    linear_s: LinearParams


class SwiGLUTransitionParams(NamedTuple):
    """Parameters for ``SwiGLUTransition`` (AF3 Algorithm 11)."""

    layer_norm: LayerNormParams
    swiglu: SwiGLUParams
    linear_out: LinearParams


def linear(x: jnp.ndarray, params: LinearParams) -> jnp.ndarray:
    """Apply a PyTorch-layout linear projection."""
    y = jnp.matmul(x, jnp.swapaxes(params.weight, -1, -2))
    if params.bias is not None:
        y = y + params.bias
    return y


def layer_norm(
    x: jnp.ndarray, params: LayerNormParams, *, eps: float = 1e-5
) -> jnp.ndarray:
    """Apply layer norm over the final axis with optional scale and offset."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax_rsqrt(variance + eps)
    if params.weight is not None:
        y = y * params.weight
    if params.bias is not None:
        y = y + params.bias
    return y


def jax_rsqrt(x: jnp.ndarray) -> jnp.ndarray:
    """Reciprocal square root, spelled out to keep the layer norm readable."""
    return jnp.reciprocal(jnp.sqrt(x))


def silu(x: jnp.ndarray) -> jnp.ndarray:
    """SiLU activation matching ``torch.nn.SiLU``."""
    return x * jax_sigmoid(x)


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable logistic sigmoid."""
    return jnp.where(
        x >= 0,
        jnp.reciprocal(1.0 + jnp.exp(-jnp.abs(x))),
        jnp.exp(-jnp.abs(x)) * jnp.reciprocal(1.0 + jnp.exp(-jnp.abs(x))),
    )


def swiglu(x: jnp.ndarray, params: SwiGLUParams) -> jnp.ndarray:
    """Apply ``swish(linear_a(x)) * linear_b(x)``.

    Upstream dispatches to a Triton kernel on CUDA, which computes the same
    product; the reference path is reproduced here.
    """
    return silu(linear(x, params.linear_a)) * linear(x, params.linear_b)


def adaln(
    a: jnp.ndarray, s: jnp.ndarray, params: AdaLNParams, *, eps: float = 1e-5
) -> jnp.ndarray:
    """Apply adaptive layer norm (AF3 Algorithm 26)."""
    a = layer_norm(a, params.layer_norm_a, eps=eps)
    s = layer_norm(s, params.layer_norm_s, eps=eps)
    gate = jax_sigmoid(linear(s, params.linear_g))
    return gate * a + linear(s, params.linear_s)


def swiglu_transition(
    x: jnp.ndarray,
    params: SwiGLUTransitionParams,
    *,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply the SwiGLU transition (AF3 Algorithm 11).

    ``mask`` is ``[..., N]``; upstream expands it to ``[..., N, 1]`` and
    multiplies the output. A missing mask means all-ones, matching upstream.
    """
    y = layer_norm(x, params.layer_norm, eps=eps)
    y = swiglu(y, params.swiglu)
    y = linear(y, params.linear_out)
    if mask is not None:
        y = y * mask[..., None]
    return y


class ConditionedTransitionBlockParams(NamedTuple):
    """Parameters for ``ConditionedTransitionBlock`` (AF3 Algorithm 25).

    ``linear_g`` carries a bias initialized to -2 (``gating_ada_zero``), which is
    what makes the block start near-closed; ``linear_out`` is bias-free.
    """

    layer_norm: AdaLNParams
    swiglu: SwiGLUParams
    linear_g: LinearParams
    linear_out: LinearParams


def conditioned_transition_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    params: ConditionedTransitionBlockParams,
    *,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply an AdaLN-conditioned SwiGLU transition with an AdaLN-zero gate.

    Args:
        a: ``[..., N, C_a]`` activation to update.
        s: ``[..., N, C_s]`` conditioning single representation.
        params: mapped parameters.
        mask: ``[..., N]`` mask; ``None`` means all ones.
        eps: layer norm epsilon.

    Returns:
        ``[..., N, C_a]`` update. Unlike ``swiglu_transition`` there is no
        residual here: the gate replaces it.
    """
    normed = adaln(a, s, params.layer_norm, eps=eps)
    hidden = swiglu(normed, params.swiglu)
    out = jax_sigmoid(linear(s, params.linear_g)) * linear(hidden, params.linear_out)
    if mask is not None:
        out = out * mask[..., None]
    return out
