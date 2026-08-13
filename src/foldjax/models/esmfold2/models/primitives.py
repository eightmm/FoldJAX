"""The small pieces every ESMFold2 block is built from.

Each function takes the parameters as the mapping upstream's `state_dict`
uses, so a block's parameters are that block's sub-tree and nothing has to be
renamed twice. Every one of these is checked against the torch module it
mirrors in `tests/models/esmfold2/test_primitives_parity.py`.

Two conventions, both upstream's rather than this project's:

* `nn.Linear` stores its weight transposed for a right-multiply, so a linear
  is `x @ w.T`, and this module does not pre-transpose anything on load --
  the checkpoint keys stay recognisable against upstream's own source.
* SwiGLU packs its two projections into one `w12` and splits the result, so
  the split point is `hidden_features` and the *first* half is the one that
  passes through silu.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

Params = Mapping[str, jnp.ndarray]


def linear(x: jnp.ndarray, params: Params, prefix: str) -> jnp.ndarray:
    """`nn.Linear`, bias included when the checkpoint has one."""
    out = jnp.matmul(x, params[f"{prefix}.weight"].T)
    bias = params.get(f"{prefix}.bias")
    return out if bias is None else out + bias


def layer_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray | None = None,
    bias: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`F.layer_norm` over the last axis, with optional affine terms.

    Upstream calls this three ways -- with both terms, with a weight alone
    (`AdaptiveLayerNorm`'s conditioning path), and with neither (its activation
    path) -- so both are optional rather than there being three functions.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    out = (x - mean) * jax.lax.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def swiglu(x: jnp.ndarray, params: Params, prefix: str = "") -> jnp.ndarray:
    """`SwiGLU`: packed `w12`, split, silu the first half, gate, project.

    The split point comes from the checkpoint rather than a configuration
    value: `w12` is `[2 * hidden, in]`, so hidden is half its first axis.
    """
    dot = f"{prefix}." if prefix else ""
    packed = linear(x, params, f"{dot}w12")
    hidden = packed.shape[-1] // 2
    gate, value = packed[..., :hidden], packed[..., hidden:]
    return linear(jax.nn.silu(gate) * value, params, f"{dot}w3")


def transition_layer(
    x: jnp.ndarray, params: Params, prefix: str = "", eps: float = 1e-5
) -> jnp.ndarray:
    """`TransitionLayer`: norm, two projections, silu gate, project back.

    The same arithmetic as `swiglu` with the projections kept apart and a
    normalisation in front; upstream keeps both forms, so this port does too
    rather than unifying them and having to un-unify them at load time.
    """
    dot = f"{prefix}." if prefix else ""
    x = layer_norm(
        x, params[f"{dot}norm.weight"], params[f"{dot}norm.bias"], eps=eps
    )
    a = linear(x, params, f"{dot}a_proj")
    b = linear(x, params, f"{dot}b_proj")
    return linear(jax.nn.silu(a) * b, params, f"{dot}out_proj")


def adaptive_layer_norm(
    a: jnp.ndarray,
    s: jnp.ndarray,
    params: Params,
    prefix: str = "",
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`AdaptiveLayerNorm` (adaLN-Zero).

    The activation is normalised without affine terms and the conditioning
    with a weight but no bias; the gate is a sigmoid and multiplies, the shift
    adds. Getting the two normalisations the same way round is the whole of
    this function -- they use different parameter sets on purpose.
    """
    dot = f"{prefix}." if prefix else ""
    a_norm = layer_norm(a, eps=eps)
    s_norm = layer_norm(s, params[f"{dot}s_scale"], eps=eps)
    gate = jax.nn.sigmoid(linear(s_norm, params, f"{dot}s_gate"))
    return gate * a_norm + linear(s_norm, params, f"{dot}s_shift")


def fourier_embedding(
    t_hat: jnp.ndarray, params: Params, prefix: str = ""
) -> jnp.ndarray:
    """`FourierEmbedding`: cos(2 pi (t w + b)), with w and b as buffers.

    They are `register_buffer`s rather than parameters -- drawn once at
    construction and frozen -- so they travel in the checkpoint and are read
    from it here rather than being redrawn, which would silently change the
    embedding.
    """
    dot = f"{prefix}." if prefix else ""
    w = params[f"{dot}w"]
    b = params[f"{dot}b"]
    t = jnp.reshape(jnp.asarray(t_hat, dtype=w.dtype), (-1,))
    return jnp.cos(2.0 * jnp.pi * (t[:, None] * w[None, :] + b[None, :]))
