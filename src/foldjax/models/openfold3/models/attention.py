"""Multi-head attention matching ``openfold3.core.model.primitives.Attention``.

Upstream can dispatch to DeepSpeed, cuEquivariance, Triton or low-memory
attention kernels. All of them compute the same function as the stock path, which
is what this module reproduces; kernel selection is a performance concern to be
handled separately once the trunk exists.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.primitives import LinearParams, jax_sigmoid, linear


class AttentionParams(NamedTuple):
    """Parameters for ``Attention``.

    Every projection is bias-free upstream (``mha_init``). ``linear_g`` is
    present only when the module was built with ``gating=True``.
    """

    linear_q: LinearParams
    linear_k: LinearParams
    linear_v: LinearParams
    linear_o: LinearParams
    linear_g: LinearParams | None = None


def split_heads(x: jnp.ndarray, no_heads: int) -> jnp.ndarray:
    """Reshape ``[..., N, H * C]`` to ``[..., N, H, C]``.

    Mirrors upstream's ``view(shape[:-1] + (no_heads, -1))``: the head axis is
    the outer factor, so a flat projection splits into contiguous per-head
    blocks. Getting this order wrong still produces correct shapes, which is why
    the parity gate matters here.
    """
    if x.shape[-1] % no_heads:
        raise ValueError(
            f"channel dimension {x.shape[-1]} is not divisible by {no_heads} heads"
        )
    return x.reshape(x.shape[:-1] + (no_heads, x.shape[-1] // no_heads))


def flatten_heads(x: jnp.ndarray) -> jnp.ndarray:
    """Reshape ``[..., N, H, C]`` back to ``[..., N, H * C]``."""
    return x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],))


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    biases: tuple[jnp.ndarray, ...] = (),
    *,
    high_precision: bool = False,
) -> jnp.ndarray:
    """Biased softmax attention over ``[..., H, N, C]`` tensors.

    ``query`` is expected to be pre-scaled, matching upstream's ``_prep_qkv``.
    When ``high_precision`` is true, ``query``, ``key``, ``value`` and each
    bias are cast to ``float32`` before the logit contraction and softmax and
    the weighted sum is cast back to ``query.dtype``, matching upstream's
    ``_attention(..., use_high_precision=True)`` under ``bf16-mixed``.
    """
    in_dtype = query.dtype
    if high_precision and in_dtype != jnp.float32:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)
        value = value.astype(jnp.float32)
        biases = tuple(bias.astype(jnp.float32) for bias in biases)
    scores = jnp.einsum("...qc,...kc->...qk", query, key)
    for bias in biases:
        scores = scores + bias
    scores = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("...qk,...kc->...qc", scores, value)
    if high_precision and out.dtype != in_dtype:
        out = out.astype(in_dtype)
    return out


def attention(
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    params: AttentionParams,
    *,
    no_heads: int,
    biases: tuple[jnp.ndarray, ...] = (),
    high_precision: bool = False,
) -> jnp.ndarray:
    """Apply gated multi-head attention.

    Args:
        q_x: ``[..., Q, C_q]`` query data.
        kv_x: ``[..., K, C_k]`` key/value data.
        params: mapped ``Attention`` parameters.
        no_heads: head count; it cannot be inferred from the flat projections.
        biases: tensors broadcasting to ``[..., H, Q, K]``.
        high_precision: evaluate the attention core (logits, biases, softmax
            and value contraction) in ``float32``, casting back before gating.

    Returns:
        ``[..., Q, C_q]`` attention update.
    """
    query = split_heads(linear(q_x, params.linear_q), no_heads)
    key = split_heads(linear(kv_x, params.linear_k), no_heads)
    value = split_heads(linear(kv_x, params.linear_v), no_heads)

    # [..., H, N, C_hidden]
    query = jnp.swapaxes(query, -2, -3)
    key = jnp.swapaxes(key, -2, -3)
    value = jnp.swapaxes(value, -2, -3)

    # Upstream scales by the per-head hidden dimension.
    query = query / math.sqrt(query.shape[-1])

    out = dot_product_attention(
        query, key, value, biases, high_precision=high_precision
    )

    # [..., Q, H, C_hidden]
    out = jnp.swapaxes(out, -2, -3)

    if params.linear_g is not None:
        gate = jax_sigmoid(linear(q_x, params.linear_g))
        out = out * split_heads(gate, no_heads)

    return linear(flatten_heads(out), params.linear_o)
