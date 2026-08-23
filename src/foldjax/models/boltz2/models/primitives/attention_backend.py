"""Attention backend helpers."""

from __future__ import annotations

import jax.numpy as jnp


def masked_softmax(
    logits: jnp.ndarray,
    key_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Softmax over valid keys, returning exact zero for an empty key row.

    ``key_mask`` may have any shape broadcastable to ``logits``.  A finite
    negative mask is not sufficient here: when every key is masked it produces
    a uniform distribution, while a plain ``softmax(-inf)`` produces NaNs.
    Choosing a finite maximum and denominator only for non-empty rows keeps the
    ordinary softmax reduction unchanged and makes the empty-row result zero.
    """

    valid = jnp.asarray(key_mask, dtype=bool)
    masked_logits = jnp.where(valid, logits, -jnp.inf)
    has_valid_key = jnp.any(valid, axis=-1, keepdims=True)
    maximum = jnp.max(masked_logits, axis=-1, keepdims=True)
    safe_maximum = jnp.where(has_valid_key, maximum, jnp.zeros_like(maximum))
    shifted = jnp.where(valid, masked_logits - safe_maximum, -jnp.inf)
    numerator = jnp.exp(shifted)
    denominator = jnp.sum(numerator, axis=-1, keepdims=True)
    safe_denominator = jnp.where(
        has_valid_key,
        denominator,
        jnp.ones_like(denominator),
    )
    return numerator / safe_denominator


def tokamax_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    scale: float,
    backend: str = "xla",
    implementation: object | None = None,
) -> jnp.ndarray:
    """Run flash/fused dot-product attention via tokamax.

    Shapes follow JAX/tokamax convention:
    q/k/v: [batch, query_or_key, heads, dim]
    bias: [batch, heads, query, key]
    mask: [batch, key]

    The inputs are passed in their EXISTING dtype -- no fp32 upcast. tokamax /
    FlashAttention (triton) kernels only pay off in low precision, so to use a
    fast kernel run the sampler with ``compute_dtype=float16``/``bfloat16``
    (then q/k/v arrive low-precision here). Forcing fp32 (the previous
    behaviour) pushed tokamax onto a slow fallback far slower than plain XLA;
    upcasting also breaks fp16 matmul on CPU. fp16 is preferred over bf16 on
    this model (smaller sampling drift).

    ``implementation=None`` lets tokamax auto-select the best kernel for the
    shape/hardware and fall back to plain XLA when a custom kernel does not help
    (small head_dim, short sequence, large dense pair bias). The previous hard
    ``("triton","cudnn","xla_chunked")`` tuple forced the slowest chunked path
    when triton/cudnn were unavailable.
    """

    if backend not in ("tokamax", "flash"):
        msg = f"Unsupported attention backend: {backend!r}"
        raise ValueError(msg)

    import tokamax
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["foldjax.models.boltz2"], known_only=True)

    key_mask = mask[:, None, None, :].astype(bool)
    empty = ~jnp.any(key_mask, axis=-1, keepdims=True)
    # Fused kernels are allowed to assume that every batch has at least one
    # valid key. Give empty batches a harmless temporary key so the kernel
    # never forms an all-masked softmax, then restore the public zero contract.
    safe_mask = key_mask.at[..., 0].set(key_mask[..., 0] | empty[..., 0])
    out = tokamax.dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask=safe_mask,
        scale=scale,
        implementation=implementation,
    ).astype(v.dtype)
    return jnp.where(empty, jnp.zeros_like(out), out)


# Back-compat alias: the attention "tokamax" backend was historically "flash".
flash_dot_product_attention = tokamax_dot_product_attention
