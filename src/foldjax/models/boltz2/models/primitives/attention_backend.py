"""Attention backend helpers."""

from __future__ import annotations

import jax.numpy as jnp


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

    return tokamax.dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask=mask[:, None, None, :].astype(bool),
        scale=scale,
        implementation=implementation,
    ).astype(v.dtype)


# Back-compat alias: the attention "tokamax" backend was historically "flash".
flash_dot_product_attention = tokamax_dot_product_attention
