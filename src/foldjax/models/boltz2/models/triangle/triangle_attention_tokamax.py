"""Tokamax (Triton/Mosaic) fused kernel for Boltz triangle attention.

Mirrors AF3's ``GridSelfAttention``, which runs the pair (axial/triangle)
attention through ``tokamax.dot_product_attention`` with a broadcast pair bias
plus a boolean key mask. Tokamax's API natively expresses Boltz's two-bias
layout WITHOUT materializing the ``[b, i, h, N, N]`` broadcast:

    q/k/v     : [b, i, h, j|k, d]      -> tokamax ``*B T N H`` = [b, i, j, h, d]
    tri_bias  : [b, 1, h, N, N]        -> ``bias [*#B #N #T #S]`` (i broadcast)
    mask_bias : [b, N, 1, 1, N]        -> boolean ``mask`` (h, T broadcast)

The leading ``[b, i]`` are both treated as batch axes (``*B``), and every
``bias``/``mask`` axis is broadcastable (``#``), so the i-broadcast of the pair
bias is never expanded in HBM -- the same memory property the hand-written
Pallas kernel achieves via BlockSpec index maps, but delegated to tokamax.

``q`` arrives already pre-scaled by ``1/sqrt(d)`` from ``_attention`` so we pass
``scale=1.0`` to avoid double scaling.

Import is guarded: on a box without a working tokamax GPU backend the module
still imports; ``tokamax_attention_core`` raises only when actually called.
"""

from __future__ import annotations

import jax.numpy as jnp

try:  # pragma: no cover - exercised only where tokamax is installed
    import tokamax

    _TOKAMAX_AVAILABLE = True
except Exception:  # pragma: no cover
    tokamax = None
    _TOKAMAX_AVAILABLE = False


def tokamax_available() -> bool:
    """True if tokamax imported."""
    return _TOKAMAX_AVAILABLE


def tokamax_attention_core(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    implementation: str | None = "triton",
) -> jnp.ndarray:
    """Drop-in replacement for ``_attention_core`` using tokamax.

    q, k, v   : [b, i, h, j|k, d]  (q pre-scaled by 1/sqrt(d))
    tri_bias  : [b, 1, h, N, N]    (broadcast over i)
    mask_bias : [b, N, 1, 1, N]    (i on axis 1, broadcast over h and j)
    returns out [b, i, h, j, d]
    """
    if not _TOKAMAX_AVAILABLE:
        msg = "tokamax backend unavailable; cannot run tokamax attention."
        raise RuntimeError(msg)

    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["foldjax.models.boltz2"], known_only=True)

    # [b, i, h, j, d] -> tokamax layout [b, i, j(T), h(N_heads), d(H)].
    qt = jnp.swapaxes(q, -3, -2)
    kt = jnp.swapaxes(k, -3, -2)
    vt = jnp.swapaxes(v, -3, -2)

    # bias [*#B #N #T #S]: tri_bias is already [b, i=1, h, N, N]; the size-1 i
    # axis broadcasts over the [b, i] batch without materializing.
    bias = tri_bias

    # mask [*#B #N #T #S]: keep where mask_bias == 0 (drop where -inf-like).
    # mask_bias [b, i, 1, 1, N] -> batch [b, i], heads=1, T=1, S=N.
    mask = mask_bias >= 0.0

    out = tokamax.dot_product_attention(
        qt,
        kt,
        vt,
        bias=bias,
        mask=mask,
        scale=1.0,
        implementation=implementation,
    ).astype(v.dtype)

    # tokamax returns [b, i, j(T), h, d] -> [b, i, h, j, d] (pallas contract).
    return jnp.swapaxes(out, -3, -2)
