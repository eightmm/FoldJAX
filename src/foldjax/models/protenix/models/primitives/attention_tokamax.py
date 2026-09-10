"""Tokamax fused attention for the Protenix pair-bias sites.

cuEquivariance covers triangle attention only. The two remaining pair-bias
attentions -- the global token/single attention and the windowed atom attention
of the diffusion atom encoder and decoder -- still form their
``[..., heads, Q, K]`` score tensor in XLA. This module routes those two through
``tokamax.dot_product_attention``, which keeps the scores inside the kernel.

Layout, following ``triangle_attention_tokamax``: tokamax takes ``*B T N H``
(batch, query, head, dim) while the port carries ``[..., H, T, D]``, so callers
swap the two middle axes before and after. ``q`` arrives pre-scaled by
``1/sqrt(d)`` from :func:`prepare_qkv`, hence ``scale=1.0``.

There is no automatic fallback. ``implementation`` is pinned to Triton so a
card that cannot run the kernel says so instead of quietly running XLA under a
name that claims otherwise; :data:`TOKAMAX_IMPLEMENTATION` is the seam CPU
tests move to ``"xla"`` to exercise this same branch off a GPU.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

try:  # pragma: no cover - exercised only where tokamax is installed
    import tokamax

    _TOKAMAX_AVAILABLE = True
except Exception:  # pragma: no cover
    tokamax = None
    _TOKAMAX_AVAILABLE = False


#: Which tokamax implementation the ``tokamax`` attention backend asks for.
#:
#: Triton, not ``None``: tokamax's own ordering advances to the next candidate
#: when one is unsupported, so a box without the kernel would run XLA and
#: report success, and a measurement taken there would be labelled ``tokamax``
#: while comparing XLA against XLA. Tests rebind this to ``"xla"`` to run the
#: same branch on CPU, which is the only supported reason to change it.
TOKAMAX_IMPLEMENTATION = "triton"

_WARNED_FLOAT32 = False


def tokamax_available() -> bool:
    """True if tokamax imported."""

    return _TOKAMAX_AVAILABLE


def tokamax_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    bias: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    implementation: str | None = None,
) -> jnp.ndarray:
    """Fused pair-bias attention in tokamax layout ``[*B, T, H, D]``.

    ``bias`` and ``mask`` are ``[*#B, #H, #T, #S]`` and may broadcast on every
    axis. ``q`` is already scaled. Dtypes are passed through untouched: the
    kernel pays off in bfloat16, and an upcast here would both hide that and
    change what the port computes.

    A query row with no valid key is widened to accept every key rather than
    left as an all-masked softmax, because a fused kernel is entitled to return
    NaN for an empty row and NaN survives the ``* mask`` that zeroes those rows
    downstream.

    What the widened row computes does not matter, and deliberately does not
    try to match the XLA path, which returns a uniform average of ``v`` there:
    its ``-1e10`` is far larger than the logits in bfloat16, so every entry
    rounds to the same number rather than cancelling. At the windowed site an
    empty row can only be a padded or sequence-masked query -- a valid query is
    always a valid key of its own window -- and those rows are sliced off or
    multiplied by zero before anything reads them. The widening is there so
    that what gets discarded is a number and not a NaN.
    """

    if not _TOKAMAX_AVAILABLE:
        msg = "tokamax backend unavailable; cannot run tokamax attention."
        raise RuntimeError(msg)

    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["foldjax.models.protenix"], known_only=True)

    if q.dtype == jnp.float32:
        _warn_float32_inputs()

    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)
        mask = mask | ~jnp.any(mask, axis=-1, keepdims=True)

    return tokamax.dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        scale=1.0,
        implementation=(
            TOKAMAX_IMPLEMENTATION if implementation is None else implementation
        ),
    ).astype(v.dtype)


def _warn_float32_inputs() -> None:
    """Say once that a float32 run is on tokamax's float32 path.

    Not an error: an fp32 x tokamax cell is a measurement someone may want. But
    the kernel is a bfloat16 lever, so a float32 run that was meant to be the
    fast one has to be visible rather than merely slow.
    """

    global _WARNED_FLOAT32
    if _WARNED_FLOAT32:
        return
    _WARNED_FLOAT32 = True
    warnings.warn(
        "tokamax attention received float32 q/k/v and will run tokamax's "
        "float32 path; the fused kernel is a bfloat16 lever, so pass "
        "--amp-policy bf16 (diffusion) or a bfloat16 trunk to reach it",
        UserWarning,
        stacklevel=3,
    )
