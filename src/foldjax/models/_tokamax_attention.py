"""Tokamax (Triton/Mosaic) fused kernel for triangle attention, shared by the
ports that have one.

Protenix and Boltz-2 carried byte-identical copies of this wrapper; it moved
here beside ``_cueq.py``, which holds the cuEquivariance counterpart for the
same reason. Each port keeps a re-exporting stub so no call site moved.

Mirrors AF3's ``GridSelfAttention``, which runs the pair (axial/triangle)
attention through ``tokamax.dot_product_attention`` with a broadcast pair bias
plus a boolean key mask. Tokamax's API natively expresses the two-bias layout
WITHOUT materializing the ``[b, i, h, N, N]`` broadcast:

    q/k/v     : [b, i, h, j|k, d]      -> tokamax ``*B T N H`` = [b, i, j, h, d]
    tri_bias  : [b, 1, h, N, N]        -> ``bias [*#B #N #T #S]`` (i broadcast)
    mask_bias : [b, N, 1, 1, N]        -> boolean ``mask`` (h, T broadcast)

The leading ``[b, i]`` are both treated as batch axes (``*B``), and every
``bias``/``mask`` axis is broadcastable (``#``), so the i-broadcast of the pair
bias is never expanded in HBM -- the same memory property the hand-written
Pallas kernel achieves via BlockSpec index maps, but delegated to tokamax.

``q`` arrives already pre-scaled by ``1/sqrt(d)`` from the caller so we pass
``scale=1.0`` to avoid double scaling.

Import is guarded: on a box without a working tokamax GPU backend the module
still imports; ``tokamax_attention_core`` raises only when actually called.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: Why ``import tokamax`` failed, kept so that a run which explicitly selects
#: the kernel can show the real cause (a missing CUDA library, a version skew)
#: instead of only "unavailable".
_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised only where tokamax is installed
    import tokamax

    _TOKAMAX_AVAILABLE = True
except Exception as error:  # pragma: no cover  # noqa: BLE001 - kept, re-raised on use
    tokamax = None
    _TOKAMAX_AVAILABLE = False
    _IMPORT_ERROR = error


def tokamax_available() -> bool:
    """True if tokamax imported."""
    return _TOKAMAX_AVAILABLE


def tokamax_import_error() -> Exception | None:
    """The exception ``import tokamax`` raised in this process, if it did."""
    return _IMPORT_ERROR


def tokamax_unavailable(what: str) -> RuntimeError:
    """The error for an explicit tokamax selection, naming the import failure.

    Raise it ``from tokamax_import_error()`` so the original traceback stays
    attached; the message carries the cause for a one-line CLI report too.
    """
    cause = _IMPORT_ERROR
    detail = (
        "it is not importable"
        if cause is None
        else f"importing it failed with {type(cause).__name__}: {cause}"
    )
    return RuntimeError(f"{what} needs the tokamax package; {detail}")


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
        raise tokamax_unavailable("tokamax attention") from _IMPORT_ERROR

    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["foldjax.models.boltz2"], known_only=True)

    from foldjax import _pallas_gpu

    _pallas_gpu.install()
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


_MIN_ATTENTION_BLOCK = 16
_MAX_ATTENTION_BLOCK = 64
_HIGHEST_MATMUL_PRECISIONS = ("highest", "float32", "F32_F32_F32")


def should_pad_attention_to_block(q: jnp.ndarray) -> bool:
    """True only where GPU measurement showed ragged-tile padding pays off.

    Tokamax's ragged Triton tile penalty appears on ``float32`` inputs at IEEE
    ``float32`` (``highest``) matmul precision; ``bfloat16`` and TF32
    (default/``high``) runs have no such penalty and run slower if padded.
    """
    return (
        jnp.dtype(q.dtype) == jnp.dtype(jnp.float32)
        and jax.config.jax_default_matmul_precision in _HIGHEST_MATMUL_PRECISIONS
    )


def tokamax_attention_block(length: int) -> int:
    """Return the tile ``tokamax``'s Triton FlashAttention picks for ``length``.

    ``PallasTritonFlashAttention._get_heuristics_config``
    (``tokamax/_src/ops/attention/pallas_triton.py``, lines 651-653) clamps
    ``next_power_of_2(seq_len)`` to ``[16, 64]`` for both ``block_q`` and
    ``block_k``, independent of operand dtype.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        msg = f"attention sequence length must be a positive int; got {length!r}"
        raise ValueError(msg)
    pow2 = 1 << (length - 1).bit_length()
    return min(max(pow2, _MIN_ATTENTION_BLOCK), _MAX_ATTENTION_BLOCK)


def pad_attention_to_block(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    *,
    block: int | None = None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray | None,
    jnp.ndarray | None,
    int,
]:
    """Pad tokamax ``[*B, T|S, H, D]`` operands to a tile multiple.

    Returns ``(q, k, v, bias, mask, orig_q_len)``. When both query and key
    lengths already divide their target block, returns the inputs unchanged
    with no JAX operations. Broadcast ``bias`` axes (size 1 or omitted leading
    axes) stay unmaterialized; padded key columns are masked ``False``.
    """
    if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
        raise ValueError("attention q, k, and v must have rank at least 3")
    if q.ndim != k.ndim or k.ndim != v.ndim:
        raise ValueError("attention q, k, and v must have matching rank")
    n_q = q.shape[-3]
    n_k = k.shape[-3]
    if v.shape[-3] != n_k:
        msg = f"attention k and v sequence lengths must match: {n_k} != {v.shape[-3]}"
        raise ValueError(msg)
    if n_q <= 0 or n_k <= 0:
        msg = f"attention sequence lengths must be positive; got q={n_q}, k={n_k}"
        raise ValueError(msg)

    if block is None:
        block_q = tokamax_attention_block(n_q)
        block_k = tokamax_attention_block(n_k)
    else:
        if not isinstance(block, int) or isinstance(block, bool) or block <= 0:
            msg = f"attention block must be a positive int; got {block!r}"
            raise ValueError(msg)
        block_q = block
        block_k = block

    if bias is not None:
        if bias.ndim > q.ndim:
            raise ValueError("attention bias rank must not exceed query rank")
        if bias.ndim >= 2 and bias.shape[-2] not in (1, n_q):
            msg = f"attention bias query axis {bias.shape[-2]} incompatible with {n_q}"
            raise ValueError(msg)
        if bias.ndim >= 1 and bias.shape[-1] not in (1, n_k):
            msg = f"attention bias col axis {bias.shape[-1]} incompatible with {n_k}"
            raise ValueError(msg)

    if mask is not None:
        if mask.ndim > q.ndim:
            raise ValueError("attention mask rank must not exceed query rank")
        if mask.ndim >= 2 and mask.shape[-2] not in (1, n_q):
            msg = f"attention mask query axis {mask.shape[-2]} incompatible with {n_q}"
            raise ValueError(msg)
        if mask.ndim >= 1 and mask.shape[-1] not in (1, n_k):
            msg = f"attention mask col axis {mask.shape[-1]} incompatible with {n_k}"
            raise ValueError(msg)

    pad_q = (-n_q) % block_q
    pad_k = (-n_k) % block_k
    if pad_q == 0 and pad_k == 0:
        return q, k, v, bias, mask, n_q

    if pad_q > 0:
        q_pads = ((0, 0),) * (q.ndim - 3) + ((0, pad_q), (0, 0), (0, 0))
        q = jnp.pad(q, q_pads)
    if pad_k > 0:
        kv_pads = ((0, 0),) * (k.ndim - 3) + ((0, pad_k), (0, 0), (0, 0))
        k = jnp.pad(k, kv_pads)
        v = jnp.pad(v, kv_pads)

    if bias is not None:
        bias_pad_q = pad_q if (bias.ndim >= 2 and bias.shape[-2] != 1) else 0
        bias_pad_k = pad_k if (bias.ndim >= 1 and bias.shape[-1] != 1) else 0
        if bias_pad_q > 0 or bias_pad_k > 0:
            if bias.ndim == 1:
                bias_pads = ((0, bias_pad_k),)
            else:
                bias_pads = ((0, 0),) * (bias.ndim - 2) + (
                    (0, bias_pad_q),
                    (0, bias_pad_k),
                )
            bias = jnp.pad(bias, bias_pads)

    padded_k = n_k + pad_k
    if mask is None:
        if pad_k > 0:
            valid_cols = jnp.arange(padded_k) < n_k
            mask = valid_cols.reshape((1,) * (q.ndim - 1) + (padded_k,))
    else:
        mask = jnp.asarray(mask, dtype=bool)
        if pad_k > 0 and (mask.ndim == 0 or mask.shape[-1] != n_k):
            valid_cols = jnp.arange(padded_k) < n_k
            if mask.ndim == 0:
                mask = (mask & valid_cols).reshape((1,) * (q.ndim - 1) + (padded_k,))
            else:
                mask = mask & valid_cols
        mask_pad_q = pad_q if (mask.ndim >= 2 and mask.shape[-2] != 1) else 0
        mask_pad_k = pad_k if (mask.ndim >= 1 and mask.shape[-1] == n_k) else 0
        if mask_pad_q > 0 or mask_pad_k > 0:
            if mask.ndim == 1:
                mask_pads = ((0, mask_pad_k),)
            else:
                mask_pads = ((0, 0),) * (mask.ndim - 2) + (
                    (0, mask_pad_q),
                    (0, mask_pad_k),
                )
            mask = jnp.pad(mask, mask_pads, constant_values=False)

    return q, k, v, bias, mask, n_q
