"""The atom stack: 3D rotary attention over a sliding window.

This is the least AlphaFold-like part of ESMFold2 and the easiest to get
subtly wrong, so each convention is stated where it is used:

* **RoPE is three-dimensional plus one discrete axis.** The angles come from
  the *reference conformer* coordinates and `ref_space_uid`, not from any
  sequence position, and `build_3d_rope`'s signature defaults are dead code --
  the call site passes `(2, 10, 20.0, 10000.0)`, which fills `half_dim` exactly.
* **The rotation is half-split**, `cos` repeated as `concat([c, c])`, pairing
  `(j, j + half)`. Interleaved pairing is the other common convention and
  produces plausible garbage.
* **The window is measured in packed rank**, so padding atoms cost no window
  budget, and the diagonal is always allowed -- that is what keeps a fully
  padded row's softmax finite.
* **`qk_norm` is RMS norm with no learnable scale and torch's `eps=None`,
  which resolves to `finfo(dtype).eps`** -- 7.8e-3 in bfloat16 against 1.2e-7
  in float32. The dtype changes the arithmetic here, not just the rounding, so
  the port takes it as an argument rather than hard-coding one.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from foldjax.models.esmfold2.models.primitives import layer_norm, linear

Params = Mapping[str, jnp.ndarray]

#: float32's `finfo.eps`, which is what `F.rms_norm(..., eps=None)` uses when
#: the tensor is float32. Callers running bfloat16 must pass bfloat16's.
FLOAT32_EPS = float(jnp.finfo(jnp.float32).eps)

#: Query rows per block in the windowed attention. The block reads
#: `rows_per_block + 2 * half_window` keys, so a smaller block wastes less of
#: what it reads (1.98x at 128 against 2.98x at 256, for `half_window=64`) and
#: pays for it in scan steps and smaller matmuls.
#:
#: **Measured, it is neither a memory knob nor a time one.** At 1,003 residues
#: and 32 samples, 128 and 256 give the same arena to the MiB and the same peak
#: to three decimals (24.460 GiB), and their times differ by 0.17 s -- 0.4%,
#: against run-to-run spreads of 0.60 and 2.10 s, so the arms overlap. Less
#: wasted arithmetic and fewer scan steps evidently cancel, or both sit under
#: the noise. An earlier version of this comment called it "a time knob, not a
#: memory one"; that was inference from the arithmetic and the timing does not
#: support it.
#:
#: So 256 is the default because nothing distinguishes it, not because it won.
#: The environment override exists as an escape hatch for a size nobody has
#: measured, and it should not be reached for expecting a result here.
ATOM_ROWS_PER_BLOCK = 256

ATOM_ATTENTION_BACKENDS = frozenset({"dense", "blocked"})


def _atom_attention_backend() -> str:
    """Return the configured atom-attention backend.

    `dense` builds the whole `[batch, heads, atoms, atoms]` score matrix and
    masks a band 129 wide out of it. `blocked` walks the query rows in blocks
    and never materialises more than one block's scores: at 1,003 residues and
    the released 32 samples that is 35,945 -> 11,312 MiB, 68.5%.

    **The default is `blocked`, and it was `dense` until the two were timed.**
    That was the right default to set at the time and the wrong one to keep:
    it was set pending a measurement, and the measurement arrived. Shipped
    settings, shipped allocator, 1,003 residues, 32 samples, three timed runs
    per arm with arms alternating and warmups discarded:

        dense    49.30 s  +-0.20   48.516 GiB
        blk256   46.73 s  +-0.60   24.460 GiB
        blk128   46.90 s  +-2.10   24.460 GiB

    Blocked wins both axes: 5.2% faster and 24.056 GiB smaller. The time
    separation is clean rather than marginal -- the slowest blocked run beats
    the fastest dense one, so no pairing of individual runs reverses it -- and
    the clock control points the conservative way, since dense held the
    *highest* mean SM clock (1749 against 1711) and still lost. The card
    favoured the loser, so the margin understates.

    The worry that motivated the cautious default was real and did not
    materialise: the blocked form trades one large matmul for 31 scanned small
    ones, so its cost should grow where its memory win shrinks. It may still,
    at a size nobody has run; it does not at the released one.

    There is no `auto`. A mode that picked a path from a probe would put two
    machines on two kernels under one name, which is the failure this
    repository keeps paying for.
    """
    backend = os.environ.get("ESMFOLD2_ATOM_ATTENTION_BACKEND", "blocked").lower()
    if backend not in ATOM_ATTENTION_BACKENDS:
        raise ValueError(
            f"unsupported atom attention backend: {backend!r}; "
            f"expected one of {sorted(ATOM_ATTENTION_BACKENDS)}"
        )
    return backend


def _resolve_rows_per_block(rows_per_block: int | None) -> int:
    """Rows per block, or zero for the dense path.

    An explicit argument wins over the environment, so a caller that has
    measured its own size is not overridden by a process-wide setting.
    """
    if rows_per_block is not None:
        return rows_per_block
    if _atom_attention_backend() == "dense":
        return 0
    return int(os.environ.get("ESMFOLD2_ATOM_ROWS_PER_BLOCK", ATOM_ROWS_PER_BLOCK))


def rms_norm(x: jnp.ndarray, eps: float = FLOAT32_EPS) -> jnp.ndarray:
    """`F.rms_norm` with no affine terms."""
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * scale


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary_3d(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Rotate `[B, L, H, D]` by per-position `cos`/`sin` of width `D/2`."""
    width = cos.shape[-1] * 2
    cos = jnp.concatenate([cos, cos], axis=-1)[:, :, None, :]
    sin = jnp.concatenate([sin, sin], axis=-1)[:, :, None, :]
    rotated = x[..., :width] * cos + _rotate_half(x[..., :width]) * sin
    return jnp.concatenate([rotated, x[..., width:]], axis=-1)


def build_3d_rope(
    ref_pos: jnp.ndarray,
    ref_space_uid: jnp.ndarray,
    *,
    head_dim: int,
    n_spatial_per_axis: int = 2,
    n_uid_pairs: int = 10,
    spatial_base_frequency: float = 20.0,
    uid_base_frequency: float = 10000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Angles from three Cartesian axes and one discrete axis.

    Spatial frequencies are axis-major and frequency-minor -- `[x f0, x f1,
    y f0, y f1, z f0, z f1]` -- and the uid block follows. Shorter than
    `head_dim // 2` is right-zero-padded, which for the released widths is
    never needed: 3*2 + 10 fills 16 exactly.
    """
    half = head_dim // 2
    spatial_inv = spatial_base_frequency ** (
        -jnp.arange(n_spatial_per_axis, dtype=jnp.float32) / n_spatial_per_axis
    )
    uid_inv = uid_base_frequency ** (
        -jnp.arange(n_uid_pairs, dtype=jnp.float32) / n_uid_pairs
    )
    spatial = jnp.einsum(
        "bna,k->bnak", ref_pos.astype(jnp.float32), spatial_inv
    ).reshape(ref_pos.shape[0], ref_pos.shape[1], 3 * n_spatial_per_axis)
    uid = jnp.einsum("bn,k->bnk", ref_space_uid.astype(jnp.float32), uid_inv)
    freqs = jnp.concatenate([spatial, uid], axis=-1)
    if freqs.shape[-1] < half:
        freqs = jnp.pad(freqs, ((0, 0), (0, 0), (0, half - freqs.shape[-1])))
    # Angles in float32, tables in bfloat16, exactly as upstream
    # (`modeling_esmfold2_common.py:513-514`) -- hardcoded there, not a knob.
    # Returning float32 tables here promoted `q` and `k` inside the rotary, so
    # a bfloat16 caller got float32 back out of the attention block. That is a
    # separate defect from the missing q/k/v cast below: this one decides the
    # dtype the block *returns*, that one decides the dtype the scores are
    # *computed* at.
    return jnp.cos(freqs).astype(jnp.bfloat16), jnp.sin(freqs).astype(jnp.bfloat16)


def sliding_window_mask(valid: jnp.ndarray, half_window: int) -> jnp.ndarray:
    """Who may attend to whom, in packed rank rather than raw index.

    `rank` counts valid atoms only, so padding between two atoms does not push
    them out of each other's window. The diagonal is forced on: without it a
    fully padded row would softmax over nothing.
    """
    rank = jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1
    within = jnp.abs(rank[:, :, None] - rank[:, None, :]) <= half_window
    allowed = within & valid[:, :, None].astype(bool) & valid[:, None, :].astype(bool)
    eye = jnp.eye(valid.shape[1], dtype=bool)[None]
    return allowed | eye


def _windowed_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    valid: jnp.ndarray,
    *,
    half_window: int,
    rows_per_block: int,
    scale: float,
) -> jnp.ndarray:
    """Sliding-window attention that never materialises the full score matrix.

    The dense path builds `[batch, heads, atoms, atoms]` and then throws away
    everything outside a band 129 wide. At the released 32 samples and 7,776
    atoms that tensor is 14,762 MiB, twice over with its softmax, against a
    35,945 MiB arena. This walks the query rows in blocks instead, so only one
    block's scores exist at a time.

    **Exactness rests on `atom_attention_mask` being a prefix**, which it is:
    `features.py` has a single writer and sets `mask[index] = True` for every
    real atom in order. That makes `rank[i] == i` throughout the valid region,
    so the window in packed rank -- which is what the model means -- coincides
    with a window in raw index, and a slice of `rows_per_block + 2 *
    half_window` keys provably contains every key a query in the block may
    attend to. The mask inside the block is still computed from `rank`, so if
    that precondition ever breaks the block masks correctly over the keys it
    holds rather than silently changing the semantics.

    Upstream leans on the same packing: its fast path is `flash_attn_varlen`
    with `cu_seqlens`, and it keeps a dense fallback for when that is
    unavailable. `swa_attention` keeps one for the same reason.

    `dynamic_slice` clamps an out-of-range start rather than truncating, which
    *shifts* the window. That is safe here and the clamp is written out rather
    than left to the primitive, because the mask has to be built against the
    indices actually taken: at the low edge the needed keys are `[0, R + h)`
    and at the high edge `[bR - h, atoms)`, both inside a slice of this width.
    """
    batch, length, n_heads, head_dim = q.shape
    keys_per_block = rows_per_block + 2 * half_window
    n_blocks = -(-length // rows_per_block)
    padded = n_blocks * rows_per_block

    rank = jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1
    pad = ((0, 0), (0, padded - length), (0, 0), (0, 0))
    q_padded = jnp.pad(q, pad)
    rank_padded = jnp.pad(rank, pad[:2])
    valid_padded = jnp.pad(valid, pad[:2])
    # The last start that still leaves a full slice; `length >= keys_per_block`
    # is the caller's static precondition for taking this path at all.
    last_start = length - keys_per_block
    offsets = jnp.arange(keys_per_block)

    def one_block(_, block: jnp.ndarray):
        row = block * rows_per_block
        start = jnp.clip(row - half_window, 0, last_start)

        def take_q(t):
            return jax.lax.dynamic_slice_in_dim(t, row, rows_per_block, 1)

        def take_k(t):
            return jax.lax.dynamic_slice_in_dim(t, start, keys_per_block, 1)

        logits = jnp.einsum("bihd,bjhd->bhij", take_q(q_padded), take_k(k)) * scale
        rank_q, rank_k = take_q(rank_padded), take_k(rank)
        within = jnp.abs(rank_q[:, :, None] - rank_k[:, None, :]) <= half_window
        allowed = (
            within
            & take_q(valid_padded)[:, :, None].astype(bool)
            & take_k(valid)[:, None, :].astype(bool)
        )
        # The diagonal, in absolute index. A padded query has no key to match
        # and masks out entirely; softmax subtracts the row max first, so the
        # row is uniform rather than NaN, and it is discarded below.
        diagonal = (row + jnp.arange(rows_per_block))[:, None] == start + offsets
        logits = jnp.where(
            (allowed | diagonal)[:, None], logits, jnp.finfo(logits.dtype).min
        )
        attention = jax.nn.softmax(logits, axis=-1)
        return None, jnp.einsum("bhij,bjhd->bihd", attention, take_k(v))

    _, blocks = jax.lax.scan(one_block, None, jnp.arange(n_blocks))
    context = blocks.swapaxes(0, 1).reshape(batch, padded, n_heads, head_dim)
    return context[:, :length]


def swa_attention(
    x: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    valid: jnp.ndarray,
    n_heads: int,
    half_window: int,
    rows_per_block: int | None = None,
    eps: float = FLOAT32_EPS,
) -> jnp.ndarray:
    """`SWA3DRoPEAttention`.

    Queries and keys are RMS-normed and rotated; values are neither. The gate
    is computed from the block's *input*, before the normalisation the caller
    applied, and multiplies the context before the output projection.
    """
    dot = f"{prefix}." if prefix else ""
    batch, length, width = x.shape
    head_dim = width // n_heads

    qkv = linear(x, params, f"{dot}Wqkv").reshape(batch, length, 3, n_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = apply_rotary_3d(rms_norm(q, eps), cos, sin)
    k = apply_rotary_3d(rms_norm(k, eps), cos, sin)

    # Upstream casts the attention inputs to bfloat16 here unconditionally
    # (`modeling_esmfold2_common.py:573-575`) and restores the entering dtype
    # after the context (`:632`). Both halves are needed, and this port had
    # neither.
    #
    # `trunk_dtype` does not reach this module: `TRUNK_PREFIXES` scopes it to
    # the trunk and deliberately excludes the diffusion stack, so `Wqkv` runs
    # float32 @ float32 and `q` is float32 before the rotary is ever applied.
    # The scores inherit that -- `f32[128, 7776, 7776]` is 29,524 MiB at the
    # released 32 samples and 7776 atoms, and the softmax of it is another,
    # together 90% of the whole temp arena.
    #
    # The softmax still accumulates at the narrowed dtype, where upstream's
    # `scaled_dot_product_attention` accumulates in float32 without ever
    # materialising the scores. Matching that needs a fused kernel rather than
    # a cast; the window is +-`half_window`, so at most 129 terms are summed.
    input_dtype = q.dtype
    if input_dtype not in (jnp.float16, jnp.bfloat16):
        q = q.astype(jnp.bfloat16)
        k = k.astype(jnp.bfloat16)
        v = v.astype(jnp.bfloat16)

    # Blocked when there are enough atoms for a whole key slice to exist, dense
    # otherwise -- the comparison is on static shapes, so one of the two is
    # traced and the other is not. Short inputs take the dense path because a
    # slice wider than the array cannot be clamped into range, not because the
    # blocked result would differ.
    rows_per_block = _resolve_rows_per_block(rows_per_block)
    if rows_per_block and length >= rows_per_block + 2 * half_window:
        context = _windowed_attention(
            q,
            k,
            v,
            valid,
            half_window=half_window,
            rows_per_block=rows_per_block,
            scale=head_dim**-0.5,
        )
    else:
        logits = jnp.einsum("bihd,bjhd->bhij", q, k) * (head_dim**-0.5)
        allowed = sliding_window_mask(valid, half_window)[:, None]
        logits = jnp.where(allowed, logits, jnp.finfo(logits.dtype).min)
        attention = jax.nn.softmax(logits, axis=-1)
        context = jnp.einsum("bhij,bjhd->bihd", attention, v)
    context = context.reshape(batch, length, width)
    context = context * valid[..., None].astype(context.dtype)
    context = context.astype(input_dtype)

    gate = jax.nn.sigmoid(linear(x, params, f"{dot}gate_proj"))
    return linear(context * gate, params, f"{dot}out_proj")


def swa_block(
    x: jnp.ndarray,
    conditioning: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    valid: jnp.ndarray,
    n_heads: int,
    half_window: int,
    eps: float = FLOAT32_EPS,
) -> jnp.ndarray:
    """`SWAAtomBlock`: adaLN-modulated attention and feed-forward.

    The modulation splits as `shift, scale, gate` per branch -- attention
    first, then the feed-forward. `scale, shift` is the other plausible order
    and the shapes do not distinguish them.
    """
    dot = f"{prefix}." if prefix else ""
    modulation = linear(jax.nn.silu(conditioning), params, f"{dot}adaln_modulation.1")
    if modulation.ndim == 2:
        modulation = modulation[:, None, :]
    chunks = jnp.split(modulation, 6, axis=-1)
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = chunks

    normed = rms_norm(x, eps) * (1.0 + scale_a) + shift_a
    x = x + gate_a * swa_attention(
        normed,
        params,
        f"{dot}attn",
        cos=cos,
        sin=sin,
        valid=valid,
        n_heads=n_heads,
        half_window=half_window,
        eps=eps,
    )
    normed = rms_norm(x, eps) * (1.0 + scale_f) + shift_f
    return x + gate_f * swiglu_ffn(normed, params, f"{dot}ffn")


def swiglu_ffn(x: jnp.ndarray, params: Params, prefix: str = "") -> jnp.ndarray:
    """`SwiGLUFFN`: one packed up-projection, silu-gated, projected down."""
    dot = f"{prefix}." if prefix else ""
    packed = linear(x, params, f"{dot}w_up")
    half = packed.shape[-1] // 2
    return linear(
        jax.nn.silu(packed[..., :half]) * packed[..., half:], params, f"{dot}w_down"
    )


def atom_transformer(
    x: jnp.ndarray,
    conditioning: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    valid: jnp.ndarray,
    n_blocks: int,
    n_heads: int,
    half_window: int,
    eps: float = FLOAT32_EPS,
) -> jnp.ndarray:
    """`SWAAtomTransformer`: the blocks in sequence, nothing else."""
    dot = f"{prefix}." if prefix else ""
    for index in range(n_blocks):
        x = swa_block(
            x,
            conditioning,
            params,
            f"{dot}blocks.{index}",
            cos=cos,
            sin=sin,
            valid=valid,
            n_heads=n_heads,
            half_window=half_window,
            eps=eps,
        )
    return x


def one_hot_atom_features(
    ref_element: jnp.ndarray,
    ref_atom_name_chars: jnp.ndarray,
    atom_mask: jnp.ndarray,
    *,
    max_atomic_number: int = 128,
    char_vocab: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One-hot the element and atom-name codes, and zero the padding.

    Upstream does this in the model rather than the atom encoder, and it zeroes
    both against the atom mask first: the projections downstream are bias-free,
    so a padded atom that kept its one-hot would contribute a real vector.
    """
    element = jax.nn.one_hot(ref_element.astype(jnp.int32), max_atomic_number)
    chars = jax.nn.one_hot(ref_atom_name_chars.astype(jnp.int32), char_vocab)
    mask = atom_mask.astype(element.dtype)
    return element * mask[..., None], chars * mask[..., None, None]


def atom_features(
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    atom_mask: jnp.ndarray,
    ref_element_one_hot: jnp.ndarray,
    ref_atom_name_chars_one_hot: jnp.ndarray,
) -> jnp.ndarray:
    """The 389-wide atom feature: 3 + 1 + 1 + 128 + 4*64.

    Both categorical blocks arrive already one-hot, the way the atom encoder
    receives them -- see `one_hot_atom_features` for the step above.
    """
    chars = ref_atom_name_chars_one_hot
    chars = chars.reshape(chars.shape[:-2] + (chars.shape[-2] * chars.shape[-1],))
    return jnp.concatenate(
        [
            ref_pos.astype(jnp.float32),
            ref_charge.astype(jnp.float32)[..., None],
            atom_mask.astype(jnp.float32)[..., None],
            ref_element_one_hot.astype(jnp.float32),
            chars.astype(jnp.float32),
        ],
        axis=-1,
    )


def scatter_atom_to_token(
    values: jnp.ndarray, atom_to_token: jnp.ndarray, n_tokens: int, mask: jnp.ndarray
) -> jnp.ndarray:
    """Masked mean of each token's atoms.

    Upstream scatters with `include_self=False`, so a token with no valid atom
    comes out zero rather than carrying whatever the destination held.
    """
    weights = mask.astype(values.dtype)[..., None]
    one_hot = jax.nn.one_hot(atom_to_token, n_tokens, dtype=values.dtype)
    totals = jnp.einsum("bnt,bnd->btd", one_hot, values * weights)
    spread = jnp.broadcast_to(weights, values.shape)
    counts = jnp.einsum("bnt,bnd->btd", one_hot, spread)
    return jnp.where(counts > 0, totals / jnp.maximum(counts, 1e-12), 0.0)


def atom_conditioning(
    ref_pos: jnp.ndarray,
    atom_mask: jnp.ndarray,
    ref_space_uid: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_element_one_hot: jnp.ndarray,
    ref_atom_name_chars_one_hot: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    n_heads: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The step-invariant half of the atom encoder: conditioning and rotary tables.

    Everything here is a function of the *reference* conformer, so a diffusion
    sampler that recomputes it once per denoiser call is rebuilding the same
    389-wide feature tensor forty-eight times. Upstream caches it under
    `inference_cache["atomencoder"]`; this port hands it back so the caller can
    hold it outside the loop.
    """
    dot = f"{prefix}." if prefix else ""
    features = atom_features(
        ref_pos,
        ref_charge,
        atom_mask,
        ref_element_one_hot,
        ref_atom_name_chars_one_hot,
    )
    conditioning = layer_norm(
        linear(features, params, f"{dot}atom_linear"),
        params[f"{dot}atom_norm.weight"],
        params[f"{dot}atom_norm.bias"],
    )
    cos, sin = build_3d_rope(
        ref_pos, ref_space_uid, head_dim=conditioning.shape[-1] // n_heads
    )
    return conditioning, cos, sin


def atom_encoder(
    ref_pos: jnp.ndarray | None,
    atom_mask: jnp.ndarray,
    ref_space_uid: jnp.ndarray | None,
    ref_charge: jnp.ndarray | None,
    ref_element_one_hot: jnp.ndarray | None,
    ref_atom_name_chars_one_hot: jnp.ndarray | None,
    atom_to_token: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    n_blocks: int,
    n_heads: int,
    half_window: int,
    coords: jnp.ndarray | None = None,
    precomputed: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    n_tokens: int | None = None,
    eps: float = FLOAT32_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """`ESMFold2AtomEncoder`.

    Returns the token representation, the atom queries the decoder resumes
    from, the conditioning it reuses, and the rotary tables -- which are
    step-invariant, and which upstream caches for exactly that reason.

    `coords` is the noisy structure in the diffusion path and absent in the
    inputs embedder; upstream concatenates a zero `pred_r1` beside it, so the
    projection's last three columns never see a non-zero value.

    `precomputed` is `atom_conditioning`'s result, already broadcast to
    however many diffusion samples the caller is running; the reference
    features are then unread and may be `None`. `n_tokens` must be given
    whenever `atom_to_token` is a tracer, since its maximum is otherwise read
    on the host.
    """
    dot = f"{prefix}." if prefix else ""
    if precomputed is None:
        conditioning, cos, sin = atom_conditioning(
            ref_pos,
            atom_mask,
            ref_space_uid,
            ref_charge,
            ref_element_one_hot,
            ref_atom_name_chars_one_hot,
            params,
            prefix,
            n_heads=n_heads,
        )
    else:
        conditioning, cos, sin = precomputed

    queries = conditioning
    if coords is not None:
        padded = jnp.concatenate([coords, jnp.zeros_like(coords)], axis=-1)
        queries = queries + linear(padded, params, f"{dot}coords_linear")

    queries = atom_transformer(
        queries,
        conditioning,
        params,
        f"{dot}atom_transformer",
        cos=cos,
        sin=sin,
        valid=atom_mask,
        n_blocks=n_blocks,
        n_heads=n_heads,
        half_window=half_window,
        eps=eps,
    )
    if n_tokens is None:
        n_tokens = int(atom_to_token.max()) + 1
    projected = jax.nn.relu(linear(queries, params, f"{dot}atom_to_token_linear"))
    tokens = scatter_atom_to_token(projected, atom_to_token, n_tokens, atom_mask)
    return tokens, queries, conditioning, (cos, sin)


def atom_decoder(
    tokens: jnp.ndarray,
    queries: jnp.ndarray,
    conditioning: jnp.ndarray,
    atom_to_token: jnp.ndarray,
    atom_mask: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    rope: Sequence[jnp.ndarray],
    n_blocks: int,
    n_heads: int,
    half_window: int,
    eps: float = FLOAT32_EPS,
) -> jnp.ndarray:
    """`ESMFold2AtomDecoder`: token update gathered back to atoms, then coords."""
    dot = f"{prefix}." if prefix else ""
    gathered = jnp.take_along_axis(
        linear(tokens, params, f"{dot}token_to_atom_linear"),
        atom_to_token[..., None],
        axis=1,
    )
    queries = queries + gathered
    queries = atom_transformer(
        queries,
        conditioning,
        params,
        f"{dot}atom_transformer",
        cos=rope[0],
        sin=rope[1],
        valid=atom_mask,
        n_blocks=n_blocks,
        n_heads=n_heads,
        half_window=half_window,
        eps=eps,
    )
    normed = layer_norm(queries, params[f"{dot}norm.weight"], params[f"{dot}norm.bias"])
    return linear(normed, params, f"{dot}output_linear")
