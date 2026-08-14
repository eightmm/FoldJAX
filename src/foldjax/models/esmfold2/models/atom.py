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

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from foldjax.models.esmfold2.models.primitives import layer_norm, linear

Params = Mapping[str, jnp.ndarray]

#: float32's `finfo.eps`, which is what `F.rms_norm(..., eps=None)` uses when
#: the tensor is float32. Callers running bfloat16 must pass bfloat16's.
FLOAT32_EPS = float(jnp.finfo(jnp.float32).eps)


def rms_norm(x: jnp.ndarray, eps: float = FLOAT32_EPS) -> jnp.ndarray:
    """`F.rms_norm` with no affine terms."""
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * scale


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary_3d(
    x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> jnp.ndarray:
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
    return jnp.cos(freqs), jnp.sin(freqs)


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

    logits = jnp.einsum("bihd,bjhd->bhij", q, k) * (head_dim**-0.5)
    allowed = sliding_window_mask(valid, half_window)[:, None]
    logits = jnp.where(allowed, logits, jnp.finfo(logits.dtype).min)
    attention = jax.nn.softmax(logits, axis=-1)
    context = jnp.einsum("bhij,bjhd->bihd", attention, v)
    context = context.reshape(batch, length, width)
    context = context * valid[..., None].astype(context.dtype)

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
    normed = layer_norm(
        queries, params[f"{dot}norm.weight"], params[f"{dot}norm.bias"]
    )
    return linear(normed, params, f"{dot}output_linear")
