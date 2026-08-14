"""ESMC, the protein language model ESMFold2 reads its representations from.

An ordinary pre-norm transformer encoder -- RoPE, QK layer norm, SwiGLU -- at
6B parameters and 80 layers. ESMFold2 does not use its last layer: the shim
softmaxes over **all 81** hidden states, the embedding included, so this
returns the whole stack rather than an output.

Four details are not the defaults a reader would assume:

* the residual is **divided** by `sqrt(n_layers / 36)`, ESM3's depth scaling.
  With 80 layers that is 1.49, so a port that omits it runs every residual
  branch half again too strong.
* `q_ln` and `k_ln` normalise across the **whole** `d_model`, before the split
  into heads -- not per head, which is the more common arrangement and gives
  different numbers for the same weights.
* the chain mask is `sequence_id[i] == sequence_id[j]`, and padding is marked
  `-1`. `-1 == -1` is true, so **padding attends to padding**. It is harmless,
  the results are discarded, and it is reproduced rather than tidied because
  "fixing" it changes nothing except agreement.
* positions for RoPE run over the whole padded input, not per chain, so the
  second chain of a complex starts at whatever offset the first one ended at.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.esmfold2.models.primitives import layer_norm, linear

Params = Mapping[str, jnp.ndarray]

#: The tokenizer's specials, which the chain packing below writes by hand.
BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MASK_TOKEN_ID = 32


@dataclass(frozen=True)
class ESMCSettings:
    """ESMC-6B's shape. The 300M and 600M releases differ only in these."""

    vocab_size: int = 64
    d_model: int = 2560
    n_heads: int = 40
    n_layers: int = 80
    rope_base: float = 10000.0
    #: ESM3's depth scaling, `sqrt(n_layers / 36)`, applied as a divisor.
    scale_residue: bool = True

    @property
    def residual_scale(self) -> float:
        return float(np.sqrt(self.n_layers / 36)) if self.scale_residue else 1.0


def settings_from_config(config: Mapping[str, object]) -> ESMCSettings:
    """Read an `ESMCSettings` out of an ESMC `config.json`."""
    base = ESMCSettings()

    def number(name: str, fallback: int) -> int:
        value = config.get(name, fallback)
        return fallback if value is None else int(value)  # type: ignore[arg-type]

    return ESMCSettings(
        vocab_size=number("vocab_size", base.vocab_size),
        d_model=number("d_model", base.d_model),
        n_heads=number("n_heads", base.n_heads),
        n_layers=number("n_layers", base.n_layers),
    )


def rotary_tables(
    length: int, head_dim: int, base: float = 10000.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`cos` and `sin` of shape `[length, head_dim // 2]`.

    Built in float32 whatever the model dtype: upstream keeps `inv_freq` in
    float32 for exactly this reason, since a bfloat16 frequency table drifts
    measurably across eighty layers.
    """
    inverse = 1.0 / (
        base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    angles = jnp.outer(jnp.arange(length, dtype=jnp.float32), inverse)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary(
    x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> jnp.ndarray:
    """Rotate `[B, L, H, D]`, half-split rather than interleaved."""
    width = cos.shape[-1] * 2
    cos = jnp.concatenate([cos, cos], axis=-1)[None, :, None, :]
    sin = jnp.concatenate([sin, sin], axis=-1)[None, :, None, :]
    head = x[..., :width]
    half = width // 2
    rotated = jnp.concatenate([-head[..., half:], head[..., :half]], axis=-1)
    return jnp.concatenate([head * cos + rotated * sin, x[..., width:]], axis=-1)


def attention(
    x: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    n_heads: int,
    sequence_id: jnp.ndarray | None,
    rope: tuple[jnp.ndarray, jnp.ndarray],
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`MultiHeadAttention`: fused LN+QKV, whole-width QK norm, RoPE, SDPA."""
    dot = f"{prefix}." if prefix else ""
    batch, length, width = x.shape
    head_dim = width // n_heads

    normed = layer_norm(
        x,
        params[f"{dot}layernorm_qkv.layer_norm_weight"],
        params[f"{dot}layernorm_qkv.layer_norm_bias"],
        eps=eps,
    )
    packed = jnp.matmul(normed, params[f"{dot}layernorm_qkv.weight"].T)
    query, key, value = jnp.split(packed, 3, axis=-1)

    # Across the full d_model, before the head split.
    query = layer_norm(query, params[f"{dot}q_ln.weight"], eps=eps)
    key = layer_norm(key, params[f"{dot}k_ln.weight"], eps=eps)

    query = apply_rotary(query.reshape(batch, length, n_heads, head_dim), *rope)
    key = apply_rotary(key.reshape(batch, length, n_heads, head_dim), *rope)
    value = value.reshape(batch, length, n_heads, head_dim)

    logits = jnp.einsum("bihd,bjhd->bhij", query, key) * (head_dim**-0.5)
    if sequence_id is not None:
        same = sequence_id[:, None, :, None] == sequence_id[:, None, None, :]
        logits = jnp.where(same, logits, jnp.finfo(logits.dtype).min)
    weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
    context = jnp.einsum("bhij,bjhd->bihd", weights, value)
    return linear(context.reshape(batch, length, width), params, f"{dot}out_proj")


def feed_forward(
    x: jnp.ndarray, params: Params, prefix: str, *, eps: float = 1e-5
) -> jnp.ndarray:
    """The fused LayerNorm + SwiGLU MLP, with upstream's flat parameter names."""
    dot = f"{prefix}." if prefix else ""
    normed = layer_norm(
        x,
        params[f"{dot}layer_norm_weight"],
        params[f"{dot}layer_norm_bias"],
        eps=eps,
    )
    packed = jnp.matmul(normed, params[f"{dot}fc1_weight"].T)
    half = packed.shape[-1] // 2
    gated = jax.nn.silu(packed[..., :half]) * packed[..., half:]
    return jnp.matmul(gated, params[f"{dot}fc2_weight"].T)


def block(
    x: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    n_heads: int,
    sequence_id: jnp.ndarray | None,
    rope: tuple[jnp.ndarray, jnp.ndarray],
    residual_scale: float,
) -> jnp.ndarray:
    """`UnifiedTransformerBlock`, residuals divided by the depth scale."""
    dot = f"{prefix}." if prefix else ""
    x = x + attention(
        x,
        params,
        f"{dot}attn",
        n_heads=n_heads,
        sequence_id=sequence_id,
        rope=rope,
    ) / residual_scale
    return x + feed_forward(x, params, f"{dot}ffn") / residual_scale


def encode(
    input_ids: jnp.ndarray,
    sequence_id: jnp.ndarray | None,
    params: Params,
    prefix: str = "",
    *,
    settings: ESMCSettings,
) -> jnp.ndarray:
    """Every hidden state, stacked `[n_layers + 1, B, L, d_model]`.

    Entry `i < n_layers` is the *input* to block `i` -- so entry 0 is the token
    embedding untouched -- and the last is the final layer norm's output. That
    is the stack `LanguageModelShim` softmaxes over, and taking only the last
    layer, or dropping the embedding, silently changes what the trunk reads.
    """
    dot = f"{prefix}." if prefix else ""
    x = params[f"{dot}embed.weight"][input_ids]
    rope = rotary_tables(
        input_ids.shape[1], settings.d_model // settings.n_heads, settings.rope_base
    )
    scale = settings.residual_scale

    collected = [x]
    for index in range(settings.n_layers):
        x = block(
            x,
            params,
            f"{dot}transformer.blocks.{index}",
            n_heads=settings.n_heads,
            sequence_id=sequence_id,
            rope=rope,
            residual_scale=scale,
        )
        if index + 1 < settings.n_layers:
            collected.append(x)
    collected.append(
        layer_norm(x, params[f"{dot}transformer.norm.weight"])
    )
    return jnp.stack(collected, axis=0)


# ---------------------------------------------------------------------------
# The ESMFold2 side: packing tokens into an LM input and scattering back
# ---------------------------------------------------------------------------


def pack_lm_inputs(
    input_ids: np.ndarray,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    mol_type: np.ndarray,
    token_mask: np.ndarray,
    *,
    packed_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn structure tokens into one LM input per batch entry.

    Two things happen here and both are easy to skip. Modified residues are
    **atom-tokenised** -- HYP, MSE, ACE and the rest occupy several structure
    tokens sharing one `(asym_id, residue_index)` -- and the LM was trained on
    one token per residue, so they are collapsed first and the hidden states
    scattered back afterwards. And chains are packed as
    `[BOS] chain [EOS BOS] chain ... [EOS]`, so a two-chain complex carries
    four special tokens, not two.

    Returns `(lm_input_ids, sequence_id, expand_map)`, where `expand_map` sends
    each structure token to its position in the LM input and `-1` marks the
    tokens -- padding and non-protein -- that the LM never saw.

    Host-side on purpose: the collapse is a unique-and-argsort whose output
    shape depends on the values, which is not a traceable computation.
    """
    batch, n_tokens = input_ids.shape
    protein = (mol_type == 0) & token_mask.astype(bool)

    sequences: list[np.ndarray] = []
    expand_maps = np.full((batch, n_tokens), -1, dtype=np.int64)

    for index in range(batch):
        selected = protein[index]
        ids = input_ids[index][selected]
        asym = asym_id[index][selected]
        residues = residue_index[index][selected]

        keys = np.stack([asym, residues], axis=1)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        n_unique = int(inverse.max()) + 1 if inverse.size else 0
        positions = np.arange(keys.shape[0], dtype=np.int64)
        first = np.full(n_unique, keys.shape[0], dtype=np.int64)
        np.minimum.at(first, inverse, positions)
        order = np.argsort(first)
        collapsed_ids = ids[first[order]]
        collapsed_asym = asym[first[order]]
        remap = np.empty_like(order)
        remap[order] = np.arange(n_unique, dtype=np.int64)
        inverse_ordered = remap[inverse]

        parts = [np.array([BOS_TOKEN_ID], dtype=ids.dtype)]
        lm_positions = np.empty(n_unique, dtype=np.int64)
        cursor = 1
        chains = np.unique(collapsed_asym)
        for chain_number, chain in enumerate(chains):
            in_chain = np.nonzero(collapsed_asym == chain)[0]
            parts.append(collapsed_ids[in_chain])
            lm_positions[in_chain] = np.arange(cursor, cursor + in_chain.shape[0])
            cursor += in_chain.shape[0]
            if chain_number < len(chains) - 1:
                parts.append(np.array([EOS_TOKEN_ID, BOS_TOKEN_ID], dtype=ids.dtype))
                cursor += 2
        parts.append(np.array([EOS_TOKEN_ID], dtype=ids.dtype))
        sequences.append(np.concatenate(parts))

        expand_maps[index, np.nonzero(selected)[0]] = lm_positions[inverse_ordered]

    natural_length = max(sequence.shape[0] for sequence in sequences)
    if packed_length is not None and packed_length < natural_length:
        raise ValueError(
            "ESMC packed length cannot be smaller than the real packed input: "
            f"{packed_length} < {natural_length}"
        )
    length = natural_length if packed_length is None else packed_length
    lm_input_ids = np.full((batch, length), PAD_TOKEN_ID, dtype=input_ids.dtype)
    for index, sequence in enumerate(sequences):
        lm_input_ids[index, : sequence.shape[0]] = sequence

    # Chains are numbered by counting the BOS tokens seen so far; padding gets
    # -1, which makes padding attend only to padding.
    sequence_id = np.cumsum(lm_input_ids == BOS_TOKEN_ID, axis=1) - 1
    sequence_id = np.where(lm_input_ids == PAD_TOKEN_ID, -1, sequence_id)
    return lm_input_ids, sequence_id, expand_maps


def hidden_states_for_tokens(
    hidden: jnp.ndarray, expand_map: np.ndarray, n_tokens: int
) -> jnp.ndarray:
    """Scatter `[n_layers + 1, B, L_lm, D]` back to `[B, n_tokens, n_layers+1, D]`.

    Tokens the LM never saw -- padding, ligands, nucleic acids -- stay zero,
    which is what the shim's layer-norm then sees for them.
    """
    layers, batch, _, width = hidden.shape
    gather = jnp.asarray(np.where(expand_map < 0, 0, expand_map))
    keep = jnp.asarray(expand_map >= 0)[..., None, None]
    del batch, layers, width, n_tokens
    rows = jnp.take_along_axis(
        jnp.transpose(hidden, (1, 2, 0, 3)), gather[:, :, None, None], axis=1
    )
    return jnp.where(keep, rows, 0.0)


def packed_lm_length(
    input_ids: np.ndarray,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    mol_type: np.ndarray,
    token_mask: np.ndarray,
) -> int:
    """Natural ESMC length after residue collapse and chain special tokens."""

    packed, _sequence_id, _expand_map = pack_lm_inputs(
        input_ids, asym_id, residue_index, mol_type, token_mask
    )
    return int(packed.shape[1])


def lm_hidden_states(
    input_ids: np.ndarray,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    mol_type: np.ndarray,
    token_mask: np.ndarray,
    params: Params,
    prefix: str = "",
    *,
    settings: ESMCSettings,
    packed_length: int | None = None,
) -> jnp.ndarray:
    """`compute_lm_hidden_states`: pack, run ESMC, scatter back."""
    lm_input_ids, sequence_id, expand_map = pack_lm_inputs(
        input_ids,
        asym_id,
        residue_index,
        mol_type,
        token_mask,
        packed_length=packed_length,
    )
    hidden = encode(
        jnp.asarray(lm_input_ids),
        jnp.asarray(sequence_id),
        params,
        prefix,
        settings=settings,
    )
    return hidden_states_for_tokens(hidden, expand_map, input_ids.shape[1])


__all__ = [
    "BOS_TOKEN_ID",
    "EOS_TOKEN_ID",
    "ESMCSettings",
    "MASK_TOKEN_ID",
    "PAD_TOKEN_ID",
    "apply_rotary",
    "attention",
    "block",
    "encode",
    "feed_forward",
    "hidden_states_for_tokens",
    "lm_hidden_states",
    "packed_lm_length",
    "pack_lm_inputs",
    "rotary_tables",
    "settings_from_config",
]
