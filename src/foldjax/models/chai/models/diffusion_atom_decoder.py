"""JAX port of Chai-1's diffusion ``atom_attention_decoder``.

The official TorchScript component inlines the decoder into each size-bucketed
forward.  This module implements only that inlined subgraph: token-to-atom
conditioning, three local atom-transformer blocks, and the final 3D update
projection.  Sampling, diffusion conditioning, and the atom encoder remain out
of scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.chai.models.primitives import layer_norm, linear
from foldjax.models.chai.models.token_embedder import (
    TransitionParams,
    conditioned_transition,
    map_transition,
)

_PREFIX = "atom_attention_decoder"
_BASE = f"{_PREFIX}.atom_transformer.local_diffn_transformer"
_MASK_FILL = -10000.0


class DecoderAttentionParams(NamedTuple):
    lin_s_merged_weight: jnp.ndarray
    to_qkv_weight: jnp.ndarray
    q_bias: jnp.ndarray
    out_proj_weight: jnp.ndarray
    out_proj_bias: jnp.ndarray


class AtomDecoderParams(NamedTuple):
    token_to_atom_weight: jnp.ndarray
    b2b_ln_weight: jnp.ndarray
    b2b_ln_bias: jnp.ndarray
    b2b_bias_weight: jnp.ndarray
    transitions: tuple[TransitionParams, ...]
    attentions: tuple[DecoderAttentionParams, ...]
    pos_ln_weight: jnp.ndarray
    pos_ln_bias: jnp.ndarray
    pos_weight: jnp.ndarray


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        f"{_PREFIX}.token_to_atom.weight": (128, 768),
        f"{_BASE}.blocked_pairs2blocked_bias.0.weight": (16,),
        f"{_BASE}.blocked_pairs2blocked_bias.0.bias": (16,),
        f"{_BASE}.blocked_pairs2blocked_bias.1.weight": (3, 4, 16),
        f"{_PREFIX}.to_pos_updates.0.weight": (128,),
        f"{_PREFIX}.to_pos_updates.0.bias": (128,),
        f"{_PREFIX}.to_pos_updates.1.weight": (3, 128),
    }
    for index in range(3):
        transition = f"{_BASE}.transitions.{index}"
        shapes.update(
            {
                f"{transition}.ada_ln.lin_s_merged.weight": (256, 128),
                f"{transition}.linear_a_nobias_double.weight": (512, 128),
                f"{transition}.linear_b_nobias.weight": (128, 256),
                f"{transition}.linear_s_biasinit_m2.weight": (128, 128),
                f"{transition}.linear_s_biasinit_m2.bias": (128,),
            }
        )
        attention = f"{_BASE}.local_attentions.{index}"
        shapes.update(
            {
                f"{attention}.q_bias": (4, 32),
                f"{attention}.single_layer_norm.lin_s_merged.weight": (256, 128),
                f"{attention}.to_qkv.weight": (3, 4, 32, 128),
                f"{attention}.out_proj.weight": (128, 128),
                f"{attention}.out_proj.bias": (128,),
            }
        )
    return shapes


_EXPECTED_SHAPES = _expected_shapes()


def _attention(state: Mapping[str, Any], index: int) -> DecoderAttentionParams:
    prefix = f"{_BASE}.local_attentions.{index}"
    return DecoderAttentionParams(
        lin_s_merged_weight=jnp.asarray(
            state[f"{prefix}.single_layer_norm.lin_s_merged.weight"]
        ),
        to_qkv_weight=jnp.asarray(state[f"{prefix}.to_qkv.weight"]),
        q_bias=jnp.asarray(state[f"{prefix}.q_bias"]),
        out_proj_weight=jnp.asarray(state[f"{prefix}.out_proj.weight"]),
        out_proj_bias=jnp.asarray(state[f"{prefix}.out_proj.bias"]),
    )


def map_atom_decoder(state: Mapping[str, Any]) -> AtomDecoderParams:
    """Map all 37 official decoder tensors, rejecting ABI drift."""
    actual = {key for key in state if key.startswith(f"{_PREFIX}.")}
    expected = set(_EXPECTED_SHAPES)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"atom decoder state mismatch: missing={missing}, unexpected={unexpected}"
        )
    for key, shape in _EXPECTED_SHAPES.items():
        actual_shape = tuple(state[key].shape)
        if actual_shape != shape:
            raise ValueError(
                f"wrong shape for {key}: expected {shape}, got {actual_shape}"
            )

    return AtomDecoderParams(
        token_to_atom_weight=jnp.asarray(state[f"{_PREFIX}.token_to_atom.weight"]),
        b2b_ln_weight=jnp.asarray(
            state[f"{_BASE}.blocked_pairs2blocked_bias.0.weight"]
        ),
        b2b_ln_bias=jnp.asarray(
            state[f"{_BASE}.blocked_pairs2blocked_bias.0.bias"]
        ),
        b2b_bias_weight=jnp.asarray(
            state[f"{_BASE}.blocked_pairs2blocked_bias.1.weight"]
        ),
        transitions=tuple(
            map_transition(state, f"{_BASE}.transitions.{index}")
            for index in range(3)
        ),
        attentions=tuple(_attention(state, index) for index in range(3)),
        pos_ln_weight=jnp.asarray(state[f"{_PREFIX}.to_pos_updates.0.weight"]),
        pos_ln_bias=jnp.asarray(state[f"{_PREFIX}.to_pos_updates.0.bias"]),
        pos_weight=jnp.asarray(state[f"{_PREFIX}.to_pos_updates.1.weight"]),
    )


def _kv_indices(atom_count: int, block_count: int, query: int, keys: int):
    query_indices = jnp.arange(atom_count).reshape(block_count, query)
    start = query_indices[:, :1] + (query - keys) // 2
    return (start + jnp.arange(keys)) % atom_count


def _attention_forward(
    single: jnp.ndarray,
    cond: jnp.ndarray,
    bias: jnp.ndarray,
    kv_indices: jnp.ndarray,
    params: DecoderAttentionParams,
) -> jnp.ndarray:
    batch, atom_count, _ = single.shape
    heads, head_dim = params.q_bias.shape
    block_count, query, keys = bias.shape[2:]

    feat = layer_norm(single, eps=0.1)
    scale, shift = jnp.split(linear(cond, params.lin_s_merged_weight), 2, axis=-1)
    feat = feat * (scale + 1.0) + shift
    qkv = jnp.einsum(
        "nhdc,bac->nbhad",
        params.to_qkv_weight,
        feat,
        precision=jax.lax.Precision.HIGHEST,
    )
    q, key, value = qkv
    q = q + params.q_bias[None, :, None, :]
    q = q.reshape(batch * heads, block_count, query, head_dim)
    key = key.reshape(batch * heads, atom_count, head_dim)[:, kv_indices]
    value = value.reshape(batch * heads, atom_count, head_dim)[:, kv_indices]
    mask = bias.reshape(batch * heads, block_count, query, keys)

    logits = jnp.einsum(
        "...qd,...kd->...qk", q, key, precision=jax.lax.Precision.HIGHEST
    )
    logits = logits / jnp.sqrt(jnp.asarray(head_dim, logits.dtype)) + mask
    probabilities = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum(
        "...qk,...kd->...qd",
        probabilities,
        value,
        precision=jax.lax.Precision.HIGHEST,
    )
    output = output.reshape(batch, heads, block_count, query, head_dim)
    output = jnp.transpose(output, (0, 2, 3, 1, 4)).reshape(
        batch, atom_count, heads * head_dim
    )
    gate = jax.nn.sigmoid(
        linear(cond, params.out_proj_weight, params.out_proj_bias)
    )
    return output * gate


def atom_decoder_forward(
    token_single: jnp.ndarray,
    atom_single: jnp.ndarray,
    atom_cond: jnp.ndarray,
    atom_pair: jnp.ndarray,
    atom_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    atom_token_indices: jnp.ndarray,
    params: AtomDecoderParams,
) -> jnp.ndarray:
    """Return the decoder's unit coordinate update with shape ``(B,S,A,3)``."""
    batch, samples, token_count, token_channels = token_single.shape
    _, _, atom_count, atom_channels = atom_single.shape
    _, _, block_count, query, keys, pair_channels = atom_pair.shape
    if token_channels != 768 or atom_channels != 128 or pair_channels != 16:
        raise ValueError("invalid atom decoder channel dimensions")
    if block_count * query != atom_count:
        raise ValueError("atom blocks must cover the atom dimension exactly")

    flat_batch = batch * samples
    token_single = token_single.reshape(flat_batch, token_count, token_channels)
    single = atom_single.reshape(flat_batch, atom_count, atom_channels)
    cond = jnp.broadcast_to(
        atom_cond[:, None], (batch, samples, atom_count, atom_channels)
    ).reshape(flat_batch, atom_count, atom_channels)
    pair = atom_pair.reshape(flat_batch, block_count, query, keys, pair_channels)
    single_mask = jnp.broadcast_to(
        atom_mask[:, None], (batch, samples, atom_count)
    ).reshape(flat_batch, atom_count)
    block_mask = jnp.broadcast_to(
        pair_mask[:, None], (batch, samples, block_count, query, keys)
    ).reshape(flat_batch, block_count, query, keys)
    token_indices = jnp.broadcast_to(
        atom_token_indices[:, None], (batch, samples, atom_count)
    ).reshape(flat_batch, atom_count)

    token_atoms = linear(token_single, params.token_to_atom_weight)
    single = single + token_atoms[jnp.arange(flat_batch)[:, None], token_indices]
    single = jnp.where(single_mask[..., None], single, 0.0)
    pair = layer_norm(
        pair, params.b2b_ln_weight, params.b2b_ln_bias, eps=1e-5
    )
    kv_indices = _kv_indices(atom_count, block_count, query, keys)

    for index in range(3):
        bias = jnp.einsum(
            "blqkc,hc->bhlqk",
            pair,
            params.b2b_bias_weight[index],
            precision=jax.lax.Precision.HIGHEST,
        )
        bias = jnp.where(block_mask[:, None], bias, _MASK_FILL)
        local = _attention_forward(
            single, cond, bias, kv_indices, params.attentions[index]
        )
        transition = conditioned_transition(
            single, cond, params.transitions[index], linear
        )
        single = single + transition + local
        if index < 2:
            single = jnp.where(single_mask[..., None], single, 0.0)

    output = layer_norm(
        single, params.pos_ln_weight, params.pos_ln_bias, eps=1e-5
    )
    output = linear(output, params.pos_weight)
    return output.reshape(batch, samples, atom_count, 3)
