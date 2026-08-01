"""Official Chai diffusion atom-attention encoder port.

The exported encoder graph is inlined into ``diffusion_module.forward_*``.
Its local transformer is identical to the token-embedder atom transformer;
this module reuses those verified primitives while implementing the distinct
diffusion conditioning, coordinate, sample-axis, and token-pooling graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from foldjax.models.chai.models.primitives import (
    contiguous_segment_mean,
    layer_norm,
    linear,
)
from foldjax.models.chai.models.token_embedder import (
    AttnParams,
    TransitionParams,
    _att,
    _blocked_bias,
    _kv_idx,
    _local_attention,
    _pair_update_block,
    conditioned_transition,
    map_transition,
)


class DiffusionAtomEncoderParams(NamedTuple):
    to_atom_cond_weight: jnp.ndarray
    token_to_atom_norm_weight: jnp.ndarray
    token_to_atom_norm_bias: jnp.ndarray
    token_to_atom_weight: jnp.ndarray
    prev_pos_weight: jnp.ndarray
    aspp_h_weight: jnp.ndarray
    aspp_w_weight: jnp.ndarray
    atom_pair_mlp0_weight: jnp.ndarray
    atom_pair_mlp2_weight: jnp.ndarray
    transitions: tuple[TransitionParams, ...]
    attns: tuple[AttnParams, ...]
    b2b_ln_weight: jnp.ndarray
    b2b_ln_bias: jnp.ndarray
    b2b_bias_weight: jnp.ndarray
    to_token_single_weight: jnp.ndarray
    token_pair_norm_weight: jnp.ndarray
    token_pair_norm_bias: jnp.ndarray
    token_pair_weight: jnp.ndarray


class DiffusionAtomEncoderOutput(NamedTuple):
    """Encoder values consumed by the token transformer and atom decoder."""

    token_single: jnp.ndarray
    atom_single: jnp.ndarray
    atom_condition: jnp.ndarray
    atom_pair: jnp.ndarray


def map_diffusion_atom_encoder(
    state: Mapping[str, Any], prefix: str = "atom_attention_encoder"
) -> DiffusionAtomEncoderParams:
    """Map all 46 official atom-attention encoder tensors exactly once."""
    ldt = f"{prefix}.atom_transformer.local_diffn_transformer"
    pair_update = f"{prefix}.pair_update_block"
    return DiffusionAtomEncoderParams(
        to_atom_cond_weight=jnp.asarray(state[f"{prefix}.to_atom_cond.weight"]),
        token_to_atom_norm_weight=jnp.asarray(
            state[f"{prefix}.token_to_atom_single.0.weight"]
        ),
        token_to_atom_norm_bias=jnp.asarray(
            state[f"{prefix}.token_to_atom_single.0.bias"]
        ),
        token_to_atom_weight=jnp.asarray(
            state[f"{prefix}.token_to_atom_single.1.weight"]
        ),
        prev_pos_weight=jnp.asarray(state[f"{prefix}.prev_pos_embed.weight"]),
        aspp_h_weight=jnp.asarray(
            state[f"{pair_update}.atom_single_to_atom_pair_proj_h.1.weight"]
        ),
        aspp_w_weight=jnp.asarray(
            state[f"{pair_update}.atom_single_to_atom_pair_proj_w.1.weight"]
        ),
        atom_pair_mlp0_weight=jnp.asarray(
            state[f"{pair_update}.atom_pair_mlp.0.weight"]
        ),
        atom_pair_mlp2_weight=jnp.asarray(
            state[f"{pair_update}.atom_pair_mlp.2.weight"]
        ),
        transitions=tuple(
            map_transition(state, f"{ldt}.transitions.{index}") for index in range(3)
        ),
        attns=tuple(_att(state, index, ldt) for index in range(3)),
        b2b_ln_weight=jnp.asarray(state[f"{ldt}.blocked_pairs2blocked_bias.0.weight"]),
        b2b_ln_bias=jnp.asarray(state[f"{ldt}.blocked_pairs2blocked_bias.0.bias"]),
        b2b_bias_weight=jnp.asarray(
            state[f"{ldt}.blocked_pairs2blocked_bias.1.weight"]
        ),
        to_token_single_weight=jnp.asarray(state[f"{prefix}.to_token_single.0.weight"]),
        token_pair_norm_weight=jnp.asarray(
            state[f"{prefix}.token_pair_to_atom_pair.0.weight"]
        ),
        token_pair_norm_bias=jnp.asarray(
            state[f"{prefix}.token_pair_to_atom_pair.0.bias"]
        ),
        token_pair_weight=jnp.asarray(
            state[f"{prefix}.token_pair_to_atom_pair.1.weight"]
        ),
    )


def diffusion_atom_encoder_with_aux(
    atom_single_input_feats: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    token_pair_repr: jnp.ndarray,
    atom_scaled_coords: jnp.ndarray,
    atom_block_pair_input_feats: jnp.ndarray,
    atom_single_mask: jnp.ndarray,
    atom_block_pair_mask: jnp.ndarray,
    block_indices_h: jnp.ndarray,
    block_indices_w: jnp.ndarray,
    atom_token_indices: jnp.ndarray,
    params: DiffusionAtomEncoderParams,
) -> DiffusionAtomEncoderOutput:
    """Encode atoms and return every value reused by the denoiser.

    ``atom_scaled_coords`` is the already sigma-scaled coordinate tensor used
    at the official encoder boundary, with shape ``[batch, samples, atoms, 3]``.
    """
    p = params
    batch, atoms, _ = atom_single_input_feats.shape
    samples = atom_scaled_coords.shape[1]
    tokens = token_single_trunk_repr.shape[1]

    raw = linear(atom_single_input_feats, p.to_atom_cond_weight)
    token_single = linear(
        layer_norm(
            token_single_trunk_repr.astype(jnp.float32),
            p.token_to_atom_norm_weight,
            p.token_to_atom_norm_bias,
        ),
        p.token_to_atom_weight,
    )
    batch_indices = jnp.arange(batch)[:, None]
    cond = raw + token_single[batch_indices, atom_token_indices]
    cond = layer_norm(cond.astype(jnp.float32))

    position_embedding = linear(atom_scaled_coords, p.prev_pos_weight)
    single = raw[:, None] + position_embedding.astype(jnp.float32)
    cond_samples = jnp.broadcast_to(
        cond[:, None], (batch, samples, atoms, cond.shape[-1])
    )

    h_cond = cond_samples[:, :, block_indices_h]
    w_cond = cond_samples[:, :, block_indices_w]
    token_pair = linear(
        layer_norm(
            token_pair_repr.astype(jnp.float32),
            p.token_pair_norm_weight,
            p.token_pair_norm_bias,
        ),
        p.token_pair_weight,
    )
    h_tokens = atom_token_indices[:, block_indices_h][..., None]
    w_tokens = atom_token_indices[:, block_indices_w][:, :, None]
    pair_batch = jnp.arange(batch)[:, None, None, None]
    atom_pair = token_pair[pair_batch, h_tokens, w_tokens]
    atom_pair = _pair_update_block(
        h_cond, w_cond, atom_block_pair_input_feats + atom_pair, p, linear
    )

    batch_samples = batch * samples
    n_blocks, query_block_size = block_indices_h.shape
    key_block_size = block_indices_w.shape[1]
    single = single.reshape(batch_samples, atoms, 128)
    cond = cond_samples.reshape(batch_samples, atoms, 128)
    single_mask = jnp.broadcast_to(
        atom_single_mask[:, None], (batch, samples, atoms)
    ).reshape(batch_samples, atoms)
    pair_mask = jnp.broadcast_to(
        atom_block_pair_mask[:, None],
        (batch, samples, n_blocks, query_block_size, key_block_size),
    ).reshape(batch_samples, n_blocks, query_block_size, key_block_size)
    atom_pair = atom_pair.reshape(
        batch, samples, n_blocks, query_block_size, key_block_size, 16
    )
    pair = atom_pair.reshape(
        batch_samples, n_blocks, query_block_size, key_block_size, 16
    )
    pair = layer_norm(pair.astype(jnp.float32), p.b2b_ln_weight, p.b2b_ln_bias)
    kv_idx = _kv_idx(atoms, n_blocks, query_block_size, key_block_size)
    single = jnp.where(single_mask[..., None], single, 0.0)

    for index in range(3):
        bias = _blocked_bias(pair, pair_mask, p.b2b_bias_weight[index], linear)
        local = _local_attention(
            single,
            cond,
            bias,
            kv_idx,
            p.attns[index],
            n_blocks,
            linear,
        )
        transition = conditioned_transition(single, cond, p.transitions[index], linear)
        single = single + transition + local
        if index < 2:
            single = jnp.where(single_mask[..., None], single, 0.0)

    decoder_atom_single = single.reshape(batch, samples, atoms, 128)
    pooled_atom_single = jnp.maximum(linear(single, p.to_token_single_weight), 0.0)
    pooled_atom_single = pooled_atom_single * single_mask[..., None].astype(
        pooled_atom_single.dtype
    )
    token_indices = jnp.broadcast_to(
        atom_token_indices[:, None], (batch, samples, atoms)
    ).reshape(batch_samples, atoms)
    pooled = contiguous_segment_mean(
        pooled_atom_single,
        single_mask,
        token_indices,
        num_segments=tokens,
    )
    return DiffusionAtomEncoderOutput(
        token_single=pooled.reshape(batch, samples, tokens, 768),
        atom_single=decoder_atom_single,
        atom_condition=cond_samples[:, 0],
        atom_pair=atom_pair,
    )


def diffusion_atom_encoder(
    atom_single_input_feats: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    token_pair_repr: jnp.ndarray,
    atom_scaled_coords: jnp.ndarray,
    atom_block_pair_input_feats: jnp.ndarray,
    atom_single_mask: jnp.ndarray,
    atom_block_pair_mask: jnp.ndarray,
    block_indices_h: jnp.ndarray,
    block_indices_w: jnp.ndarray,
    atom_token_indices: jnp.ndarray,
    params: DiffusionAtomEncoderParams,
) -> jnp.ndarray:
    """Encode atoms and mean-pool to ``[batch, samples, tokens, 768]``."""
    return diffusion_atom_encoder_with_aux(
        atom_single_input_feats,
        token_single_trunk_repr,
        token_pair_repr,
        atom_scaled_coords,
        atom_block_pair_input_feats,
        atom_single_mask,
        atom_block_pair_mask,
        block_indices_h,
        block_indices_w,
        atom_token_indices,
        params,
    ).token_single
