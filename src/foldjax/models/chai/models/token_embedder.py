"""Chai-1 token_embedder.pt port (JAX).

token_embedder is the largest leaf component so far (43 tensors): an atom encoder
with a local attention transformer (``local_diffn_transformer``) whose blocks are
AdaLN-conditioned SwiGLU transitions + local attention, followed by atom->token
aggregation and the output projections (token_single/pair trunk + structure).

The exported forward graph is fully inlined and leaf submodules expose no
callable ``.forward``/``.code``, so sub-blocks cannot be invoked in isolation.
Their math is instead recovered from the application sites in the top
``forward_256.code`` (read directly, not re-derived from architecture priors).

This module ports the first sub-block: the AdaLN-conditioned SwiGLU transition
(``...local_diffn_transformer.transitions.N``), the reusable building block of
the atom transformer. Recovered graph (per forward_256.code):

    feat  = layer_norm(x, eps=0.1, no affine)          # over last axis
    scale, shift = chunk(lin_s_merged(cond), 2, -1)    # AdaLN params
    feat  = feat * (scale + 1.0) + shift
    a, gate = chunk(linear_a_nobias_double(feat), 2, -1)
    h     = silu(a) * gate                             # SwiGLU
    out_gate = sigmoid(linear_s_biasinit_m2(cond))
    trans = out_gate * linear_b_nobias(h)

In the atom transformer the conditioning ``cond`` is the single repr itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.chai.models.primitives import (
    contiguous_segment_mean,
    layer_norm,
    linear,
    linear_bf16,
)

# AdaLN scale offset (scale + ADALN_BIAS); standard AF3 value, verified by the
# token_embedder transition parity gate.
ADALN_BIAS = 1.0
# Non-affine LayerNorm epsilon used inside the transition (explicit in the graph).
TRANSITION_LN_EPS = 0.1


class TransitionParams(NamedTuple):
    lin_s_merged_weight: jnp.ndarray  # (2*d, d) -> scale, shift
    a_double_weight: jnp.ndarray  # (2*hidden, d) -> value, gate
    b_weight: jnp.ndarray  # (d, hidden)
    s_gate_weight: jnp.ndarray  # (d, d)
    s_gate_bias: jnp.ndarray  # (d,)


def map_transition(state: Mapping[str, Any], prefix: str) -> TransitionParams:
    """Extract a conditioned-transition block's params at ``prefix``."""
    return TransitionParams(
        lin_s_merged_weight=jnp.asarray(state[f"{prefix}.ada_ln.lin_s_merged.weight"]),
        a_double_weight=jnp.asarray(state[f"{prefix}.linear_a_nobias_double.weight"]),
        b_weight=jnp.asarray(state[f"{prefix}.linear_b_nobias.weight"]),
        s_gate_weight=jnp.asarray(state[f"{prefix}.linear_s_biasinit_m2.weight"]),
        s_gate_bias=jnp.asarray(state[f"{prefix}.linear_s_biasinit_m2.bias"]),
    )


LinearFn = Callable[..., jnp.ndarray]


def conditioned_transition(
    x: jnp.ndarray,
    cond: jnp.ndarray,
    params: TransitionParams,
    lin: LinearFn = linear,
) -> jnp.ndarray:
    """AdaLN-conditioned SwiGLU transition; returns the residual term (not added).

    ``lin`` selects the precision (``linear`` fp32 or ``linear_bf16``); the
    LayerNorm/silu/sigmoid run in the input dtype to mirror the graph.
    """
    feat = layer_norm(x.astype(jnp.float32), eps=TRANSITION_LN_EPS).astype(x.dtype)
    scale, shift = jnp.split(lin(cond, params.lin_s_merged_weight), 2, axis=-1)
    feat = feat * (scale + ADALN_BIAS) + shift

    a, gate = jnp.split(lin(feat, params.a_double_weight), 2, axis=-1)
    h = jax_silu(a) * gate

    out_gate = jax_sigmoid(lin(cond, params.s_gate_weight, params.s_gate_bias))
    return out_gate * lin(h, params.b_weight)


def jax_silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax_sigmoid(x)


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


# ---------------------------------------------------------------------------
# Full token_embedder forward (atom encoder + local attention + projections).
# All ops recovered verbatim from forward_256.code (te_code.txt). Per-block
# weight binding (traced from SSA names): block i -> transitions.i,
# local_attentions.i, and blocked bias weight[i]. The compute order in the
# graph is block 0,1,2 ascending.
# ---------------------------------------------------------------------------

# eps for the affine LayerNorm inside blocked_pairs2blocked_bias (torch default).
B2B_LN_EPS = 1e-5
# Attention bias fill for masked-out kv positions.
ATTN_MASK_FILL = -10000.0


class AttnParams(NamedTuple):
    lin_s_merged_weight: jnp.ndarray  # (256,128) AdaLN scale/shift
    to_qkv_weight: jnp.ndarray  # (3,4,32,128)
    q_bias: jnp.ndarray  # (4,32)
    out_proj_weight: jnp.ndarray  # (128,128)
    out_proj_bias: jnp.ndarray  # (128,)


class TokenEmbedderParams(NamedTuple):
    to_atom_cond_weight: jnp.ndarray  # (128,128)
    aspp_h_weight: jnp.ndarray  # (16,128)
    aspp_w_weight: jnp.ndarray  # (16,128)
    atom_pair_mlp0_weight: jnp.ndarray  # (16,16)
    atom_pair_mlp2_weight: jnp.ndarray  # (16,16)
    b2b_ln_weight: jnp.ndarray  # (16,)
    b2b_ln_bias: jnp.ndarray  # (16,)
    b2b_bias_weight: jnp.ndarray  # (3,4,16)
    transitions: tuple[TransitionParams, ...]  # 3
    attns: tuple[AttnParams, ...]  # 3
    to_token_single_weight: jnp.ndarray  # (384,128)
    token_single_proj_in_trunk: jnp.ndarray  # (384,768)
    token_single_proj_in_structure: jnp.ndarray  # (384,768)
    token_single_to_token_pair_outer_sum_proj: jnp.ndarray  # (512,384)
    token_pair_proj_in_trunk: jnp.ndarray  # (256,256)


def _att(state: Mapping[str, Any], i: int, attn_prefix: str) -> AttnParams:
    p = f"{attn_prefix}.local_attentions.{i}"
    return AttnParams(
        lin_s_merged_weight=jnp.asarray(
            state[f"{p}.single_layer_norm.lin_s_merged.weight"]
        ),
        to_qkv_weight=jnp.asarray(state[f"{p}.to_qkv.weight"]),
        q_bias=jnp.asarray(state[f"{p}.q_bias"]),
        out_proj_weight=jnp.asarray(state[f"{p}.out_proj.weight"]),
        out_proj_bias=jnp.asarray(state[f"{p}.out_proj.bias"]),
    )


def map_token_embedder(state: Mapping[str, Any]) -> TokenEmbedderParams:
    """Build all token_embedder params from the state dict."""
    enc = "token_single_input_emb.atom_encoder"
    ldt = f"{enc}.atom_transformer.local_diffn_transformer"
    pu = f"{enc}.pair_update_block"
    return TokenEmbedderParams(
        to_atom_cond_weight=jnp.asarray(state[f"{enc}.to_atom_cond.weight"]),
        aspp_h_weight=jnp.asarray(
            state[f"{pu}.atom_single_to_atom_pair_proj_h.1.weight"]
        ),
        aspp_w_weight=jnp.asarray(
            state[f"{pu}.atom_single_to_atom_pair_proj_w.1.weight"]
        ),
        atom_pair_mlp0_weight=jnp.asarray(state[f"{pu}.atom_pair_mlp.0.weight"]),
        atom_pair_mlp2_weight=jnp.asarray(state[f"{pu}.atom_pair_mlp.2.weight"]),
        b2b_ln_weight=jnp.asarray(
            state[f"{ldt}.blocked_pairs2blocked_bias.0.weight"]
        ),
        b2b_ln_bias=jnp.asarray(state[f"{ldt}.blocked_pairs2blocked_bias.0.bias"]),
        b2b_bias_weight=jnp.asarray(
            state[f"{ldt}.blocked_pairs2blocked_bias.1.weight"]
        ),
        transitions=tuple(
            map_transition(state, f"{ldt}.transitions.{i}") for i in range(3)
        ),
        attns=tuple(_att(state, i, ldt) for i in range(3)),
        to_token_single_weight=jnp.asarray(
            state[f"{enc}.to_token_single.0.weight"]
        ),
        token_single_proj_in_trunk=jnp.asarray(
            state["token_single_proj_in_trunk.weight"]
        ),
        token_single_proj_in_structure=jnp.asarray(
            state["token_single_proj_in_structure.weight"]
        ),
        token_single_to_token_pair_outer_sum_proj=jnp.asarray(
            state["token_single_to_token_pair_outer_sum_proj.weight"]
        ),
        token_pair_proj_in_trunk=jnp.asarray(
            state["token_pair_proj_in_trunk.weight"]
        ),
    )


def _pair_update_block(
    h_in: jnp.ndarray,
    w_in: jnp.ndarray,
    pair_feat: jnp.ndarray,
    p: TokenEmbedderParams,
    lin: LinearFn,
) -> jnp.ndarray:
    """aspp_h/w + atom_pair_mlp residual. h_in/w_in: (b,1,nblk,bq|bk,128)."""
    h = lin(jax.nn.relu(h_in), p.aspp_h_weight)  # (b,1,nblk,bq,16)
    w = lin(jax.nn.relu(w_in), p.aspp_w_weight)  # (b,1,nblk,bk,16)
    pair = (
        pair_feat[:, None]
        + h[..., :, None, :]
        + w[..., None, :, :]
    )
    mlp = lin(jax.nn.relu(lin(pair, p.atom_pair_mlp0_weight)), p.atom_pair_mlp2_weight)
    return mlp + pair


def _kv_idx(seq: int, nblk: int, bq: int, bk: int) -> jnp.ndarray:
    q_indices = jnp.arange(seq).reshape(nblk, bq)
    kv_start = q_indices[:, :1] + (bq - bk) // 2  # (nblk,1)
    return (kv_start + jnp.arange(bk)) % seq  # (nblk,bk)


def _local_attention(
    single_cur: jnp.ndarray,
    cond: jnp.ndarray,
    bias: jnp.ndarray,
    kv_idx: jnp.ndarray,
    ap: AttnParams,
    nblk: int,
    lin: LinearFn,
) -> jnp.ndarray:
    """One local-attention block. single_cur/cond: (b,a,128). bias:(b,h,nblk,bq,bk)."""
    b, a, _ = single_cur.shape
    h, d = ap.q_bias.shape
    bq = bias.shape[3]
    bk = bias.shape[4]
    feat = layer_norm(single_cur.astype(jnp.float32), eps=TRANSITION_LN_EPS)
    feat = feat.astype(single_cur.dtype)
    scale, shift = jnp.split(lin(cond, ap.lin_s_merged_weight), 2, axis=-1)
    feat = feat * (scale + ADALN_BIAS) + shift
    # to_qkv: (3,h,d,c) x (b,a,c) -> (3,b,h,a,d)
    qkv = jnp.einsum("nhdc,bac->nbhad", ap.to_qkv_weight.astype(feat.dtype), feat)
    q, k, v = qkv[0], qkv[1], qkv[2]  # each (b,h,a,d)
    q = q + ap.q_bias[None, :, None, :]
    q = q.reshape(b * h, a, d).reshape(b * h, nblk, bq, d)
    k = k.reshape(b * h, a, d)[:, kv_idx]  # (b*h,nblk,bk,d)
    v = v.reshape(b * h, a, d)[:, kv_idx]
    mask = bias.reshape(b * h, nblk, bq, bk)
    out = _sdpa(q.astype(jnp.bfloat16) if lin is linear_bf16 else q, k, v, mask)
    out = out.reshape(b, h, nblk, bq, d)
    out = jnp.transpose(out, (0, 2, 3, 1, 4)).reshape(b, nblk * bq, h * d)
    gate = jax_sigmoid(lin(cond, ap.out_proj_weight, ap.out_proj_bias))
    return out.astype(gate.dtype) * gate


def _sdpa(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """scaled_dot_product_attention with additive mask, fp32-accum softmax."""
    qf = q.astype(jnp.float32)
    kf = k.astype(jnp.float32)
    scale = 1.0 / jnp.sqrt(qf.shape[-1])
    logits = jnp.einsum("...qd,...kd->...qk", qf, kf) * scale + mask.astype(jnp.float32)
    attn = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("...qk,...kd->...qd", attn.astype(v.dtype), v)


def _blocked_bias(
    b2b_out: jnp.ndarray,
    mask: jnp.ndarray,
    weight_i: jnp.ndarray,
    lin: LinearFn,
) -> jnp.ndarray:
    """einsum bias for block i; b2b_out:(b,nblk,bq,bk,16), mask:(b,nblk,bq,bk)."""
    if lin is linear_bf16:
        val = jnp.einsum(
            "blqkc,hc->bhlqk",
            b2b_out.astype(jnp.bfloat16),
            weight_i.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
    else:
        val = jnp.einsum("blqkc,hc->bhlqk", b2b_out, weight_i)
    m = mask[:, None]  # (b,1,nblk,bq,bk)
    return jnp.where(m, val, ATTN_MASK_FILL)


def token_embedder_forward(
    inputs: Mapping[str, jnp.ndarray],
    params: TokenEmbedderParams,
    *,
    bf16: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full forward. Returns (token_single_trunk, token_single_structure,
    token_pair_trunk)."""
    lin = linear_bf16 if bf16 else linear
    p = params
    atom_feat = inputs["atom_single_input_feats"]
    pair_feat = inputs["block_atom_pair_feat"]
    pair_mask = inputs["block_atom_pair_mask"]
    idx_h = inputs["block_indices_h"]
    idx_w = inputs["block_indices_w"]
    atom_mask = inputs["atom_single_mask"]
    atom_token_indices = inputs["atom_token_indices"]
    ts_feats = inputs["token_single_input_feats"]
    tp_feats = inputs["token_pair_input_feats"]

    raw = lin(atom_feat, p.to_atom_cond_weight)  # (b,a,128)
    cond = layer_norm(raw.astype(jnp.float32)).astype(raw.dtype)  # input1 base
    single_repr = raw  # un-normed
    b, a, c = raw.shape
    nblk, bq = idx_h.shape
    bk = idx_w.shape[1]

    # gather cond by block indices -> (b,1,nblk,bq|bk,128)
    cond_g = cond[:, None]  # (b,1,a,128)
    h_in = cond_g[:, :, idx_h]  # (b,1,nblk,bq,128)
    w_in = cond_g[:, :, idx_w]  # (b,1,nblk,bk,128)
    pu = _pair_update_block(h_in, w_in, pair_feat, p, lin)
    b2b_in = pu.reshape(b, nblk, bq, bk, 16)

    cond1 = cond  # (b,a,128) input1
    smask = atom_mask  # (b,a)
    single_repr = jnp.where(smask[..., None], single_repr, 0.0)

    b2b = layer_norm(
        b2b_in.astype(jnp.float32), p.b2b_ln_weight, p.b2b_ln_bias, eps=B2B_LN_EPS
    ).astype(b2b_in.dtype)
    kv_idx = _kv_idx(a, nblk, bq, bk)

    for i in range(3):
        bias = _blocked_bias(b2b, pair_mask, p.b2b_bias_weight[i], lin)
        feat_local = _local_attention(
            single_repr, cond1, bias, kv_idx, p.attns[i], nblk, lin
        )
        feat_trans = conditioned_transition(single_repr, cond1, p.transitions[i], lin)
        single_repr = single_repr + feat_trans + feat_local
        single_repr = jnp.where(smask[..., None], single_repr, 0.0)

    atom_single = jax.nn.relu(lin(single_repr, p.to_token_single_weight))  # (b,a,384)

    # atom -> token mean pool
    ntok = ts_feats.shape[1]
    masked = atom_single * atom_mask[..., None].astype(atom_single.dtype)
    seg = atom_token_indices.astype(jnp.int32)
    pooled = contiguous_segment_mean(
        masked,
        atom_mask,
        seg,
        num_segments=ntok,
    )

    input13 = jnp.concatenate([pooled, ts_feats], axis=-1)  # (b,ntok,768)
    s_struct = lin(input13, p.token_single_proj_in_structure)
    s_trunk = lin(input13, p.token_single_proj_in_trunk)
    outer = lin(s_trunk, p.token_single_to_token_pair_outer_sum_proj)  # (b,ntok,512)
    x37, x38 = jnp.split(outer, 2, axis=-1)  # each (b,ntok,256)
    tp = tp_feats + x37[:, :, None, :] + x38[:, None, :, :]
    pair_trunk = lin(tp, p.token_pair_proj_in_trunk)
    return s_trunk, s_struct, pair_trunk
