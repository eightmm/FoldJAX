"""Pure JAX diffusion transformer blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.primitives.attention_backend import (
    tokamax_dot_product_attention,
)

Params = Mapping[str, object]


def adaln_s_terms(
    params: Params,
    s: jnp.ndarray,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute the s-derived AdaLN scale (sigmoid) and bias.

    These depend only on the (layer-constant) single conditioning ``s`` and so
    can be hoisted out of the per-layer / per-step loop.
    """

    s_n = _layer_norm_scale(s, params["s_norm"]["scale"], eps)
    scale = jax.nn.sigmoid(
        _linear(s_n, params["s_scale"]["kernel"], params["s_scale"]["bias"])
    )
    bias = _linear(s_n, params["s_bias"]["kernel"])
    return scale, bias


def adaln_apply(
    a: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply precomputed AdaLN scale/bias to the per-layer activations ``a``."""

    return scale * _layer_norm_no_affine(a, eps) + bias


def _gate_s(params: Params, s: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid output gate derived from the layer-constant ``s``."""

    return jax.nn.sigmoid(_linear(s, params["kernel"], params["bias"]))


def conditioned_transition_block_forward(
    params: Params,
    a: jnp.ndarray,
    s: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz ConditionedTransitionBlock with mapped PyTorch parameters."""

    adaln_scale, adaln_bias = adaln_s_terms(params["adaln"], s, eps)
    gate = _gate_s(params["output_projection"], s)
    return _conditioned_transition_block_apply(
        params, a, adaln_scale, adaln_bias, gate, eps
    )


def _conditioned_transition_block_apply(
    params: Params,
    a: jnp.ndarray,
    adaln_scale: jnp.ndarray,
    adaln_bias: jnp.ndarray,
    gate: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    a = adaln_apply(a, adaln_scale, adaln_bias, eps)
    swish_x, swish_gate = jnp.split(_linear(a, params["swish_gate"]["kernel"]), 2, -1)
    b = (jax.nn.silu(swish_gate) * swish_x) * _linear(a, params["a_to_b"]["kernel"])
    out = _linear(b, params["b_to_a"]["kernel"])
    return gate * out


def diffusion_transformer_layer_forward(
    params: Params,
    a: jnp.ndarray,
    s: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    multiplicity: int = 1,
    eps: float = 1e-5,
    inf: float = 1e6,
    attention_backend: str = "xla",
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run one Boltz DiffusionTransformerLayer with pair-bias attention."""

    s_terms = layer_s_terms(params, s, eps)
    return diffusion_transformer_layer_apply(
        params,
        a,
        s_terms,
        bias,
        mask,
        to_keys=to_keys,
        multiplicity=multiplicity,
        eps=eps,
        inf=inf,
        attention_backend=attention_backend,
        chunk_size=chunk_size,
    )


def layer_s_terms(
    params: Params,
    s: jnp.ndarray,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, ...]:
    """Precompute all s-derived (layer-constant) terms for one layer.

    Returns the attention AdaLN scale/bias, the attention output gate, the
    transition AdaLN scale/bias, and the transition output gate. All depend
    only on ``s``, which is constant across layers and diffusion steps.
    """

    attn_scale, attn_bias = adaln_s_terms(params["adaln"], s, eps)
    attn_gate = _gate_s(params["output_projection"], s)
    trans = params["transition"]
    trans_scale, trans_bias = adaln_s_terms(trans["adaln"], s, eps)
    trans_gate = _gate_s(trans["output_projection"], s)
    return attn_scale, attn_bias, attn_gate, trans_scale, trans_bias, trans_gate


def diffusion_transformer_layer_apply(
    params: Params,
    a: jnp.ndarray,
    s_terms: tuple[jnp.ndarray, ...],
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    multiplicity: int = 1,
    eps: float = 1e-5,
    inf: float = 1e6,
    attention_backend: str = "xla",
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run one layer using precomputed (hoisted) s-derived terms."""

    attn_scale, attn_bias, attn_gate, trans_scale, trans_bias, trans_gate = s_terms

    b = adaln_apply(a, attn_scale, attn_bias, eps)
    k_in = b
    if to_keys is not None:
        k_in = to_keys(b)
        mask = jnp.squeeze(to_keys(mask[..., None]), axis=-1)

    b = _attention_pair_bias_no_proj_z_forward(
        params["pair_bias_attn"],
        s=b,
        bias=bias,
        mask=mask,
        k_in=k_in,
        multiplicity=multiplicity,
        inf=inf,
        attention_backend=attention_backend,
        chunk_size=chunk_size,
    )
    b = attn_gate * b

    a = a + b
    a = a + _conditioned_transition_block_apply(
        params["transition"], a, trans_scale, trans_bias, trans_gate, eps
    )
    if "post_lnorm" in params:
        post_lnorm = params["post_lnorm"]
        a = _layer_norm(a, post_lnorm["scale"], post_lnorm["bias"], eps)
    return a


def _attention_pair_bias_no_proj_z_forward(
    params: Params,
    s: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    k_in: jnp.ndarray,
    multiplicity: int,
    inf: float,
    attention_backend: str = "xla",
    chunk_size: int | None = None,
) -> jnp.ndarray:
    batch, _, c_s = s.shape
    num_heads = bias.shape[-1]
    head_dim = c_s // num_heads

    qg = _linear(
        s,
        jnp.concatenate(
            (params["proj_q"]["kernel"], params["proj_g"]["kernel"]), axis=-1
        ),
    )
    q, g_logits = jnp.split(qg, (params["proj_q"]["kernel"].shape[-1],), axis=-1)
    q = (q + params["proj_q"]["bias"]).reshape(batch, -1, num_heads, head_dim)
    g = jax.nn.sigmoid(g_logits)
    kv = _linear(
        k_in,
        jnp.concatenate(
            (params["proj_k"]["kernel"], params["proj_v"]["kernel"]), axis=-1
        ),
    )
    k, v = jnp.split(kv, 2, axis=-1)
    k = k.reshape(batch, -1, num_heads, head_dim)
    v = v.reshape(batch, -1, num_heads, head_dim)

    bias = jnp.transpose(bias, (0, 3, 1, 2))
    bias = jnp.repeat(bias, multiplicity, axis=0)
    if attention_backend in ("tokamax", "flash"):
        out = tokamax_dot_product_attention(
            q,
            k,
            v,
            bias,
            mask,
            scale=float(head_dim) ** -0.5,
            backend=attention_backend,
        )
    elif attention_backend == "xla":
        # Score dtype: keep the DEFAULT fp32 path bit-exact (compute_dtype is
        # fp32 -> these casts are no-ops). Under an opted-in low-precision
        # compute dtype, run the scores/value-contraction matmuls in that dtype
        # so the [b, heads, N, N] score buffer (the N^2 token-attention OOM
        # blocker) actually shrinks; the softmax denominator still reduces in
        # fp32 (precision-sensitive island).
        score_dtype = q.dtype
        q_s = q.astype(score_dtype)
        k_s = k.astype(score_dtype)
        v_s = v.astype(score_dtype)
        bias_s = bias.astype(score_dtype)
        scale = jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
        mask_bias = (1.0 - mask[:, None, None].astype(jnp.float32)) * -inf
        n_q = q_s.shape[1]
        if chunk_size is None or chunk_size <= 0 or n_q <= chunk_size:
            out = _no_proj_qblock(q_s, k_s, v_s, bias_s, mask_bias, scale)
        else:
            blocks = []
            for start in range(0, n_q, chunk_size):
                stop = min(start + chunk_size, n_q)
                blocks.append(
                    _no_proj_qblock(
                        q_s[:, start:stop],
                        k_s,
                        v_s,
                        bias_s[:, :, start:stop],
                        mask_bias,
                        scale,
                    )
                )
            out = jnp.concatenate(blocks, axis=1)
        out = out.astype(v.dtype)
    else:
        msg = f"Unsupported attention backend: {attention_backend!r}"
        raise ValueError(msg)
    out = out.reshape(batch, -1, c_s)
    return _linear(g * out, params["proj_o"]["kernel"])


def _no_proj_qblock(
    q_blk: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias_blk: jnp.ndarray,
    mask_bias: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """Exact pair-bias attention for a query block.

    The scores einsum runs in the inputs' dtype (fp32 by default, bf16/fp16 when
    opted in). The softmax (subtract-max + exp + sum) reduces in fp32, then casts
    the probabilities back to the value dtype for the @v contraction. With fp32
    inputs every cast is a no-op, so the default path is bit-exact with the
    previous single-shot fp32 implementation.
    """

    in_dtype = q_blk.dtype
    attn = jnp.einsum("bihd,bjhd->bhij", q_blk, k)
    attn = attn.astype(jnp.float32) / scale
    attn = attn + bias_blk.astype(jnp.float32) + mask_bias
    attn = jax.nn.softmax(attn, axis=-1).astype(in_dtype)
    return jnp.einsum("bhij,bjhd->bihd", attn, v)


def _layer_norm_scale(x: jnp.ndarray, scale: jnp.ndarray, eps: float) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * scale


def _layer_norm_no_affine(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps)
