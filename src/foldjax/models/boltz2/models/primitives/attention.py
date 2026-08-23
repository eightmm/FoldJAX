"""Pure JAX attention blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.primitives.attention_backend import (
    masked_softmax,
    tokamax_dot_product_attention,
)

AttentionPairBiasParams = Mapping[str, Mapping[str, jnp.ndarray]]


def attention_pair_bias_forward(
    params: AttentionPairBiasParams,
    s: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    k_in: jnp.ndarray | None = None,
    multiplicity: int = 1,
    inf: float = 1e6,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    attention_backend: str = "xla",
) -> jnp.ndarray:
    """Run Boltz AttentionPairBias v2 with mapped PyTorch parameters.

    ``chunk_size`` enables query-axis (i) blocking: scores/softmax/@v are
    computed one query block at a time and concatenated over the query axis.
    The softmax denominator is over the FULL key axis within each query row, so
    this is bit-exact; only independent query rows are blocked. ``None`` (default)
    or ``N <= chunk_size`` falls back to the single-shot path.
    """

    if k_in is None:
        k_in = s
    batch, _, c_s = s.shape
    num_heads = params["proj_z"]["kernel"].shape[-1]
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

    bias = _layer_norm(
        z,
        params["proj_z_norm"]["scale"],
        params["proj_z_norm"]["bias"],
        eps,
    )
    bias = _linear(bias, params["proj_z"]["kernel"])
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
        q_f = q.astype(jnp.float32)
        k_f = k.astype(jnp.float32)
        v_f = v.astype(jnp.float32)
        bias_f = bias.astype(jnp.float32)
        scale = jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
        key_mask = mask[:, None, None, :].astype(bool)
        n_q = q.shape[1]
        if chunk_size is None or chunk_size <= 0 or n_q <= chunk_size:
            out = _attention_qblock(q_f, k_f, v_f, bias_f, key_mask, scale)
        else:
            blocks = []
            for start in range(0, n_q, chunk_size):
                stop = min(start + chunk_size, n_q)
                blocks.append(
                    _attention_qblock(
                        q_f[:, start:stop],
                        k_f,
                        v_f,
                        bias_f[:, :, start:stop],
                        key_mask,
                        scale,
                    )
                )
            out = jnp.concatenate(blocks, axis=1)
    else:
        msg = f"Unsupported attention backend: {attention_backend!r}"
        raise ValueError(msg)

    out = out.astype(v.dtype).reshape(batch, -1, c_s)
    return _linear(g * out, params["proj_o"]["kernel"])


def _attention_qblock(
    q_blk: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias_blk: jnp.ndarray,
    key_mask: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """Exact attention for a query block. q_blk:[b,iq,h,d]; softmax over full j."""
    attn = jnp.einsum("bihd,bjhd->bhij", q_blk, k) / scale
    attn = attn + bias_blk
    attn = masked_softmax(attn, key_mask)
    return jnp.einsum("bhij,bjhd->bihd", attn, v)
