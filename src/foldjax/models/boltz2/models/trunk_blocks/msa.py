"""Pure JAX MSA module for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.primitives._scan_utils import stack_layer_params
from foldjax.models.boltz2.models.primitives.transition import transition_forward
from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    resolve_triangle_attention_chunk,
    resolve_triangle_attention_q_chunk,
    triangle_attention_forward,
)

Params = Mapping[str, object]


def msa_module_forward(
    params: Params,
    z: jnp.ndarray,
    emb: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    num_tokens: int = 33,
    eps: float = 1e-5,
    use_scan: bool = True,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
    subsample_msa: bool = False,
    num_subsampled_msa: int = 1024,
) -> jnp.ndarray:
    """Run Boltz MSAModule.

    ``use_scan=False`` (default) unrolls the layer stack in Python (lower steady
    latency). ``use_scan=True`` runs the stack via ``lax.scan`` (faster compile).

    ``subsample_msa`` caps the MSA depth at ``num_subsampled_msa`` (default 1024)
    by taking the first rows, matching the Boltz CLI's inference-time MSA
    subsampling (Boltz draws a random 1024 subset per call; AF3 takes the first
    1024 deterministically — we follow AF3's deterministic truncation, which is
    reproducible and keeps the highest-ranked rows). Cuts MSA-module cost ~Nx for
    deep MSAs.
    """

    # Equivalent to one_hot(msa, num_tokens) @ kernel[:num_tokens] but without
    # materializing the [batch, num_seq, N, num_tokens] one-hot tensor.
    # Concatenation order in the original code is: one-hot block (rows
    # [0:num_tokens]), then has_deletion, deletion_value, msa_paired
    # (rows [num_tokens:num_tokens+3]). Split the kernel at call time; the
    # stored param pytree is left unchanged.
    kernel = params["msa_proj"]["kernel"]
    kernel_onehot = kernel[:num_tokens]  # [num_tokens, d]
    kernel_extra = kernel[num_tokens:]  # [3, d]
    # Subsample MSA depth (axis 1) to num_subsampled_msa, matching Boltz CLI's
    # inference-time cap; take the first rows (deterministic, query-first order).
    msa_d = feats["msa"]
    has_del = feats["has_deletion"]
    del_val = feats["deletion_value"]
    msa_paired = feats["msa_paired"]
    msa_mask = feats["msa_mask"]
    if subsample_msa and msa_d.shape[1] > num_subsampled_msa:
        sl = slice(0, num_subsampled_msa)
        msa_d = msa_d[:, sl]
        has_del = has_del[:, sl]
        del_val = del_val[:, sl]
        msa_paired = msa_paired[:, sl]
        msa_mask = msa_mask[:, sl]
    msa_idx = msa_d.astype(jnp.int32)
    extra = jnp.stack((has_del, del_val, msa_paired), axis=-1).astype(
        kernel_extra.dtype
    )
    m = kernel_onehot[msa_idx] + _linear(extra, kernel_extra)
    m = m + _linear(emb, params["s_proj"]["kernel"])[:, None]

    token_mask = feats["token_pad_mask"].astype(m.dtype)
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    msa_mask = msa_mask.astype(m.dtype)

    layers = params["layers"]
    if not use_scan:
        for layer in layers:
            z, m = msa_layer_forward(
                layer,
                z,
                m,
                token_mask,
                msa_mask,
                eps=eps,
                chunk_size=chunk_size,
                triangle_attention_chunk=triangle_attention_chunk,
                triangle_attention_q_chunk=triangle_attention_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                matmul_precision=matmul_precision,
                triangle_backend=triangle_backend,
                glu_backend=glu_backend,
            )
        return z

    stacked = stack_layer_params(layers)

    def body(carry, layer):
        z_c, m_c = carry
        z_c, m_c = msa_layer_forward(
            layer,
            z_c,
            m_c,
            token_mask,
            msa_mask,
            eps=eps,
            chunk_size=chunk_size,
            triangle_attention_chunk=triangle_attention_chunk,
            triangle_attention_q_chunk=triangle_attention_q_chunk,
            transition_hidden_chunk=transition_hidden_chunk,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
            glu_backend=glu_backend,
        )
        return (z_c, m_c), None

    (z, m), _ = jax.lax.scan(body, (z, m), stacked)
    return z


def msa_layer_forward(
    params: Params,
    z: jnp.ndarray,
    m: jnp.ndarray,
    token_mask: jnp.ndarray,
    msa_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one Boltz MSALayer in eval mode."""

    m = m + pair_weighted_averaging_forward(
        params["pair_weighted_averaging"], m, z, token_mask, eps=eps
    )
    m = m + transition_forward(
        params["msa_transition"], m, eps=eps, glu_backend=glu_backend
    )
    z = z + outer_product_mean_forward(
        params["outer_product_mean"], m, msa_mask, eps, chunk_size=chunk_size
    )
    z = pairformer_no_seq_layer_forward(
        params["pairformer_layer"],
        z,
        token_mask,
        eps,
        chunk_size=chunk_size,
        triangle_attention_chunk=triangle_attention_chunk,
        triangle_attention_q_chunk=triangle_attention_q_chunk,
        transition_hidden_chunk=transition_hidden_chunk,
        matmul_precision=matmul_precision,
        triangle_backend=triangle_backend,
        glu_backend=glu_backend,
    )
    return z, m


def pair_weighted_averaging_forward(
    params: Params,
    m: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-5,
    inf: float = 1e6,
) -> jnp.ndarray:
    """Run Boltz PairWeightedAveraging."""

    m = _layer_norm(m, params["norm_m"]["scale"], params["norm_m"]["bias"], eps)
    z = _layer_norm(z, params["norm_z"]["scale"], params["norm_z"]["bias"], eps)
    num_heads = params["proj_z"]["kernel"].shape[-1]
    c_h = params["proj_m"]["kernel"].shape[-1] // num_heads

    v = _linear(m, params["proj_m"]["kernel"])
    v = jnp.reshape(v, (*v.shape[:3], num_heads, c_h))
    v = jnp.transpose(v, (0, 3, 1, 2, 4))
    b = _linear(z, params["proj_z"]["kernel"])
    b = jnp.transpose(b, (0, 3, 1, 2))
    # Mask + softmax in fp32 so the `inf` constant stays finite (in fp16 it would
    # saturate to inf -> fully-masked rows give NaN). fp32 runtime is unchanged.
    mask = mask.astype(jnp.float32)
    b = b.astype(jnp.float32) + (1.0 - mask[:, None]) * (-inf)
    w = jax.nn.softmax(b, axis=-1).astype(v.dtype)
    g = jax.nn.sigmoid(_linear(m, params["proj_g"]["kernel"]))
    o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
    o = jnp.transpose(o, (0, 2, 3, 1, 4))
    o = jnp.reshape(o, (*o.shape[:3], num_heads * c_h))
    return _linear(g * o, params["proj_o"]["kernel"])


def outer_product_mean_forward(
    params: Params,
    m: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
) -> jnp.ndarray:
    """Run Boltz OuterProductMean.

    Computes the result in chunks over the i (token) axis so the full
    [b, i, j, c, d] fp32 intermediate is never materialized at once. Peak
    intermediate goes from [N, N, c*d] to [chunk_size, N, c*d].
    """

    mask = mask[..., None].astype(m.dtype)
    m = _layer_norm(m, params["norm"]["scale"], params["norm"]["bias"], eps)
    a = _linear(m, params["proj_a"]["kernel"]) * mask
    b = _linear(m, params["proj_b"]["kernel"]) * mask
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    pair_mask = mask[:, :, None, :] * mask[:, :, :, None]
    num_mask = jnp.maximum(jnp.sum(pair_mask, axis=1), 1.0)

    proj_o = params["proj_o"]
    n = a.shape[2]
    out_dtype = m.dtype
    out = jnp.zeros((a.shape[0], n, n, proj_o["kernel"].shape[-1]), dtype=out_dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        a_blk = a[:, :, start:end]
        z = jnp.einsum("bsic,bsjd->bijcd", a_blk, b)
        z = jnp.reshape(z, (*z.shape[:3], -1)) / num_mask[:, start:end]
        out = out.at[:, start:end].set(
            _linear(z.astype(out_dtype), proj_o["kernel"], proj_o["bias"])
        )
    return out


def pairformer_no_seq_layer_forward(
    params: Params,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
) -> jnp.ndarray:
    """Run Boltz PairformerNoSeqLayer in eval mode."""

    tri_att_chunk = resolve_triangle_attention_chunk(
        z.shape[1], chunk_size, triangle_attention_chunk
    )
    tri_att_q_chunk = resolve_triangle_attention_q_chunk(
        z.shape[1], triangle_attention_q_chunk
    )
    z = z + triangle_multiplication_forward(
        params["tri_mul_out"],
        z,
        pair_mask,
        "outgoing",
        eps=eps,
        chunk_size=chunk_size,
        glu_backend=glu_backend,
    )
    z = z + triangle_multiplication_forward(
        params["tri_mul_in"],
        z,
        pair_mask,
        "incoming",
        eps=eps,
        chunk_size=chunk_size,
        glu_backend=glu_backend,
    )
    z = z + triangle_attention_forward(
        params["tri_att_start"],
        z,
        pair_mask,
        starting=True,
        eps=eps,
        chunk_size=tri_att_chunk,
        q_chunk_size=tri_att_q_chunk,
        matmul_precision=matmul_precision,
        triangle_backend=triangle_backend,
    )
    z = z + triangle_attention_forward(
        params["tri_att_end"],
        z,
        pair_mask,
        starting=False,
        eps=eps,
        chunk_size=tri_att_chunk,
        q_chunk_size=tri_att_q_chunk,
        matmul_precision=matmul_precision,
        triangle_backend=triangle_backend,
    )
    z = z + transition_forward(
        params["transition_z"],
        z,
        chunk_size=transition_hidden_chunk,
        eps=eps,
        row_chunk_size=chunk_size,
        glu_backend=glu_backend,
    )
    return z
