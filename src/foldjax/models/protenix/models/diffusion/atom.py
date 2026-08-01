"""Atom-level layout and cache helpers for the Protenix JAX port."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.diffusion.transformer import (
    DiffusionTransformerStackParams,
    diffusion_transformer_stack,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)


class AtomAttentionEncoderCacheParams(NamedTuple):
    """Parameters for AtomAttentionEncoder cache preparation."""

    linear_ref_pos: LinearParams
    linear_ref_charge: LinearParams
    linear_f: LinearParams
    linear_d: LinearParams
    linear_invd: LinearParams
    linear_v: LinearParams


class AtomPairMlpParams(NamedTuple):
    """Parameters for AtomAttentionEncoder's small atom-pair MLP."""

    linear_1: LinearParams
    linear_2: LinearParams
    linear_3: LinearParams


class AtomAttentionEncoderParams(NamedTuple):
    """Parameters for Protenix ``AtomAttentionEncoder``."""

    cache: AtomAttentionEncoderCacheParams
    linear_cl: LinearParams
    linear_cm: LinearParams
    small_mlp: AtomPairMlpParams
    atom_transformer: DiffusionTransformerStackParams
    linear_q: LinearParams
    layernorm_s: LayerNormParams | None = None
    linear_s: LinearParams | None = None
    layernorm_z: LayerNormParams | None = None
    linear_z: LinearParams | None = None
    linear_r: LinearParams | None = None


class AtomAttentionDecoderParams(NamedTuple):
    """Parameters for Protenix ``AtomAttentionDecoder``."""

    linear_a: LinearParams
    layernorm_q: LayerNormParams
    linear_out: LinearParams
    atom_transformer: DiffusionTransformerStackParams


def rearrange_qk_to_dense_trunk(
    q: jnp.ndarray,
    k: jnp.ndarray,
    *,
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray | int]]:
    """Rearrange atom query/key arrays into local dense trunks.

    This JAX version supports the common Protenix atom layout where the trunked
    dimension is the leading axis of ``q``/``k``.
    """

    if n_keys < n_queries:
        raise ValueError("n_keys must be >= n_queries")
    if n_queries % 2 or n_keys % 2:
        raise ValueError("n_queries and n_keys must be even")
    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k must have matching atom dimensions")

    n = q.shape[0]
    n_trunks = int(math.ceil(n / n_queries))
    q_pad = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 0.5) * n_queries + n_keys / 2 - n + 0.5)

    q_padded = _pad_leading_dim(q, 0, q_pad)
    k_padded = _pad_leading_dim(k, pad_left, pad_right)
    q_trunked = q_padded.reshape((n_trunks, n_queries) + q.shape[1:])
    k_trunked = jnp.stack(
        [
            k_padded[i * n_queries : i * n_queries + n_keys]
            for i in range(n_trunks)
        ],
        axis=0,
    )

    mask_trunked = None
    if compute_mask:
        q_abs = jnp.arange(n_trunks * n_queries).reshape(n_trunks, n_queries)
        k_abs = (
            jnp.arange(n_keys)[None, :]
            + jnp.arange(n_trunks)[:, None] * n_queries
            - pad_left
        )
        mask_trunked = (q_abs[..., None] < n) & (k_abs[:, None, :] >= 0) & (
            k_abs[:, None, :] < n
        )

    return q_trunked, k_trunked, {
        "mask_trunked": mask_trunked,
        "q_pad": q_pad,
        "k_pad_left": pad_left,
        "k_pad_right": pad_right,
    }


def broadcast_token_to_atom(
    x_token: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Broadcast token embeddings to atom embeddings."""

    return jnp.take(x_token, atom_to_token_idx, axis=-2)


def aggregate_atom_to_token(
    x_atom: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    *,
    n_token: int,
    reduce: str = "mean",
) -> jnp.ndarray:
    """Aggregate atom embeddings to token embeddings."""

    if reduce not in {"mean", "sum"}:
        raise ValueError("reduce must be 'mean' or 'sum'")
    out_shape = x_atom.shape[:-2] + (n_token, x_atom.shape[-1])
    out = jnp.zeros(out_shape, dtype=x_atom.dtype)
    out = out.at[..., atom_to_token_idx, :].add(x_atom)
    if reduce == "sum":
        return out
    counts = jnp.zeros((n_token,), dtype=x_atom.dtype)
    counts = counts.at[atom_to_token_idx].add(1.0)
    return out / jnp.maximum(counts[..., None], 1.0)


def broadcast_token_to_local_atom_pair(
    z_token: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    *,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = True,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray | int]]:
    """Broadcast token pair embeddings to local atom-pair trunks."""

    idx_q, idx_k, pad_info = rearrange_qk_to_dense_trunk(
        atom_to_token_idx,
        atom_to_token_idx,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )
    return gather_pair_embedding_in_dense_trunk(z_token, idx_q, idx_k), pad_info


def gather_pair_embedding_in_dense_trunk(
    z_token: jnp.ndarray,
    idx_q: jnp.ndarray,
    idx_k: jnp.ndarray,
) -> jnp.ndarray:
    """Gather ``z_token[..., idx_q, idx_k, :]`` into local dense trunks."""

    idx_q_expanded = idx_q[..., :, None]
    idx_k_expanded = idx_k[..., None, :]
    return z_token[..., idx_q_expanded, idx_k_expanded, :]


def atom_attention_encoder_prepare_cache(
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_mask: jnp.ndarray,
    ref_element: jnp.ndarray,
    ref_atom_name_chars: jnp.ndarray,
    d_lm: jnp.ndarray,
    v_lm: jnp.ndarray,
    pad_info: dict[str, jnp.ndarray],
    params: AtomAttentionEncoderCacheParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Prepare AtomAttentionEncoder pair and single cache without trunk inputs."""

    batch_shape = ref_pos.shape[:-2]
    n_atom = ref_pos.shape[-2]
    charge = jnp.arcsinh(ref_charge).reshape(batch_shape + (n_atom, 1))
    atom_features = jnp.concatenate(
        [
            ref_mask.reshape(batch_shape + (n_atom, 1)),
            ref_element.reshape(batch_shape + (n_atom, 128)),
            ref_atom_name_chars.reshape(batch_shape + (n_atom, 4 * 64)),
        ],
        axis=-1,
    )
    c_l = (
        linear(ref_pos, params.linear_ref_pos)
        + linear(charge, params.linear_ref_charge)
        + linear(atom_features, params.linear_f)
    )
    c_l = c_l * ref_mask.reshape(batch_shape + (n_atom, 1))

    mask = pad_info["mask_trunked"][..., None].astype(d_lm.dtype)
    p_lm = linear(d_lm, params.linear_d) * v_lm * mask
    inv_d = 1.0 / (1.0 + jnp.sum(jnp.square(d_lm), axis=-1, keepdims=True))
    p_lm = p_lm + linear(inv_d, params.linear_invd) * v_lm
    p_lm = p_lm + linear(v_lm.astype(p_lm.dtype), params.linear_v)
    return p_lm, c_l


def atom_attention_encoder_prepare_diffusion_cache(
    atom_to_token_idx: jnp.ndarray,
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_mask: jnp.ndarray,
    ref_element: jnp.ndarray,
    ref_atom_name_chars: jnp.ndarray,
    d_lm: jnp.ndarray,
    v_lm: jnp.ndarray,
    pad_info: dict[str, jnp.ndarray],
    z: jnp.ndarray,
    params: AtomAttentionEncoderParams,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Prepare the reusable coordinate-conditioned atom encoder cache."""

    _require_has_coords_params(params)
    p_lm, c_l = atom_attention_encoder_prepare_cache(
        ref_pos,
        ref_charge,
        ref_mask,
        ref_element,
        ref_atom_name_chars,
        d_lm,
        v_lm,
        pad_info,
        params.cache,
    )
    z_local = broadcast_token_to_local_atom_pair(
        linear(layer_norm(z, params.layernorm_z), params.linear_z),
        atom_to_token_idx,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=False,
    )[0]
    return jnp.expand_dims(p_lm, axis=-5) + z_local, c_l


def atom_pair_conditioning(
    p_lm: jnp.ndarray,
    c_l: jnp.ndarray,
    linear_cl: LinearParams,
    linear_cm: LinearParams,
) -> jnp.ndarray:
    """Add combined atom single conditioning to local atom-pair features."""

    c_l_q, c_l_k = _rearrange_qk_to_dense_trunk_atom_axis(
        c_l,
        c_l,
        n_queries=p_lm.shape[-3],
        n_keys=p_lm.shape[-2],
    )
    return p_lm + linear(jax.nn.relu(c_l_q[..., None, :]), linear_cl) + linear(
        jax.nn.relu(c_l_k[..., None, :, :]),
        linear_cm,
    )


def atom_pair_small_mlp(
    p_lm: jnp.ndarray,
    params: AtomPairMlpParams,
) -> jnp.ndarray:
    """Apply the Protenix AtomAttentionEncoder small MLP."""

    p_lm = linear(jax.nn.relu(p_lm), params.linear_1)
    p_lm = linear(jax.nn.relu(p_lm), params.linear_2)
    return linear(jax.nn.relu(p_lm), params.linear_3)


def atom_attention_encoder(
    atom_to_token_idx: jnp.ndarray,
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_mask: jnp.ndarray,
    ref_atom_name_chars: jnp.ndarray,
    ref_element: jnp.ndarray,
    d_lm: jnp.ndarray,
    v_lm: jnp.ndarray,
    pad_info: dict[str, jnp.ndarray],
    params: AtomAttentionEncoderParams,
    *,
    r_l: jnp.ndarray | None = None,
    s: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    p_lm: jnp.ndarray | None = None,
    c_l: jnp.ndarray | None = None,
    n_token: int,
    n_heads: int,
    n_queries: int,
    n_keys: int,
    use_scan: bool = False,
    attention_backend: str = "xla",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run Protenix AtomAttentionEncoder in input or diffusion mode."""

    if r_l is not None:
        _require_has_coords_params(params)
        if s is None or z is None:
            raise ValueError("r_l requires trunk single s and pair z")

    if p_lm is None or c_l is None:
        p_lm, c_l = atom_attention_encoder_prepare_cache(
            ref_pos,
            ref_charge,
            ref_mask,
            ref_element,
            ref_atom_name_chars,
            d_lm,
            v_lm,
            pad_info,
            params.cache,
        )
        if r_l is not None:
            z_local = broadcast_token_to_local_atom_pair(
                linear(layer_norm(z, params.layernorm_z), params.linear_z),
                atom_to_token_idx,
                n_queries=n_queries,
                n_keys=n_keys,
                compute_mask=False,
            )[0]
            p_lm = jnp.expand_dims(p_lm, axis=-5) + z_local

    if r_l is not None:
        c_l = jnp.expand_dims(c_l, axis=-3) + broadcast_token_to_atom(
            linear(layer_norm(s, params.layernorm_s), params.linear_s),
            atom_to_token_idx,
        )
        q_l = c_l + linear(r_l, params.linear_r)
    else:
        q_l = c_l

    p_lm = atom_pair_conditioning(p_lm, c_l, params.linear_cl, params.linear_cm)
    p_lm = p_lm + atom_pair_small_mlp(p_lm, params.small_mlp)
    q_l = diffusion_transformer_stack(
        q_l,
        c_l,
        p_lm,
        params.atom_transformer,
        num_heads=n_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_scan,
        attention_backend=attention_backend,
    )
    a = aggregate_atom_to_token(
        jax.nn.relu(linear(q_l, params.linear_q)),
        atom_to_token_idx,
        n_token=n_token,
        reduce="mean",
    )
    return a, q_l, c_l, p_lm


def atom_attention_decoder(
    atom_to_token_idx: jnp.ndarray,
    a: jnp.ndarray,
    q_skip: jnp.ndarray,
    c_skip: jnp.ndarray,
    p_skip: jnp.ndarray,
    params: AtomAttentionDecoderParams,
    *,
    n_heads: int,
    n_queries: int,
    n_keys: int,
    use_scan: bool = False,
    attention_backend: str = "xla",
) -> jnp.ndarray:
    """Run Protenix ``AtomAttentionDecoder`` in inference mode."""

    q = broadcast_token_to_atom(linear(a, params.linear_a), atom_to_token_idx)
    q = q + q_skip
    q = diffusion_transformer_stack(
        q,
        c_skip,
        p_skip,
        params.atom_transformer,
        num_heads=n_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_scan,
        attention_backend=attention_backend,
    )
    return linear(layer_norm(q, params.layernorm_q), params.linear_out)


def _require_has_coords_params(params: AtomAttentionEncoderParams) -> None:
    if (
        params.layernorm_s is None
        or params.linear_s is None
        or params.layernorm_z is None
        or params.linear_z is None
        or params.linear_r is None
    ):
        raise ValueError("has_coords=True path requires s/z/r projection params")


def _rearrange_qk_to_dense_trunk_atom_axis(
    q: jnp.ndarray,
    k: jnp.ndarray,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if n_keys < n_queries:
        raise ValueError("n_keys must be >= n_queries")
    if n_queries % 2 or n_keys % 2:
        raise ValueError("n_queries and n_keys must be even")
    if q.shape[-2] != k.shape[-2]:
        raise ValueError("q and k must have matching atom dimensions")

    n = q.shape[-2]
    n_trunks = int(math.ceil(n / n_queries))
    q_pad = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 0.5) * n_queries + n_keys / 2 - n + 0.5)

    pad_q = ((0, 0),) * (q.ndim - 2) + ((0, q_pad), (0, 0))
    pad_k = ((0, 0),) * (k.ndim - 2) + ((pad_left, pad_right), (0, 0))
    q_padded = jnp.pad(q, pad_q)
    k_padded = jnp.pad(k, pad_k)
    q_trunked = q_padded.reshape(
        q.shape[:-2] + (n_trunks, n_queries, q.shape[-1])
    )
    k_trunked = jnp.stack(
        [
            k_padded[..., i * n_queries : i * n_queries + n_keys, :]
            for i in range(n_trunks)
        ],
        axis=-3,
    )
    return q_trunked, k_trunked


def _pad_leading_dim(
    x: jnp.ndarray,
    pad_left: int,
    pad_right: int,
) -> jnp.ndarray:
    return jnp.pad(x, ((pad_left, pad_right),) + ((0, 0),) * (x.ndim - 1))
