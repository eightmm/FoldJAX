"""Pure JAX atom-window transformer utilities for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.nn
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import cp_mesh, cp_row_shards, shard_single
from foldjax.models._cp_atom import (
    atom_axis_name,
    atom_spec,
    gather_tokens_to_atoms_cp,
    scatter_atoms_to_tokens_mean_cp,
    shard_atoms,
    single_to_keys_local,
    window_spec,
)
from foldjax.models._stacking import take_layers
from foldjax.models.boltz2.models.diffusion.diffusion_transformer import (
    diffusion_transformer_layer_apply,
    layer_s_terms,
)
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.primitives._scan_utils import stack_layer_params

Params = Mapping[str, object]


def get_indexing_matrix(k: int, w: int, h_keys: int) -> jnp.ndarray:
    """Return Boltz atom-window key indexing matrix."""

    if w % 2 != 0:
        msg = f"W must be even, got {w}"
        raise ValueError(msg)
    if h_keys % (w // 2) != 0:
        msg = f"H must be divisible by W // 2, got W={w}, H={h_keys}"
        raise ValueError(msg)

    h = h_keys // (w // 2)
    if h % 2 != 0:
        msg = f"H // (W // 2) must be even, got {h}"
        raise ValueError(msg)

    arange = jnp.arange(2 * k)
    index = jnp.clip((arange[None, :] - arange[:, None]) + h // 2, 0, h + 1)
    index = jnp.reshape(index, (k, 2, 2 * k))[:, 0, :]
    onehot = jax.nn.one_hot(index, h + 2)[..., 1:-1]
    onehot = jnp.transpose(onehot, (1, 0, 2))
    return jnp.reshape(onehot, (2 * k, h * k)).astype(jnp.float32)


def single_to_keys(
    single: jnp.ndarray,
    indexing_matrix: jnp.ndarray,
    w: int,
    h_keys: int,
) -> jnp.ndarray:
    """Map atom-window query-aligned values to key windows."""

    batch, atoms, dim = single.shape
    k = atoms // w
    single = jnp.reshape(single, (batch, 2 * k, w // 2, dim))
    key_source = jnp.argmax(indexing_matrix, axis=0)
    key_valid = jnp.any(indexing_matrix > 0, axis=0)
    keys = jnp.take(single, key_source, axis=1)
    keys = keys * key_valid[None, :, None, None].astype(keys.dtype)
    return jnp.reshape(keys, (batch, k, h_keys, dim))


def gather_tokens_to_atoms(
    atom_to_token: jnp.ndarray,
    token_values: jnp.ndarray,
    index: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Apply dense one-hot atom->token map as a batched gather.

    ``index`` may carry precomputed ``(token_idx, valid)`` from
    :func:`atom_to_token_index` so the argmax over the dense map is not
    recomputed inside the per-step sampler loop. Bit-identical either way.
    """

    token_idx, valid = _one_hot_index(atom_to_token) if index is None else index
    gathered = jnp.take_along_axis(token_values, token_idx[..., None], axis=1)
    return gathered * valid[..., None].astype(gathered.dtype)


def scatter_atoms_to_tokens_mean(
    atom_to_token: jnp.ndarray,
    atom_values: jnp.ndarray,
    eps: float = 1e-6,
    index: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Apply normalized dense one-hot atom->token map as scatter mean.

    ``index`` may carry precomputed ``(token_idx, valid)``; see
    :func:`gather_tokens_to_atoms`. Bit-identical either way.
    """

    batch, _, tokens = atom_to_token.shape
    token_idx, valid = _one_hot_index(atom_to_token) if index is None else index

    def one(batch_idx: jnp.ndarray, valid_b: jnp.ndarray, values_b: jnp.ndarray):
        values_b = values_b * valid_b[:, None].astype(values_b.dtype)
        summed = jnp.zeros((tokens, values_b.shape[-1]), dtype=values_b.dtype)
        counts = jnp.zeros((tokens,), dtype=values_b.dtype)
        summed = summed.at[batch_idx].add(values_b)
        counts = counts.at[batch_idx].add(valid_b.astype(values_b.dtype))
        return summed / (counts[:, None] + eps)

    return jax.vmap(one)(token_idx, valid, atom_values)


def gather_token_pairs_to_atom_windows(
    token_pair_values: jnp.ndarray,
    atom_to_token_queries: jnp.ndarray,
    atom_to_token_keys: jnp.ndarray,
) -> jnp.ndarray:
    """Apply two dense one-hot atom->token maps to token-pair values."""

    q_idx, q_valid = _one_hot_index(atom_to_token_queries)
    k_idx, k_valid = _one_hot_index(atom_to_token_keys)
    batch = token_pair_values.shape[0]
    batch_idx = jnp.arange(batch)[:, None, None, None]
    values = token_pair_values[
        batch_idx,
        q_idx[:, :, :, None],
        k_idx[:, :, None, :],
    ]
    valid = q_valid[:, :, :, None] & k_valid[:, :, None, :]
    return values * valid[..., None].astype(values.dtype)


def gather_rep_atoms_to_tokens(
    token_to_rep_atom: jnp.ndarray,
    atom_values: jnp.ndarray,
) -> jnp.ndarray:
    """Apply dense one-hot token->representative-atom map as a gather."""

    atom_idx, valid = _one_hot_index(token_to_rep_atom)
    gathered = jnp.take_along_axis(atom_values, atom_idx[..., None], axis=1)
    return gathered * valid[..., None].astype(gathered.dtype)


def _one_hot_index(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = jnp.argmax(x, axis=-1)
    valid = jnp.any(x > 0, axis=-1)
    return idx, valid


def atom_to_token_index(
    atom_to_token: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute ``(token_idx, valid)`` for a dense one-hot atom->token map.

    Returns exactly what :func:`_one_hot_index` produces on the (unrepeated)
    dense map, so callers can hoist the argmax/any out of the per-step sampler
    loop while staying bit-identical. Repeat by multiplicity at use site via
    :func:`_repeat_index`.
    """

    return _one_hot_index(atom_to_token.astype(jnp.float32))


def atom_to_token_index_from_feats(
    feats: Mapping[str, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Read compact global token IDs, falling back to the legacy one-hot map."""

    if "atom_to_token_ids_global" in feats:
        indices = feats["atom_to_token_ids_global"].astype(jnp.int32)
        valid = feats.get("atom_to_token_valid")
        if valid is None:
            valid = indices >= 0
        return jnp.maximum(indices, 0), valid.astype(bool)
    return atom_to_token_index(feats["atom_to_token"])


def _repeat_index(
    index: tuple[jnp.ndarray, jnp.ndarray] | None, multiplicity: int
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    if index is None or multiplicity == 1:
        return index
    idx, valid = index
    return (
        jnp.repeat(idx, multiplicity, axis=0),
        jnp.repeat(valid, multiplicity, axis=0),
    )


def diffusion_transformer_forward(
    params: Params,
    a: jnp.ndarray,
    s: jnp.ndarray,
    bias: jnp.ndarray | None,
    mask: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    multiplicity: int = 1,
    eps: float = 1e-5,
    use_scan: bool = True,
    attention_backend: str = "xla",
    chunk_size: int | None = None,
    layer_limit: int | None = None,
    bias_params: list[Params] | None = None,
    bias_input: jnp.ndarray | None = None,
    bias_normed_input: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Run a Boltz DiffusionTransformer stack.

    ``use_scan=False`` unrolls the layer stack in Python. ``use_scan=True``
    (default) runs the stack via ``lax.scan`` and preserves the memory-stable
    production path.

    ``chunk_size`` (default ``None``) enables query-axis blocking of the
    per-layer pair-bias self-attention so the ``[b, heads, N, N]`` score buffer
    is never fully materialized. It is bit-exact (softmax reduces over the full
    key axis within each query block). Used for the token transformer at large
    N; left ``None`` for the windowed atom transformer.
    """

    layers = take_layers(params["layers"], layer_limit)
    num_layers = len(layers)
    if bias is None:
        if bias_params is None or (bias_input is None and bias_normed_input is None):
            msg = (
                "bias_params and either bias_input or bias_normed_input are "
                "required when bias is None"
            )
            raise ValueError(msg)
        bias_params = take_layers(bias_params, num_layers)
        bias_per_layer = None
        stacked_bias_params = stack_layer_params(bias_params)
    else:
        batch, rows, cols, dim = bias.shape
        bias = jnp.reshape(bias, (batch, rows, cols, num_layers, dim // num_layers))
        # Move the per-layer axis to the front so it indexes by layer.
        bias_per_layer = jnp.transpose(bias, (3, 0, 1, 2, 4))
        stacked_bias_params = None

    # Hoist the layer-constant, s-derived AdaLN scale/bias and output gates out
    # of the per-layer loop. ``s`` is identical for every layer (and every
    # diffusion step), so all layers' s-terms are computed in one batched pass
    # over the stacked layer params instead of L times inside the loop.
    stacked = stack_layer_params(layers)
    s_terms_stacked = jax.vmap(lambda lp: layer_s_terms(lp, s, eps))(stacked)

    if not use_scan:
        for i, layer_params in enumerate(layers):
            s_terms_i = jax.tree.map(lambda x, i=i: x[i], s_terms_stacked)
            layer_bias = (
                _projection_layer_forward(
                    bias_params[i],
                    bias_input,
                    eps,
                    normed_input=bias_normed_input,
                )
                if bias_per_layer is None
                else bias_per_layer[i]
            )
            a = diffusion_transformer_layer_apply(
                layer_params,
                a,
                s_terms_i,
                layer_bias,
                mask,
                to_keys=to_keys,
                multiplicity=multiplicity,
                eps=eps,
                attention_backend=attention_backend,
                chunk_size=chunk_size,
            )
        return a

    def body(a_c, layer):
        in_dtype = a_c.dtype
        if bias_per_layer is None:
            layer_params, layer_bias_params, layer_s_terms_i = layer
            layer_bias = _projection_layer_forward(
                layer_bias_params,
                bias_input,
                eps,
                normed_input=bias_normed_input,
            )
        else:
            layer_params, layer_bias, layer_s_terms_i = layer
        a_c = diffusion_transformer_layer_apply(
            layer_params,
            a_c,
            layer_s_terms_i,
            layer_bias,
            mask,
            to_keys=to_keys,
            multiplicity=multiplicity,
            eps=eps,
            attention_backend=attention_backend,
            chunk_size=chunk_size,
        )
        # lax.scan requires carry dtype stability; an internal op may promote
        # to fp32 (e.g. under jax_default_matmul_precision='default' with low
        # dtype inputs). Keep the carry in its input dtype. fp32 in -> fp32 out
        # is a no-op, so fp32 behavior is unchanged.
        return a_c.astype(in_dtype), None

    scan_bias = stacked_bias_params if bias_per_layer is None else bias_per_layer
    a, _ = jax.lax.scan(body, a, (stacked, scan_bias, s_terms_stacked))
    return a


def _projection_layer_forward(
    params: Params,
    x: jnp.ndarray | None,
    eps: float,
    *,
    normed_input: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """One layer of DiffusionConditioning token bias projection."""
    if normed_input is None:
        if x is None:
            msg = "x is required when normed_input is None"
            raise ValueError(msg)
        in_dtype = x.dtype
        xf = x.astype(jnp.float32)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        x_n = ((xf - mean) * jax.lax.rsqrt(variance + eps)).astype(in_dtype)
    else:
        x_n = normed_input
    normed = x_n * params["norm"]["scale"] + params["norm"]["bias"]
    return normed @ params["linear"]["kernel"]


def atom_transformer_forward(
    params: Params,
    q: jnp.ndarray,
    c: jnp.ndarray,
    bias: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray],
    mask: jnp.ndarray,
    attn_window_queries: int,
    attn_window_keys: int,
    multiplicity: int = 1,
    eps: float = 1e-5,
    attention_backend: str = "xla",
    atom_context_parallel: bool = False,
) -> jnp.ndarray:
    """Run Boltz AtomTransformer with serial or halo-sharded windows."""

    if atom_context_parallel and cp_mesh() is not None:
        return _atom_transformer_forward_cp(
            params,
            q,
            c,
            bias,
            mask,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            multiplicity=multiplicity,
            eps=eps,
            attention_backend=attention_backend,
        )

    w = attn_window_queries
    h_keys = attn_window_keys
    batch, atoms, dim = q.shape
    num_windows = atoms // w

    q = jnp.reshape(q, (batch * num_windows, w, dim))
    c = jnp.reshape(c, (batch * num_windows, w, c.shape[-1]))
    mask = jnp.reshape(mask, (batch * num_windows, w))
    bias = jnp.repeat(bias, multiplicity, axis=0)
    bias = jnp.reshape(
        bias,
        (bias.shape[0] * num_windows, w, h_keys, bias.shape[-1]),
    )

    def to_keys_new(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.reshape(x, (batch, num_windows * w, -1))
        return jnp.reshape(to_keys(x), (batch * num_windows, h_keys, -1))

    q = diffusion_transformer_forward(
        params["diffusion_transformer"],
        a=q,
        s=c,
        bias=bias,
        mask=mask.astype(jnp.float32),
        to_keys=to_keys_new,
        multiplicity=1,
        eps=eps,
        attention_backend=attention_backend,
    )
    return jnp.reshape(q, (batch, num_windows * w, dim))


def _atom_transformer_forward_cp(
    params: Params,
    q: jnp.ndarray,
    c: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    attn_window_queries: int,
    attn_window_keys: int,
    multiplicity: int,
    eps: float,
    attention_backend: str,
) -> jnp.ndarray:
    """Run each device's owned query windows with neighbour halo exchange."""

    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("atom CP adapter requires an active mesh")
    if q.ndim != 3 or c.ndim != 3 or mask.ndim != 2 or bias.ndim != 5:
        raise ValueError(
            "atom CP expects q/c [B,A,C], mask [B,A], and bias [B,K,W,H,C]"
        )
    w = attn_window_queries
    h_keys = attn_window_keys
    axis_name = atom_axis_name()
    rows = cp_row_shards()

    def local(params_l, q_l, c_l, bias_l, mask_l):
        batch, local_atoms, dim = q_l.shape
        if local_atoms % w:
            raise ValueError("local atom shard is not query-window aligned")
        local_windows = local_atoms // w
        q_flat = q_l.reshape(batch * local_windows, w, dim)
        c_flat = c_l.reshape(batch * local_windows, w, c_l.shape[-1])
        mask_flat = mask_l.reshape(batch * local_windows, w)
        bias_l = jnp.repeat(bias_l, multiplicity, axis=0)
        if bias_l.shape[0] != batch:
            raise ValueError(
                "atom bias multiplicity does not match the atom activation batch"
            )
        bias_flat = bias_l.reshape(
            batch * local_windows,
            w,
            h_keys,
            bias_l.shape[-1],
        )

        def to_keys_local(x: jnp.ndarray) -> jnp.ndarray:
            x = x.reshape(batch, local_atoms, -1)
            keys = single_to_keys_local(
                x,
                query_window=w,
                key_window=h_keys,
                axis_name=axis_name,
                axis_size=rows,
            )
            return keys.reshape(batch * local_windows, h_keys, -1)

        out = diffusion_transformer_forward(
            params_l["diffusion_transformer"],
            a=q_flat,
            s=c_flat,
            bias=bias_flat,
            mask=mask_flat.astype(jnp.float32),
            to_keys=to_keys_local,
            multiplicity=1,
            eps=eps,
            attention_backend=attention_backend,
        )
        return out.reshape(batch, local_atoms, dim)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            PartitionSpec(),
            atom_spec(3, atom_axis=1),
            atom_spec(3, atom_axis=1),
            window_spec(5, window_axis=1),
            atom_spec(2, atom_axis=1),
        ),
        out_specs=atom_spec(3, atom_axis=1),
    )(params, q, c, bias, mask)


def atom_attention_encoder_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    q: jnp.ndarray,
    c: jnp.ndarray,
    atom_enc_bias: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray],
    r: jnp.ndarray | None = None,
    multiplicity: int = 1,
    attn_window_queries: int = 32,
    attn_window_keys: int = 128,
    structure_prediction: bool = True,
    eps: float = 1e-5,
    attention_backend: str = "xla",
    atom_to_token_idx: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    atom_context_parallel: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run AtomAttentionEncoder with dense or CP-row atom ownership."""

    active = atom_context_parallel and cp_mesh() is not None
    atom_mask = feats["atom_pad_mask"].astype(bool)
    if active:
        atom_mask = shard_atoms(atom_mask, atom_axis=1)
        q = shard_atoms(q, atom_axis=1)
        c = shard_atoms(c, atom_axis=1)
        atom_enc_bias = atom_enc_bias
    if structure_prediction:
        if r is None:
            raise ValueError("r is required when structure_prediction=True")
        if active:
            r = shard_atoms(r, atom_axis=1)
        q = jnp.repeat(q, multiplicity, axis=0)
        q = q + _linear(r, params["r_to_q_trans"]["kernel"])
    c = jnp.repeat(c, multiplicity, axis=0)
    atom_mask = jnp.repeat(atom_mask, multiplicity, axis=0)

    q = atom_transformer_forward(
        params["atom_encoder"],
        q=q,
        c=c,
        bias=atom_enc_bias,
        to_keys=to_keys,
        mask=atom_mask.astype(jnp.float32),
        attn_window_queries=attn_window_queries,
        attn_window_keys=attn_window_keys,
        multiplicity=multiplicity,
        eps=eps,
        attention_backend=attention_backend,
        atom_context_parallel=active,
    )

    q_to_a = jax.nn.relu(_linear(q, params["atom_to_token_trans"]["kernel"])).astype(
        q.dtype
    )
    idx = _repeat_index(
        atom_to_token_idx or atom_to_token_index_from_feats(feats),
        multiplicity,
    )
    if active:
        a = scatter_atoms_to_tokens_mean_cp(
            q_to_a,
            idx[0],
            idx[1],
            num_tokens=feats["token_pad_mask"].shape[1],
        ).astype(q.dtype)
        a = shard_single(a, token_axis=1)
    else:
        atom_to_token = jnp.repeat(
            feats["atom_to_token"].astype(jnp.float32), multiplicity, axis=0
        )
        a = scatter_atoms_to_tokens_mean(
            atom_to_token,
            q_to_a,
            index=idx,
        ).astype(q.dtype)
    return a, q, c


def atom_attention_decoder_forward(
    params: Params,
    a: jnp.ndarray,
    q: jnp.ndarray,
    c: jnp.ndarray,
    atom_dec_bias: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    to_keys: Callable[[jnp.ndarray], jnp.ndarray],
    multiplicity: int = 1,
    attn_window_queries: int = 32,
    attn_window_keys: int = 128,
    eps: float = 1e-5,
    attention_backend: str = "xla",
    atom_to_token_idx: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    atom_context_parallel: bool = False,
) -> jnp.ndarray:
    """Run AtomAttentionDecoder with explicit token-to-atom routing."""

    active = atom_context_parallel and cp_mesh() is not None
    idx = _repeat_index(
        atom_to_token_idx or atom_to_token_index_from_feats(feats),
        multiplicity,
    )
    projected = _linear(a.astype(q.dtype), params["a_to_q_trans"]["kernel"])
    if active:
        a = shard_single(a, token_axis=1)
        projected = shard_single(projected, token_axis=1)
        q = shard_atoms(q, atom_axis=1)
        c = shard_atoms(c, atom_axis=1)
        a_to_q = gather_tokens_to_atoms_cp(projected, idx[0], idx[1])
    else:
        atom_to_token = jnp.repeat(
            feats["atom_to_token"].astype(jnp.float32), multiplicity, axis=0
        )
        a_to_q = gather_tokens_to_atoms(
            atom_to_token,
            projected,
            index=idx,
        )
    q = q + a_to_q.astype(q.dtype)
    atom_mask = jnp.repeat(feats["atom_pad_mask"], multiplicity, axis=0)
    if active:
        atom_mask = shard_atoms(atom_mask, atom_axis=1)

    q = atom_transformer_forward(
        params["atom_decoder"],
        q=q,
        c=c,
        bias=atom_dec_bias,
        to_keys=to_keys,
        mask=atom_mask.astype(jnp.float32),
        attn_window_queries=attn_window_queries,
        attn_window_keys=attn_window_keys,
        multiplicity=multiplicity,
        eps=eps,
        attention_backend=attention_backend,
        atom_context_parallel=active,
    )

    update = params["atom_feat_to_atom_pos_update"]
    q = _layer_norm(q, update["norm"]["scale"], update["norm"]["bias"], eps)
    out = _linear(q, update["linear"]["kernel"])
    return shard_atoms(out, atom_axis=1) if active else out
