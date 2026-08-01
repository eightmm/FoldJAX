"""Pure JAX diffusion conditioning for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.diffusion.atom import (
    atom_to_token_index,
    gather_token_pairs_to_atom_windows,
    gather_tokens_to_atoms,
    get_indexing_matrix,
    single_to_keys,
)
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.trunk_blocks.conditioning import (
    pairwise_conditioning_forward,
)

Params = Mapping[str, object]


def diffusion_conditioning_forward(
    params: Params,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    relative_position_encoding: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    token_layers: int | None = None,
    atoms_per_window_queries: int = 32,
    atoms_per_window_keys: int = 128,
    eps: float = 1e-5,
    lazy_token_trans_bias: bool = False,
) -> dict[str, jnp.ndarray]:
    """Run Boltz DiffusionConditioning."""

    z = pairwise_conditioning_forward(
        params["pairwise_conditioner"],
        z_trunk,
        relative_position_encoding,
        eps=eps,
    )
    q, c, p = atom_encoder_forward(
        params["atom_encoder"],
        feats,
        s_trunk,
        z,
        atoms_per_window_queries=atoms_per_window_queries,
        atoms_per_window_keys=atoms_per_window_keys,
        eps=eps,
    )
    token_proj = params["token_trans_proj_z"]
    if token_layers is not None:
        token_proj = token_proj[:token_layers]
    atoms = feats["ref_pos"].shape[1]
    w = atoms_per_window_queries
    h_keys = atoms_per_window_keys
    indexing = get_indexing_matrix(k=atoms // w, w=w, h_keys=h_keys)
    out = {
        "q": q,
        "c": c,
        "atom_to_token_idx": atom_to_token_index(feats["atom_to_token"]),
        "to_keys": lambda x: single_to_keys(x, indexing, w=w, h_keys=h_keys),
        "atom_enc_bias": _projection_list_forward(params["atom_enc_proj_z"], p, eps),
        "atom_dec_bias": _projection_list_forward(params["atom_dec_proj_z"], p, eps),
    }
    if lazy_token_trans_bias:
        out["token_trans_bias_params"] = token_proj
        out["token_trans_bias_normed_input"] = _projection_input_norm(z, eps)
    else:
        out["token_trans_bias"] = _projection_list_forward(token_proj, z, eps)
    return out


def atom_encoder_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    s_trunk: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    structure_prediction: bool = True,
    atoms_per_window_queries: int = 32,
    atoms_per_window_keys: int = 128,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run Boltz AtomEncoder for structure diffusion conditioning."""

    atom_ref_pos = feats["ref_pos"]
    batch, atoms, _ = atom_ref_pos.shape
    atom_mask = feats["atom_pad_mask"].astype(bool)
    atom_uid = feats["ref_space_uid"]

    atom_feats = jnp.concatenate(
        (
            atom_ref_pos,
            feats["ref_charge"][..., None],
            feats["ref_element"],
            jnp.reshape(feats["ref_atom_name_chars"], (batch, atoms, 4 * 64)),
        ),
        axis=-1,
    )
    c = _linear(
        atom_feats,
        params["embed_atom_features"]["kernel"],
        params["embed_atom_features"]["bias"],
    )

    w = atoms_per_window_queries
    h_keys = atoms_per_window_keys
    num_windows = atoms // w
    indexing = get_indexing_matrix(k=num_windows, w=w, h_keys=h_keys)

    def to_keys(x: jnp.ndarray) -> jnp.ndarray:
        return single_to_keys(x, indexing, w=w, h_keys=h_keys)

    atom_ref_pos_queries = jnp.reshape(atom_ref_pos, (batch, num_windows, w, 1, 3))
    atom_ref_pos_keys = jnp.reshape(
        to_keys(atom_ref_pos), (batch, num_windows, 1, h_keys, 3)
    )
    d = atom_ref_pos_keys - atom_ref_pos_queries
    d_norm = 1.0 / (1.0 + jnp.sum(d * d, axis=-1, keepdims=True))

    atom_mask_queries = jnp.reshape(atom_mask, (batch, num_windows, w, 1))
    atom_mask_keys = jnp.reshape(
        to_keys(atom_mask[..., None].astype(jnp.float32)),
        (batch, num_windows, 1, h_keys),
    ).astype(bool)
    atom_uid_queries = jnp.reshape(atom_uid, (batch, num_windows, w, 1))
    atom_uid_keys = jnp.reshape(
        to_keys(atom_uid[..., None].astype(jnp.float32)),
        (batch, num_windows, 1, h_keys),
    )
    valid = (
        atom_mask_queries
        & atom_mask_keys
        & (atom_uid_queries == atom_uid_keys.astype(atom_uid.dtype))
    ).astype(atom_ref_pos.dtype)[..., None]

    p = _linear(d, params["embed_atompair_ref_pos"]["kernel"]) * valid
    p = p + _linear(d_norm, params["embed_atompair_ref_dist"]["kernel"]) * valid
    p = p + _linear(valid, params["embed_atompair_mask"]["kernel"]) * valid

    q = c
    if structure_prediction:
        if s_trunk is None or z is None:
            msg = "s_trunk and z are required when structure_prediction=True"
            raise ValueError(msg)
        s_to_c = params["s_to_c_trans"]
        s_to_c_out = _linear(
            _layer_norm(
                s_trunk,
                s_to_c["norm"]["scale"],
                s_to_c["norm"]["bias"],
                eps,
            ),
            s_to_c["linear"]["kernel"],
        )
        c = c + gather_tokens_to_atoms(
            feats["atom_to_token"].astype(jnp.float32),
            s_to_c_out,
        ).astype(c.dtype)

        z_to_p = params["z_to_p_trans"]
        z_to_p_out = _linear(
            _layer_norm(
                z,
                z_to_p["norm"]["scale"],
                z_to_p["norm"]["bias"],
                eps,
            ),
            z_to_p["linear"]["kernel"],
        )
        atom_to_token_queries = jnp.reshape(
            feats["atom_to_token"].astype(jnp.float32),
            (batch, num_windows, w, feats["atom_to_token"].shape[-1]),
        )
        atom_to_token_keys = to_keys(feats["atom_to_token"].astype(jnp.float32))
        p = p + gather_token_pairs_to_atom_windows(
            z_to_p_out,
            atom_to_token_queries,
            atom_to_token_keys,
        ).astype(p.dtype)

    p = p + _linear(
        jax.nn.relu(jnp.reshape(c, (batch, num_windows, w, 1, c.shape[-1]))),
        params["c_to_p_trans_q"]["kernel"],
    )
    p = p + _linear(
        jax.nn.relu(
            jnp.reshape(to_keys(c), (batch, num_windows, 1, h_keys, c.shape[-1]))
        ),
        params["c_to_p_trans_k"]["kernel"],
    )
    p = p + _p_mlp_forward(params["p_mlp"], p)
    return q, c, p


def _projection_list_forward(
    params: list[Params],
    x: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    # boltz-faithful per-layer loop: each layer LayerNorms the same input ``x``
    # then applies a Linear(..., heads). boltz holds the full [N,N,L*heads]
    # concat result, but its *compute* intermediate is only [N,N,heads] per
    # layer (boltz/diffusion_conditioning.py:74-114).
    #
    # The previous implementation stacked the L layers into one einsum, which
    # materialized a [..., L, C] ``normed`` buffer (B*N*N*L*C floats, ~48 GiB at
    # N=2048, C=128, L=24) and OOMed. Here we LayerNorm ``x`` once (the reduction
    # is layer-constant) and then loop over the L layers, producing one
    # [..., heads] block at a time and concatenating along the feature axis. The
    # only large live buffers are ``x_n`` ([..., C]) and the accumulating output
    # ([..., L*heads]) -- never the [..., L, C] product. Concat order over layers
    # is preserved, so the result is identical to the old path up to fp32
    # reduce-order (< 1e-5).
    x_n = _projection_input_norm(x, eps)

    outs = []
    for layer in params:
        normed = x_n * layer["norm"]["scale"] + layer["norm"]["bias"]
        outs.append(normed @ layer["linear"]["kernel"])
    return jnp.concatenate(outs, axis=-1)


def _projection_input_norm(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    """LayerNorm input shared by all layers in a ProjectionList."""

    in_dtype = x.dtype
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * jax.lax.rsqrt(variance + eps)).astype(in_dtype)


def _p_mlp_forward(params: list[Params], p: jnp.ndarray) -> jnp.ndarray:
    out = p
    for layer in params:
        out = _linear(jax.nn.relu(out), layer["kernel"])
    return out
