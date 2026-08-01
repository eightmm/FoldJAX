"""OpenDDE structural-token expansion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
    silu,
)

STRUCTURAL_TOKEN_ROLES = {
    "atom": 0,
    "protein_bb": 1,
    "protein_sc": 2,
    "dna_bb": 3,
    "dna_base": 4,
    "rna_bb": 5,
    "rna_base": 6,
}


class StructuralTokenExpanderParams(NamedTuple):
    """Parameters for OpenDDE's released full role-pair projection."""

    single_split_layer_norm: LayerNormParams
    single_split_linear_in: LinearParams
    single_split_linear_out: LinearParams
    single_input_role_embedding: jnp.ndarray
    single_role_embedding: jnp.ndarray
    pair_block_proj: jnp.ndarray
    same_parent_embedding: jnp.ndarray
    same_residue_twin_embedding: jnp.ndarray
    prev_bb_chain_embedding: jnp.ndarray
    next_bb_chain_embedding: jnp.ndarray
    role_pair_type_embedding: jnp.ndarray
    attn_bias_same_parent: jnp.ndarray
    attn_bias_same_residue_twin: jnp.ndarray
    attn_bias_prev_bb_chain: jnp.ndarray
    attn_bias_next_bb_chain: jnp.ndarray
    attn_bias_role_pair_type: jnp.ndarray


def build_structural_pair_features(
    input_features: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Build OpenDDE role, twin, and directional backbone pair features."""

    role = jnp.asarray(input_features["subtoken_role_id"], dtype=jnp.int32)
    parent = jnp.asarray(input_features["parent_residue_idx"], dtype=jnp.int32)
    asym_id = jnp.take(jnp.asarray(input_features["asym_id"]), parent, axis=-1)
    residue_index = jnp.take(
        jnp.asarray(input_features["residue_index"]), parent, axis=-1
    )
    polymer_type = jnp.asarray(
        input_features.get("structural_polymer_type", jnp.zeros_like(role)),
        dtype=jnp.int32,
    )
    prev_parent = jnp.asarray(
        input_features.get("prev_parent_residue_idx", jnp.full_like(parent, -1)),
        dtype=jnp.int32,
    )
    next_parent = jnp.asarray(
        input_features.get("next_parent_residue_idx", jnp.full_like(parent, -1)),
        dtype=jnp.int32,
    )

    backbone = (
        (role == STRUCTURAL_TOKEN_ROLES["protein_bb"])
        | (role == STRUCTURAL_TOKEN_ROLES["dna_bb"])
        | (role == STRUCTURAL_TOKEN_ROLES["rna_bb"])
    )
    sidechain = role == STRUCTURAL_TOKEN_ROLES["protein_sc"]
    base = (role == STRUCTURAL_TOKEN_ROLES["dna_base"]) | (
        role == STRUCTURAL_TOKEN_ROLES["rna_base"]
    )
    same_parent = parent[:, None] == parent[None, :]
    same_chain = asym_id[:, None] == asym_id[None, :]
    same_polymer_type = (polymer_type[:, None] == polymer_type[None, :]) & (
        polymer_type[:, None] > 0
    )
    same_twin = same_parent & (
        (backbone[:, None] & (sidechain[None, :] | base[None, :]))
        | (backbone[None, :] & (sidechain[:, None] | base[:, None]))
    )
    prev_chain = (
        backbone[:, None]
        & backbone[None, :]
        & same_chain
        & (prev_parent[:, None] == parent[None, :])
    )
    next_chain = (
        backbone[:, None]
        & backbone[None, :]
        & same_chain
        & (next_parent[:, None] == parent[None, :])
    )

    role_pair_type = jnp.full(same_parent.shape, 7, dtype=jnp.int32)
    role_pair_type = jnp.where(backbone[:, None] & backbone[None, :], 0, role_pair_type)
    role_pair_type = jnp.where(
        backbone[:, None] & sidechain[None, :], 1, role_pair_type
    )
    role_pair_type = jnp.where(
        sidechain[:, None] & backbone[None, :], 2, role_pair_type
    )
    role_pair_type = jnp.where(
        sidechain[:, None] & sidechain[None, :], 3, role_pair_type
    )
    role_pair_type = jnp.where(backbone[:, None] & base[None, :], 4, role_pair_type)
    role_pair_type = jnp.where(base[:, None] & backbone[None, :], 5, role_pair_type)
    role_pair_type = jnp.where(base[:, None] & base[None, :], 6, role_pair_type)

    return {
        "same_parent_residue": same_parent,
        "same_residue_twin": same_twin,
        "prev_bb_chain": prev_chain,
        "next_bb_chain": next_chain,
        "role_pair_type": role_pair_type,
        "same_chain": same_chain,
        "same_polymer_type": same_polymer_type,
        "residue_index": residue_index,
    }


def structural_token_expand(
    input_features: Mapping[str, jnp.ndarray],
    s_inputs_res: jnp.ndarray,
    s_res: jnp.ndarray,
    z_res: jnp.ndarray,
    params: StructuralTokenExpanderParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Expand residue activations into OpenDDE structural-token activations."""

    parent = jnp.asarray(input_features["parent_residue_idx"], dtype=jnp.int32)
    role = jnp.asarray(input_features["subtoken_role_id"], dtype=jnp.int32)
    s_inputs_struct = jnp.take(
        s_inputs_res, parent, axis=-2
    ) + params.single_input_role_embedding[role].astype(s_inputs_res.dtype)
    s_parent = jnp.take(s_res, parent, axis=-2)
    hidden = layer_norm(s_parent, params.single_split_layer_norm)
    hidden = linear(hidden, params.single_split_linear_in)
    hidden = linear(silu(hidden), params.single_split_linear_out)
    s_struct = (
        s_parent + hidden + params.single_role_embedding[role].astype(s_parent.dtype)
    )

    pair = build_structural_pair_features(input_features)
    z_parent = jnp.take(jnp.take(z_res, parent, axis=-3), parent, axis=-2)
    z_struct = z_parent + _pair_project_by_role(
        z_parent,
        role,
        params.pair_block_proj.astype(z_parent.dtype),
    )
    z_struct = z_struct + _pair_embedding_bias(pair, params, z_parent.dtype)
    pair["structural_pair_attn_bias"] = _pair_attention_bias(
        pair, params, z_parent.dtype
    )
    return s_inputs_struct, s_struct, z_struct, pair


def _pair_project_by_role(
    z: jnp.ndarray,
    role: jnp.ndarray,
    pair_block_proj: jnp.ndarray,
) -> jnp.ndarray:
    """Apply one released projection per structural-token role pair.

    Selecting all pair matrices at once creates an ``[N, N, C_out, C_in]``
    tensor.  At the released width and a few hundred structural tokens that is
    over 100 GiB.  Mapping over structural rows keeps only
    ``[N, C_out, C_in]`` selected weights live while preserving the exact
    per-pair linear projection and arbitrary leading batch dimensions.
    """

    z = jnp.asarray(z)
    role = jnp.asarray(role, dtype=jnp.int32)
    pair_block_proj = jnp.asarray(pair_block_proj)
    if role.ndim != 1:
        raise ValueError(f"role must be one-dimensional, got {role.shape}")
    n_structural = role.shape[0]
    if z.ndim < 3 or z.shape[-3:-1] != (n_structural, n_structural):
        raise ValueError(
            f"z pair axes must match role length, got z={z.shape}, role={role.shape}"
        )
    if pair_block_proj.ndim != 4 or pair_block_proj.shape[-1] != z.shape[-1]:
        raise ValueError(
            "pair_block_proj must have shape [roles, roles, C_out, C_in], got "
            f"{pair_block_proj.shape} for z={z.shape}"
        )

    rows = jnp.moveaxis(z, -3, 0)

    def project_row(inputs):
        row, row_role = inputs
        row_weights = pair_block_proj[row_role, role]
        return jnp.einsum("...jc,joc->...jo", row, row_weights)

    projected_rows = jax.lax.map(project_row, (rows, role))
    return jnp.moveaxis(projected_rows, 0, -3)


def _pair_embedding_bias(
    pair: Mapping[str, jnp.ndarray],
    params: StructuralTokenExpanderParams,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    return (
        params.same_parent_embedding[
            pair["same_parent_residue"].astype(jnp.int32)
        ].astype(dtype)
        + params.same_residue_twin_embedding[
            pair["same_residue_twin"].astype(jnp.int32)
        ].astype(dtype)
        + params.prev_bb_chain_embedding[
            pair["prev_bb_chain"].astype(jnp.int32)
        ].astype(dtype)
        + params.next_bb_chain_embedding[
            pair["next_bb_chain"].astype(jnp.int32)
        ].astype(dtype)
        + params.role_pair_type_embedding[pair["role_pair_type"]].astype(dtype)
    )


def _pair_attention_bias(
    pair: Mapping[str, jnp.ndarray],
    params: StructuralTokenExpanderParams,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    return (
        params.attn_bias_same_parent.astype(dtype)
        * pair["same_parent_residue"].astype(dtype)
        + params.attn_bias_same_residue_twin.astype(dtype)
        * pair["same_residue_twin"].astype(dtype)
        + params.attn_bias_prev_bb_chain.astype(dtype)
        * pair["prev_bb_chain"].astype(dtype)
        + params.attn_bias_next_bb_chain.astype(dtype)
        * pair["next_bb_chain"].astype(dtype)
        + params.attn_bias_role_pair_type[pair["role_pair_type"]].astype(dtype)
    )
