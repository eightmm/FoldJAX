"""Embedder leaf functions ported from Protenix."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionEncoderParams,
    atom_attention_encoder,
)
from foldjax.models.protenix.models.primitives.primitives import LinearParams, linear


class FourierParams(NamedTuple):
    """Parameters for ``protenix.model.modules.embedders.FourierEmbedding``."""

    w: jnp.ndarray
    b: jnp.ndarray


class RelativePositionParams(NamedTuple):
    """Parameters for ``RelativePositionEncoding``."""

    linear_no_bias: LinearParams


class InputFeatureEmbedderParams(NamedTuple):
    """Parameters for ``InputFeatureEmbedder``."""

    atom_encoder: AtomAttentionEncoderParams
    linear_esm: LinearParams | None = None


class SubstructureMlpParams(NamedTuple):
    """Parameters for MLP-mode ``SubstructureEmbedder``."""

    layers: tuple[LinearParams, ...]


class ConstraintEmbedderParams(NamedTuple):
    """Parameters for ``ConstraintEmbedder`` optional projections."""

    pocket_z: LinearParams | None = None
    contact_z: LinearParams | None = None
    contact_atom_z: LinearParams | None = None
    substructure_z: SubstructureMlpParams | None = None


def fourier_embedding(
    t_hat_noise_level: jnp.ndarray,
    params: FourierParams,
) -> jnp.ndarray:
    """Apply Protenix Fourier noise-level embedding."""

    return jnp.cos(2 * jnp.pi * (t_hat_noise_level[..., None] * params.w + params.b))


def relative_position_features(
    input_feature_dict: dict[str, jnp.ndarray],
    *,
    r_max: int = 32,
    s_max: int = 2,
) -> jnp.ndarray:
    """Generate Protenix relative-position features from token metadata."""

    asym_id = input_feature_dict["asym_id"]
    residue_index = input_feature_dict["residue_index"]
    entity_id = input_feature_dict["entity_id"]
    sym_id = input_feature_dict["sym_id"]
    token_index = input_feature_dict["token_index"]

    same_chain = asym_id[..., :, None] == asym_id[..., None, :]
    same_residue = residue_index[..., :, None] == residue_index[..., None, :]
    same_entity = entity_id[..., :, None] == entity_id[..., None, :]

    residue_delta = residue_index[..., :, None] - residue_index[..., None, :] + r_max
    residue_delta = jnp.clip(residue_delta, 0, 2 * r_max)
    residue_bins = jnp.where(same_chain, residue_delta, 2 * r_max + 1)

    token_delta = token_index[..., :, None] - token_index[..., None, :] + r_max
    token_delta = jnp.clip(token_delta, 0, 2 * r_max)
    same_chain_residue = same_chain & same_residue
    token_bins = jnp.where(same_chain_residue, token_delta, 2 * r_max + 1)

    chain_delta = sym_id[..., :, None] - sym_id[..., None, :] + s_max
    chain_delta = jnp.clip(chain_delta, 0, 2 * s_max)
    chain_bins = jnp.where(same_entity, chain_delta, 2 * s_max + 1)

    rel_pos = jnp.eye(2 * (r_max + 1), dtype=jnp.float32)[residue_bins]
    rel_token = jnp.eye(2 * (r_max + 1), dtype=jnp.float32)[token_bins]
    rel_chain = jnp.eye(2 * (s_max + 1), dtype=jnp.float32)[chain_bins]
    return jnp.concatenate(
        [
            rel_pos,
            rel_token,
            same_entity[..., None].astype(jnp.float32),
            rel_chain,
        ],
        axis=-1,
    )


def relative_position_encoding(
    relp_feature: jnp.ndarray,
    params: RelativePositionParams,
) -> jnp.ndarray:
    """Apply the relative-position linear projection."""

    # Protenix may supply the compact int8 feature, while OpenDDE retains its
    # float feature. Cast only the integer representation: matching the weight
    # preserves the historical BF16-trunk/FP32-diffusion split and also works
    # under JAX's strict dtype-promotion mode. Existing float callers retain
    # their normal promotion behavior.
    if jnp.issubdtype(relp_feature.dtype, jnp.integer):
        relp_feature = relp_feature.astype(params.linear_no_bias.weight.dtype)
    return linear(relp_feature, params.linear_no_bias)


def input_feature_embedder(
    input_feature_dict: dict[str, jnp.ndarray],
    params: InputFeatureEmbedderParams,
    *,
    n_token: int,
    n_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    use_scan: bool = False,
) -> jnp.ndarray:
    """Apply Protenix ``InputFeatureEmbedder`` inference path."""

    a, _, _, _ = atom_attention_encoder(
        input_feature_dict["atom_to_token_idx"],
        input_feature_dict["ref_pos"],
        input_feature_dict["ref_charge"],
        input_feature_dict["ref_mask"],
        input_feature_dict["ref_atom_name_chars"],
        input_feature_dict["ref_element"],
        input_feature_dict["d_lm"],
        input_feature_dict["v_lm"],
        input_feature_dict["pad_info"],
        params.atom_encoder,
        n_token=n_token,
        n_heads=n_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_scan,
        atom_mask=input_feature_dict.get("atom_padding_mask"),
    )
    batch_shape = input_feature_dict["restype"].shape[:-1]
    s_inputs = jnp.concatenate(
        [
            a,
            input_feature_dict["restype"].reshape(batch_shape + (32,)),
            input_feature_dict["profile"].reshape(batch_shape + (32,)),
            input_feature_dict["deletion_mean"].reshape(batch_shape + (1,)),
        ],
        axis=-1,
    )
    if params.linear_esm is not None and "esm_token_embedding" in input_feature_dict:
        s_inputs = s_inputs + linear(
            input_feature_dict["esm_token_embedding"],
            params.linear_esm,
        )
    token_mask = input_feature_dict.get("token_padding_mask")
    if token_mask is not None:
        s_inputs = s_inputs * jnp.asarray(token_mask, dtype=s_inputs.dtype)[..., None]
    return s_inputs


def substructure_mlp(
    x: jnp.ndarray,
    params: SubstructureMlpParams,
) -> jnp.ndarray:
    """Apply inference-mode MLP ``SubstructureEmbedder`` without dropout."""

    if not params.layers:
        return x
    for layer in params.layers[:-1]:
        x = jnp.maximum(linear(x, layer), 0.0)
    return linear(x, params.layers[-1])


def constraint_embedder(
    constraint_feature_dict: dict[str, jnp.ndarray],
    params: ConstraintEmbedderParams,
) -> jnp.ndarray | None:
    """Apply Protenix ``ConstraintEmbedder`` optional pair projections."""

    supported = {
        "pocket": params.pocket_z,
        "contact": params.contact_z,
        "contact_atom": params.contact_atom_z,
        "substructure": params.substructure_z,
    }
    auxiliary = {"contact_atom_index_pairs", "contact_atom_token_pairs"}
    unknown = set(constraint_feature_dict) - set(supported) - auxiliary
    if unknown:
        raise ValueError(f"unsupported constraint feature channels: {sorted(unknown)}")
    present_channels = set(constraint_feature_dict) & set(supported)
    if not present_channels:
        return None
    for channel in present_channels:
        if supported[channel] is None:
            raise ValueError(
                f"constraint channel {channel!r} has no matching embedder weights"
            )

    z_constraint = None
    if "pocket" in constraint_feature_dict:
        assert params.pocket_z is not None
        z_constraint = linear(constraint_feature_dict["pocket"], params.pocket_z)
    if "contact" in constraint_feature_dict:
        assert params.contact_z is not None
        z_contact = linear(constraint_feature_dict["contact"], params.contact_z)
        z_constraint = z_contact if z_constraint is None else z_constraint + z_contact
    if "contact_atom" in constraint_feature_dict:
        assert params.contact_atom_z is not None
        z_contact_atom = linear(
            constraint_feature_dict["contact_atom"],
            params.contact_atom_z,
        )
        z_constraint = (
            z_contact_atom if z_constraint is None else z_constraint + z_contact_atom
        )
    if "substructure" in constraint_feature_dict:
        assert params.substructure_z is not None
        z_substructure = substructure_mlp(
            constraint_feature_dict["substructure"],
            params.substructure_z,
        )
        z_constraint = (
            z_substructure if z_constraint is None else z_constraint + z_substructure
        )
    return z_constraint
