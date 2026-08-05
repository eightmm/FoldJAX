"""Reference atom feature embedding (AF3 Algorithm 5, lines 1-6).

Builds the atom single conditioning ``cl`` from reference-conformer features, and
the atom pair conditioning ``plm`` from pairwise offsets within each
sequence-local block.

The ``vlm`` term is what keeps this chemically meaningful: pair features only
survive between atoms sharing a ``ref_space_uid``, i.e. atoms from the same
reference conformer. Offsets across two different conformers are meaningless, so
they are zeroed rather than embedded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.openfold3.models.atom_blocks import (
    pair_rep_to_blocks,
    single_rep_to_blocks,
)
from foldjax.models.openfold3.models.atomize import broadcast_token_feat_to_atoms
from foldjax.models.openfold3.models.diffusion_transformer import (
    AtomTransformerParams,
    atom_transformer,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)


class RefAtomFeatureEmbedderParams(NamedTuple):
    """Parameters for ``RefAtomFeatureEmbedder``; every projection is bias-free."""

    linear_ref_pos: LinearParams
    linear_ref_charge: LinearParams
    linear_ref_mask: LinearParams
    linear_ref_element: LinearParams
    linear_ref_atom_chars: LinearParams
    linear_ref_offset: LinearParams
    linear_inv_sq_dists: LinearParams
    linear_valid_mask: LinearParams


def ref_atom_feature_embedder(
    batch: Mapping[str, jnp.ndarray],
    params: RefAtomFeatureEmbedderParams,
    *,
    n_query: int,
    n_key: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Embed reference atom features into single and pair conditioning.

    Args:
        batch: feature mapping needing ``ref_pos`` ``[..., N_atom, 3]``,
            ``ref_charge`` ``[..., N_atom]``, ``ref_mask`` ``[..., N_atom]``,
            ``ref_element`` ``[..., N_atom, C_element]``,
            ``ref_atom_name_chars`` ``[..., N_atom, 4, C_chars]``,
            ``ref_space_uid`` ``[..., N_atom]`` and ``atom_mask``
            ``[..., N_atom]``.
        params: mapped parameters.
        n_query: query block height.
        n_key: key window width.

    Returns:
        ``(cl, plm)``: ``[..., N_atom, c_atom]`` atom single conditioning and
        ``[..., N_blocks, N_query, N_key, c_atom_pair]`` atom pair conditioning.
    """
    ref_pos = batch["ref_pos"]
    dtype = ref_pos.dtype

    cl = linear(ref_pos, params.linear_ref_pos)
    cl = cl + linear(
        jnp.arcsinh(batch["ref_charge"][..., None]), params.linear_ref_charge
    )
    cl = cl + linear(
        batch["ref_mask"][..., None].astype(dtype), params.linear_ref_mask
    )
    cl = cl + linear(batch["ref_element"].astype(dtype), params.linear_ref_element)
    # The 4 name characters are flattened into one feature axis.
    chars = batch["ref_atom_name_chars"]
    chars = chars.reshape(chars.shape[:-2] + (chars.shape[-2] * chars.shape[-1],))
    cl = cl + linear(chars.astype(dtype), params.linear_ref_atom_chars)

    atom_mask = batch["atom_mask"]
    d_l, d_m, block_mask = single_rep_to_blocks(
        ref_pos, atom_mask, n_query=n_query, n_key=n_key
    )
    v_l, v_m, _ = single_rep_to_blocks(
        batch["ref_space_uid"][..., None].astype(dtype),
        atom_mask,
        n_query=n_query,
        n_key=n_key,
    )

    # [..., N_blocks, N_query, N_key, 3] and [..., N_blocks, N_query, N_key, 1]
    dlm = (d_l[..., None, :] - d_m[..., None, :, :]) * block_mask[..., None]
    vlm = (v_l[..., None, :] == v_m[..., None, :, :]).astype(dlm.dtype)
    vlm = vlm * block_mask[..., None]

    plm = linear(dlm, params.linear_ref_offset) * vlm
    inv_sq_dists = 1.0 / (1.0 + jnp.sum(dlm**2, axis=-1, keepdims=True))
    plm = plm + linear(inv_sq_dists, params.linear_inv_sq_dists) * vlm
    return cl, plm + linear(vlm, params.linear_valid_mask) * vlm


class NoisyPositionEmbedderParams(NamedTuple):
    """Parameters for ``NoisyPositionEmbedder`` (AF3 Alg. 5, lines 8-12).

    Both layer norms are scale-only (``create_offset=False``) and all three
    projections are bias-free.
    """

    layer_norm_s: LayerNormParams
    linear_s: LinearParams
    layer_norm_z: LayerNormParams
    linear_z: LinearParams
    linear_r: LinearParams


def noisy_position_embedder(
    batch: Mapping[str, jnp.ndarray],
    cl: jnp.ndarray,
    plm: jnp.ndarray,
    si_trunk: jnp.ndarray,
    zij_trunk: jnp.ndarray,
    rl: jnp.ndarray,
    params: NoisyPositionEmbedderParams,
    *,
    n_query: int,
    n_key: int,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fold the trunk representations and noisy coordinates into atom conditioning.

    Args:
        batch: needs ``token_mask``, ``num_atoms_per_token``,
            ``atom_to_token_index`` and ``atom_mask``.
        cl: ``[..., N_atom, c_atom]`` atom single conditioning.
        plm: ``[..., N_blocks, N_query, N_key, c_atom_pair]`` atom pair
            conditioning.
        si_trunk: ``[..., N_token, c_s]`` trunk single representation.
        zij_trunk: ``[..., N_token, N_token, c_z]`` trunk pair representation.
        rl: ``[..., N_atom, 3]`` noisy atom positions.
        params: mapped parameters.
        n_query: query block height.
        n_key: key window width.
        eps: layer norm epsilon.

    Returns:
        ``(cl, plm, ql)``. ``ql`` is the atom single representation carrying the
        noisy coordinate projection; ``cl`` deliberately does not.
    """
    single = linear(
        layer_norm(si_trunk, params.layer_norm_s, eps=eps), params.linear_s
    )
    cl = cl + broadcast_token_feat_to_atoms(
        batch["token_mask"],
        batch["num_atoms_per_token"],
        single,
        n_atom=cl.shape[-2],
    )

    pair = linear(layer_norm(zij_trunk, params.layer_norm_z, eps=eps), params.linear_z)
    plm = plm + pair_rep_to_blocks(
        pair,
        batch["atom_to_token_index"],
        batch["atom_mask"],
        n_query=n_query,
        n_key=n_key,
    )

    return cl, plm, cl + linear(rl, params.linear_r)


class AtomPairConditioningParams(NamedTuple):
    """Parameters for the atom pair conditioning stage of ``AtomAttentionEncoder``.

    ``pair_mlp`` is upstream's ``nn.Sequential(ReLU, Linear, ReLU, Linear, ReLU,
    Linear)``: the activation comes *first*, so the MLP sees a ReLU'd input rather
    than applying the activation between projections only.
    """

    linear_l: LinearParams
    linear_m: LinearParams
    pair_mlp_1: LinearParams
    pair_mlp_2: LinearParams
    pair_mlp_3: LinearParams


def relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU, spelled out so the pair-MLP ordering stays readable."""
    return jnp.maximum(x, 0.0)


def atom_pair_conditioning(
    cl: jnp.ndarray,
    plm: jnp.ndarray,
    atom_mask: jnp.ndarray,
    params: AtomPairConditioningParams,
    *,
    n_query: int,
    n_key: int,
) -> jnp.ndarray:
    """Fold the atom single conditioning into the atom pair conditioning.

    Args:
        cl: ``[..., N_atom, c_atom]`` atom single conditioning.
        plm: ``[..., N_blocks, N_query, N_key, c_atom_pair]`` pair conditioning.
        atom_mask: ``[..., N_atom]`` atom mask.
        params: mapped parameters.
        n_query: query block height.
        n_key: key window width.

    Returns:
        ``[..., N_blocks, N_query, N_key, c_atom_pair]`` updated conditioning.
    """
    cl_l, cl_m, block_mask = single_rep_to_blocks(
        cl, atom_mask, n_query=n_query, n_key=n_key
    )

    # Broadcast the query and key blockings against each other.
    cl_lm = (
        linear(relu(cl_l[..., :, None, :]), params.linear_l)
        + linear(relu(cl_m[..., None, :, :]), params.linear_m)
    ) * block_mask[..., None]

    plm = plm + cl_lm

    hidden = linear(relu(plm), params.pair_mlp_1)
    hidden = linear(relu(hidden), params.pair_mlp_2)
    hidden = linear(relu(hidden), params.pair_mlp_3)

    return (plm + hidden) * block_mask[..., None]


class AtomAttentionEncoderParams(NamedTuple):
    """Parameters for ``AtomAttentionEncoder`` (AF3 Algorithm 5).

    ``noisy_position_embedder`` is present only when the encoder was built with
    ``add_noisy_pos=True`` (the diffusion path). ``linear_q`` is upstream's
    ``Sequential(Linear, ReLU)``, so the ReLU is applied after the projection.
    """

    ref_atom_feature_embedder: RefAtomFeatureEmbedderParams
    pair_conditioning: AtomPairConditioningParams
    atom_transformer: AtomTransformerParams
    linear_q: LinearParams
    noisy_position_embedder: NoisyPositionEmbedderParams | None = None


def atom_attention_encoder(
    batch: Mapping[str, jnp.ndarray],
    params: AtomAttentionEncoderParams,
    *,
    n_query: int,
    n_key: int,
    no_heads: int,
    n_token: int,
    rl: jnp.ndarray | None = None,
    si_trunk: jnp.ndarray | None = None,
    zij_trunk: jnp.ndarray | None = None,
    inf: float = 1e9,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the atom attention encoder (AF3 Algorithm 5).

    Args:
        batch: feature mapping; see ``ref_atom_feature_embedder`` plus
            ``token_mask``, ``num_atoms_per_token`` and ``atom_to_token_index``
            when ``rl`` is given.
        params: mapped parameters.
        n_query: query block height.
        n_key: key window width.
        no_heads: atom transformer head count.
        n_token: static token count for the final aggregation.
        rl: ``[..., N_atom, 3]`` noisy positions; ``None`` on the input path.
        si_trunk: trunk single representation, required with ``rl``.
        zij_trunk: trunk pair representation, required with ``rl``.
        inf: masking constant.
        eps: layer norm epsilon.

    Returns:
        ``(ai, ql, cl, plm)``: token representation, atom single representation,
        atom single conditioning and blocked atom pair conditioning.
    """
    from foldjax.models.openfold3.models.atomize import aggregate_atom_feat_to_tokens

    atom_mask = batch["atom_mask"]
    cl, plm = ref_atom_feature_embedder(
        batch, params.ref_atom_feature_embedder, n_query=n_query, n_key=n_key
    )

    if rl is None:
        # Input path: the atom single representation starts as the conditioning.
        ql = cl
    else:
        if params.noisy_position_embedder is None:
            raise ValueError(
                "rl was supplied but this encoder has no noisy_position_embedder"
            )
        if si_trunk is None or zij_trunk is None:
            raise ValueError("si_trunk and zij_trunk are required alongside rl")
        cl, plm, ql = noisy_position_embedder(
            batch,
            cl,
            plm,
            si_trunk,
            zij_trunk,
            rl,
            params.noisy_position_embedder,
            n_query=n_query,
            n_key=n_key,
            eps=eps,
        )

    plm = atom_pair_conditioning(
        cl, plm, atom_mask, params.pair_conditioning, n_query=n_query, n_key=n_key
    )

    ql = atom_transformer(
        ql,
        cl,
        plm,
        params.atom_transformer,
        no_heads=no_heads,
        n_query=n_query,
        n_key=n_key,
        mask=atom_mask,
        inf=inf,
        eps=eps,
    )
    ql = ql * atom_mask[..., None]

    # linear_q is Sequential(Linear, ReLU): projection first, then activation.
    ai = aggregate_atom_feat_to_tokens(
        relu(linear(ql, params.linear_q)),
        batch["atom_to_token_index"],
        atom_mask,
        n_token=n_token,
        aggregate="mean",
    )
    return ai, ql, cl, plm


class AtomAttentionDecoderParams(NamedTuple):
    """Parameters for ``AtomAttentionDecoder`` (AF3 Algorithm 6).

    ``layer_norm`` is scale-only (``create_offset=False``); ``linear_q_out``
    projects to 3 coordinate components.
    """

    linear_q_in: LinearParams
    atom_transformer: AtomTransformerParams
    layer_norm: LayerNormParams
    linear_q_out: LinearParams


def atom_attention_decoder(
    batch: Mapping[str, jnp.ndarray],
    ai: jnp.ndarray,
    ql: jnp.ndarray,
    cl: jnp.ndarray,
    plm: jnp.ndarray,
    params: AtomAttentionDecoderParams,
    *,
    n_query: int,
    n_key: int,
    no_heads: int,
    inf: float = 1e9,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Decode token activations back to per-atom coordinate updates.

    Args:
        batch: needs ``token_mask``, ``num_atoms_per_token`` and ``atom_mask``.
        ai: ``[..., N_token, c_token]`` token representation.
        ql: ``[..., N_atom, c_atom]`` atom single representation.
        cl: ``[..., N_atom, c_atom]`` atom single conditioning.
        plm: ``[..., N_blocks, N_query, N_key, c_atom_pair]`` pair conditioning.
        params: mapped parameters.
        n_query: query block height.
        n_key: key window width.
        no_heads: atom transformer head count.
        inf: masking constant.
        eps: layer norm epsilon.

    Returns:
        ``[..., N_atom, 3]`` coordinate update.
    """
    ql = ql + broadcast_token_feat_to_atoms(
        batch["token_mask"],
        batch["num_atoms_per_token"],
        linear(ai, params.linear_q_in),
        n_atom=ql.shape[-2],
    )

    ql = atom_transformer(
        ql,
        cl,
        plm,
        params.atom_transformer,
        no_heads=no_heads,
        n_query=n_query,
        n_key=n_key,
        mask=batch["atom_mask"],
        inf=inf,
        eps=eps,
    )

    return linear(layer_norm(ql, params.layer_norm, eps=eps), params.linear_q_out)
