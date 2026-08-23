"""Pure JAX Boltz-2 ConfidenceModule forward.

Mirrors ``boltz.model.modules.confidencev2.ConfidenceModule`` /
``ConfidenceHeads`` for the ``boltz2_conf.ckpt`` configuration:
``add_s_to_z_prod``, ``add_s_input_to_s``, ``add_z_input_to_z``,
``bond_type_feature``, ``no_update_s=False``, ``token_level_confidence=True``,
``use_separate_heads=True``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_mesh, shard_pair_rows
from foldjax.models._cp_atom import replicate_atoms
from foldjax.models.boltz2.models.diffusion.atom import (
    gather_rep_atoms_to_tokens,
    gather_tokens_to_atoms,
)
from foldjax.models.boltz2.models.trunk_blocks.pairformer import (
    pairformer_module_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    contact_conditioning_forward,
    relative_position_forward,
)

Params = Mapping[str, Any]

# const.chain_type_ids
_NONPOLYMER = 3
_PROTEIN = 0


def _linear(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    return x @ kernel


def _layer_norm(
    x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float
) -> jnp.ndarray:
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * scale + bias


def _cdist(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)
    b2 = jnp.sum(b * b, axis=-1, keepdims=True)
    d2 = a2 + jnp.swapaxes(b2, -1, -2) - 2.0 * (a @ jnp.swapaxes(b, -1, -2))
    return jnp.sqrt(jnp.maximum(d2, 0.0))


def compute_aggregated_metric(logits: jnp.ndarray, end: float = 1.0) -> jnp.ndarray:
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = jnp.arange(0.5 * bin_width, end, bin_width)
    probs = jax.nn.softmax(logits, axis=-1)
    shape = (1,) * (probs.ndim - 1) + bounds.shape
    return jnp.sum(probs * bounds.reshape(shape), axis=-1)


def _tm_function(d: jnp.ndarray, n_res: jnp.ndarray) -> jnp.ndarray:
    d0 = 1.24 * (jnp.clip(n_res, 19, None) - 15) ** (1 / 3) - 1.8
    return 1.0 / (1.0 + (d / d0) ** 2)


def confidence_module_forward(
    params: Params,
    s_inputs: jnp.ndarray,  # b n ts
    s: jnp.ndarray,  # b n ts
    z: jnp.ndarray,  # b n n tz
    x_pred: jnp.ndarray,  # bm m 3
    feats: Mapping[str, jnp.ndarray],
    pred_distogram_logits: jnp.ndarray,
    *,
    multiplicity: int = 1,
    # Match the Boltz-2 conf checkpoint config (fix_sym_check / cyclic_pos_enc
    # are True in the trained hparams; see boltz2_trunk_forward).
    cyclic_pos_enc: bool = True,
    fix_sym_check: bool = True,
    eps: float = 1e-5,
    use_scan: bool = True,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    attention_backend: str = "xla",
    triangle_backend: str = "cueq",
    glu_backend: str = "xla",
    return_pair_chains_iptm: bool = True,
    recompute_nonpolymer_frames: bool = True,
    atom_context_parallel: bool = False,
) -> dict[str, Any]:
    """Run ConfidenceModule (real boltz2_conf config) in eval mode."""

    # Confidence frame construction currently consumes a global linear
    # atom stream. Make that O(A) gather explicit at the stage boundary; the
    # quadratic pair state remains context-parallel throughout the head.
    if atom_context_parallel and cp_mesh() is not None:
        x_pred = replicate_atoms(x_pred)

    s_inputs = _layer_norm(
        s_inputs,
        params["s_inputs_norm"]["scale"],
        params["s_inputs_norm"]["bias"],
        eps,
    )
    # no_update_s=False
    s = _layer_norm(s, params["s_norm"]["scale"], params["s_norm"]["bias"], eps)

    # add_s_input_to_s=True
    s = s + _linear(s_inputs, params["s_input_to_s"]["kernel"])

    z = _layer_norm(z, params["z_norm"]["scale"], params["z_norm"]["bias"], eps)

    # add_z_input_to_z=True
    z = z + relative_position_forward(
        params["rel_pos"],
        feats,
        cyclic_pos_enc=cyclic_pos_enc,
        fix_sym_check=fix_sym_check,
    )
    z = z + _linear(
        feats["token_bonds"].astype(jnp.float32), params["token_bonds"]["kernel"]
    )
    # bond_type_feature=True
    z = z + params["token_bonds_type"][feats["type_bonds"].astype(jnp.int32)]
    z = z + contact_conditioning_forward(params["contact_conditioning"], feats)

    s = jnp.repeat(s, multiplicity, axis=0)

    z = (
        z
        + _linear(s_inputs, params["s_to_z"]["kernel"])[:, :, None, :]
        + _linear(s_inputs, params["s_to_z_transpose"]["kernel"])[:, None, :, :]
    )
    # add_s_to_z_prod=True
    z = z + _linear(
        _linear(s_inputs, params["s_to_z_prod_in1"]["kernel"])[:, :, None, :]
        * _linear(s_inputs, params["s_to_z_prod_in2"]["kernel"])[:, None, :, :],
        params["s_to_z_prod_out"]["kernel"],
    )

    z = jnp.repeat(z, multiplicity, axis=0)
    s_inputs = jnp.repeat(s_inputs, multiplicity, axis=0)

    token_to_rep_atom = jnp.repeat(
        feats["token_to_rep_atom"].astype(jnp.float32), multiplicity, axis=0
    )
    if x_pred.ndim == 4:
        b, mult, n, _ = x_pred.shape
        x_pred = x_pred.reshape(b * mult, n, -1)
    x_pred_repr = gather_rep_atoms_to_tokens(token_to_rep_atom, x_pred)
    d = _cdist(x_pred_repr, x_pred_repr)
    boundaries = params["boundaries"]
    distogram = jnp.sum(d[..., None] > boundaries, axis=-1).astype(jnp.int32)
    distogram = params["dist_bin_pairwise_embed"][distogram]
    # Born sharded under context parallelism: the head's z assembly above
    # (relpos, bonds, outer sums, distogram) otherwise materializes whole on
    # every device before the stack's own seams take over.
    z = shard_pair_rows(z + distogram)

    mask = jnp.repeat(feats["token_pad_mask"].astype(jnp.float32), multiplicity, axis=0)
    pair_mask = mask[:, :, None] * mask[:, None, :]

    s, z = pairformer_module_forward(
        params["pairformer_stack"],
        s,
        z,
        mask=mask,
        pair_mask=pair_mask,
        eps=eps,
        use_scan=use_scan,
        chunk_size=chunk_size,
        triangle_attention_chunk=triangle_attention_chunk,
        triangle_attention_q_chunk=triangle_attention_q_chunk,
        transition_hidden_chunk=transition_hidden_chunk,
        matmul_precision=matmul_precision,
        attention_backend=attention_backend,
        triangle_backend=triangle_backend,
        glu_backend=glu_backend,
    )

    return _confidence_heads_forward(
        params["confidence_heads"],
        s=s,
        z=z,
        x_pred=x_pred,
        d=d,
        feats=feats,
        pred_distogram_logits=pred_distogram_logits,
        multiplicity=multiplicity,
        return_pair_chains_iptm=return_pair_chains_iptm,
        recompute_nonpolymer_frames=recompute_nonpolymer_frames,
    )


def _confidence_heads_forward(
    params: Params,
    *,
    s: jnp.ndarray,
    z: jnp.ndarray,
    x_pred: jnp.ndarray,
    d: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    pred_distogram_logits: jnp.ndarray,
    multiplicity: int,
    return_pair_chains_iptm: bool,
    recompute_nonpolymer_frames: bool,
) -> dict[str, Any]:
    # use_separate_heads=True
    asym_id_token = feats["asym_id"]
    is_same_chain = asym_id_token[..., :, None] == asym_id_token[..., None, :]
    is_different_chain_bool = ~is_same_chain

    pae_intra = (
        _linear(z, params["to_pae_intra_logits"]["kernel"])
        * is_same_chain.astype(jnp.float32)[..., None]
    )
    pae_inter = (
        _linear(z, params["to_pae_inter_logits"]["kernel"])
        * is_different_chain_bool.astype(jnp.float32)[..., None]
    )
    pae_logits = pae_inter + pae_intra

    z_sym = z + jnp.swapaxes(z, 1, 2)
    pde_intra = (
        _linear(z_sym, params["to_pde_intra_logits"]["kernel"])
        * is_same_chain.astype(jnp.float32)[..., None]
    )
    pde_inter = (
        _linear(z_sym, params["to_pde_inter_logits"]["kernel"])
        * is_different_chain_bool.astype(jnp.float32)[..., None]
    )
    pde_logits = pde_inter + pde_intra

    resolved_logits = _linear(s, params["to_resolved_logits"]["kernel"])
    plddt_logits = _linear(s, params["to_plddt_logits"]["kernel"])

    ligand_weight = 20.0
    non_interface_weight = 1.0
    interface_weight = 10.0

    token_type = jnp.repeat(feats["mol_type"], multiplicity, axis=0)
    is_ligand_token = (token_type == _NONPOLYMER).astype(jnp.float32)

    # token_level_confidence=True
    plddt = compute_aggregated_metric(plddt_logits)
    token_pad_mask = jnp.repeat(
        feats["token_pad_mask"].astype(jnp.float32), multiplicity, axis=0
    )
    complex_plddt = (plddt * token_pad_mask).sum(axis=-1) / token_pad_mask.sum(axis=-1)

    is_contact = (d < 8).astype(jnp.float32)
    is_different_chain = jnp.repeat(
        (asym_id_token[..., :, None] != asym_id_token[..., None, :]).astype(
            jnp.float32
        ),
        multiplicity,
        axis=0,
    )
    token_interface_mask = jnp.max(
        is_contact * is_different_chain * (1 - is_ligand_token)[..., None], axis=-1
    )
    token_non_interface_mask = (1 - token_interface_mask) * (1 - is_ligand_token)
    iplddt_weight = (
        is_ligand_token * ligand_weight
        + token_interface_mask * interface_weight
        + token_non_interface_mask * non_interface_weight
    )
    complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(axis=-1) / jnp.sum(
        token_pad_mask * iplddt_weight, axis=-1
    )

    pde = compute_aggregated_metric(pde_logits, end=32)
    pred_distogram_prob = jnp.repeat(
        jax.nn.softmax(pred_distogram_logits, axis=-1), multiplicity, axis=0
    )
    contacts = jnp.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype)
    contacts = contacts.at[:, :, :, :20].set(1.0)
    prob_contact = (pred_distogram_prob * contacts).sum(-1)
    n_tok = token_pad_mask.shape[1]
    token_pad_pair_mask = (
        token_pad_mask[:, :, None]
        * token_pad_mask[:, None, :]
        * (1 - jnp.eye(n_tok)[None])
    )
    token_pair_mask = token_pad_pair_mask * prob_contact
    complex_pde = (pde * token_pair_mask).sum(axis=(1, 2)) / token_pair_mask.sum(
        axis=(1, 2)
    )
    asym_id = jnp.repeat(asym_id_token, multiplicity, axis=0)
    token_interface_pair_mask = token_pair_mask * (
        asym_id[:, :, None] != asym_id[:, None, :]
    )
    complex_ipde = (pde * token_interface_pair_mask).sum(axis=(1, 2)) / (
        token_interface_pair_mask.sum(axis=(1, 2)) + 1e-5
    )

    out_dict: dict[str, Any] = dict(
        pde_logits=pde_logits,
        plddt_logits=plddt_logits,
        resolved_logits=resolved_logits,
        pde=pde,
        plddt=plddt,
        complex_plddt=complex_plddt,
        complex_iplddt=complex_iplddt,
        complex_pde=complex_pde,
        complex_ipde=complex_ipde,
    )
    out_dict["pae_logits"] = pae_logits
    out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)

    ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = _compute_ptms(
        pae_logits,
        x_pred,
        feats,
        multiplicity,
        return_pair_chains_iptm=return_pair_chains_iptm,
        recompute_nonpolymer_frames=recompute_nonpolymer_frames,
    )
    out_dict["ptm"] = ptm
    out_dict["iptm"] = iptm
    out_dict["ligand_iptm"] = ligand_iptm
    out_dict["protein_iptm"] = protein_iptm
    out_dict["pair_chains_iptm"] = pair_chains_iptm
    return out_dict


def _compute_collinear_mask(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    norm1 = jnp.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = jnp.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = jnp.abs(jnp.sum(v1 * v2, axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def _nearest_nonpolymer_frames(
    pred_atom_coords: jnp.ndarray,
    asym_id_atom: jnp.ndarray,
    atom_pad_mask: jnp.ndarray,
    token_to_rep_atom: jnp.ndarray,
) -> jnp.ndarray:
    """Find each token representative's three nearest same-chain atoms."""

    rep_atom_idx = jnp.argmax(token_to_rep_atom, axis=-1)
    rep_coord_idx = jnp.broadcast_to(
        rep_atom_idx[:, None, :, None],
        (
            pred_atom_coords.shape[0],
            pred_atom_coords.shape[1],
            rep_atom_idx.shape[1],
            pred_atom_coords.shape[-1],
        ),
    )
    rep_atom_coords = jnp.take_along_axis(pred_atom_coords, rep_coord_idx, axis=2)
    rep_atom_pad_mask = jnp.take_along_axis(atom_pad_mask, rep_atom_idx, axis=1)
    rep_asym_id_atom = jnp.take_along_axis(asym_id_atom, rep_atom_idx, axis=1)

    # Only representative-atom rows are consumed by confidence frames. Retain
    # the historical full atom-axis sort for each such row, including its tie
    # order, while avoiding the unused atom-by-atom distance matrix.
    delta = rep_atom_coords[:, :, :, None, :] - pred_atom_coords[:, :, None, :, :]
    dist_mat = jnp.sqrt(jnp.maximum(jnp.sum(delta * delta, axis=-1), 0.0))
    same_chain = rep_asym_id_atom[:, :, None] == asym_id_atom[:, None, :]
    valid_atom_pair = (
        same_chain
        & rep_atom_pad_mask[:, :, None].astype(bool)
        & atom_pad_mask[:, None, :].astype(bool)
    )
    dist_mat = jnp.where(valid_atom_pair[:, None, :, :], dist_mat, jnp.inf)
    nearest = jnp.argsort(dist_mat, axis=-1)[..., :3]
    return nearest[..., jnp.array([1, 0, 2])]


def _compute_frame_pred_inference(
    pred_atom_coords: jnp.ndarray,
    frames_idx_true: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    multiplicity: int,
    recompute_nonpolymer_frames: bool = True,
) -> jnp.ndarray:
    """JAX port of compute_frame_pred (inference=True).

    Non-polymer frames are derived with fixed-shape gathers so this path remains
    compatible with ``jax.jit``. Each non-polymer token uses its representative
    atom and the three nearest valid atoms in the same chain, matching the
    reference's one-token-per-ligand-atom inference representation.
    """

    asym_id_token = feats["asym_id"]
    asym_id_atom = gather_tokens_to_atoms(
        feats["atom_to_token"].astype(jnp.float32),
        asym_id_token[..., None].astype(jnp.float32),
    )[..., 0]

    b, n, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(b // multiplicity, multiplicity, -1, 3)
    frames_idx_pred = jnp.reshape(
        jnp.repeat(frames_idx_true, multiplicity, axis=0),
        (b // multiplicity, multiplicity, -1, 3),
    )

    mol_type = feats["mol_type"]
    token_pad_mask = feats["token_pad_mask"]
    atom_pad_mask = feats["atom_pad_mask"]

    if recompute_nonpolymer_frames:
        nonpolymer_frames = _nearest_nonpolymer_frames(
            pred_atom_coords,
            asym_id_atom,
            atom_pad_mask,
            feats["token_to_rep_atom"],
        )
        chain_atom_count = jnp.sum(
            (asym_id_token[:, :, None] == asym_id_atom[:, None, :])
            * atom_pad_mask[:, None, :],
            axis=-1,
        )
        same_chain_token = (
            asym_id_token[:, :, None] == asym_id_token[:, None, :]
        ) & token_pad_mask[:, None, :].astype(bool)
        first_chain_token = jnp.argmax(same_chain_token, axis=-1)
        first_chain_mol_type = jnp.take_along_axis(mol_type, first_chain_token, axis=1)
        use_nonpolymer_frame = (
            (first_chain_mol_type == _NONPOLYMER)
            & (chain_atom_count >= 3)
            & token_pad_mask.astype(bool)
        )
        frames_idx_pred = jnp.where(
            use_nonpolymer_frame[:, None, :, None],
            nonpolymer_frames,
            frames_idx_pred,
        )

    bm = b // multiplicity
    idx_b = jnp.arange(bm)[:, None, None, None]
    idx_m = jnp.arange(multiplicity)[None, :, None, None]
    frames_expanded = pred_atom_coords[idx_b, idx_m, frames_idx_pred].reshape(-1, 3, 3)
    mask_collinear = _compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(bm, multiplicity, -1)
    return mask_collinear.astype(jnp.float32) * token_pad_mask[:, None, :]


def _compute_ptms(
    logits: jnp.ndarray,
    x_preds: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    multiplicity: int,
    *,
    return_pair_chains_iptm: bool = True,
    recompute_nonpolymer_frames: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    mask_collinear_pred = _compute_frame_pred_inference(
        x_preds,
        feats["frames_idx"],
        feats,
        multiplicity,
        recompute_nonpolymer_frames=recompute_nonpolymer_frames,
    )
    mask_pad = jnp.repeat(
        feats["token_pad_mask"].astype(jnp.float32), multiplicity, axis=0
    )
    maski = mask_collinear_pred.reshape(-1, mask_collinear_pred.shape[-1])
    pair_mask_ptm = maski[:, :, None] * mask_pad[:, None, :] * mask_pad[:, :, None]
    asym_id = jnp.repeat(feats["asym_id"], multiplicity, axis=0)
    diff_chain = (asym_id[:, None, :] != asym_id[:, :, None]).astype(jnp.float32)
    pair_mask_iptm = (
        maski[:, :, None] * diff_chain * mask_pad[:, None, :] * mask_pad[:, :, None]
    )

    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    pae_value = jnp.arange(0.5 * bin_width, 32.0, bin_width)[None, :]
    n_res = mask_pad.sum(axis=-1, keepdims=True)
    tm_value = _tm_function(pae_value, n_res)[:, None, None, :]
    probs = jax.nn.softmax(logits, axis=-1)
    tm_expected_value = jnp.sum(probs * tm_value, axis=-1)

    def _agg(mask: jnp.ndarray) -> jnp.ndarray:
        return jnp.max(
            jnp.sum(tm_expected_value * mask, axis=-1)
            / (jnp.sum(mask, axis=-1) + 1e-5),
            axis=1,
        )

    ptm = _agg(pair_mask_ptm)
    iptm = _agg(pair_mask_iptm)

    token_type = jnp.repeat(feats["mol_type"], multiplicity, axis=0)
    is_ligand = (token_type == _NONPOLYMER).astype(jnp.float32)
    is_protein = (token_type == _PROTEIN).astype(jnp.float32)
    base = maski[:, :, None] * diff_chain * mask_pad[:, None, :] * mask_pad[:, :, None]
    ligand_mask = base * (
        is_ligand[:, :, None] * is_protein[:, None, :]
        + is_protein[:, :, None] * is_ligand[:, None, :]
    )
    protein_mask = base * (is_protein[:, :, None] * is_protein[:, None, :])
    ligand_iptm = _agg(ligand_mask)
    protein_iptm = _agg(protein_mask)

    chain_pair_iptm: dict[Any, dict[Any, jnp.ndarray]] = {}
    if return_pair_chains_iptm:
        # This legacy nested-dict output depends on runtime chain IDs and is
        # therefore intentionally excluded from fully-jitted serving graphs.
        asym_ids_list = jnp.unique(asym_id).tolist()
        for idx1 in asym_ids_list:
            chain_iptm: dict[Any, jnp.ndarray] = {}
            for idx2 in asym_ids_list:
                m = (
                    maski[:, :, None]
                    * (asym_id[:, None, :] == idx1)
                    * (asym_id[:, :, None] == idx2)
                    * mask_pad[:, None, :]
                    * mask_pad[:, :, None]
                )
                chain_iptm[idx2] = _agg(m)
            chain_pair_iptm[idx1] = chain_iptm

    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm
