"""Pure JAX valid-frame construction matching Chai-1 ranking."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def get_centre_positions_and_mask(
    atom_coords: jnp.ndarray,
    atom_exists_mask: jnp.ndarray,
    token_centre_atom_index: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gather each token's centre atom and reapply the token padding mask."""

    center_index = token_centre_atom_index.astype(jnp.int32)
    center_pos = jnp.take_along_axis(
        atom_coords,
        jnp.broadcast_to(center_index[..., None], center_index.shape + (3,)),
        axis=-2,
    )
    center_mask = jnp.take_along_axis(atom_exists_mask, center_index, axis=-1)
    return center_pos, center_mask & token_exists_mask


def abc_is_colinear(
    atoms_a: jnp.ndarray,
    atoms_b: jnp.ndarray,
    atoms_c: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Chai's 25-degree frame degeneracy threshold."""

    w1 = atoms_a - atoms_b
    w1 = w1 / jnp.linalg.norm(w1, axis=-1, keepdims=True)
    w2 = atoms_c - atoms_b
    w2 = w2 / jnp.linalg.norm(w2, axis=-1, keepdims=True)
    cosine = jnp.clip(jnp.sum(w1 * w2, axis=-1), -1.0, 1.0)
    angle = jnp.arccos(cosine)
    lower = 25.0 / 180.0 * jnp.pi
    upper = 155.0 / 180.0 * jnp.pi
    return jnp.isnan(angle) | (angle < lower) | (angle > upper)


def _gather_tokens(values: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(values, indices, axis=-1)


def get_single_atom_frames(
    atom_coords: jnp.ndarray,
    token_asym_id: jnp.ndarray,
    token_residue_index: jnp.ndarray,
    token_backbone_frame_mask: jnp.ndarray,
    token_centre_atom_index: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
    atom_exists_mask: jnp.ndarray,
    atom_token_index: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct frames for atom-tokenized residues such as ligands."""

    centre_coords, centre_mask = get_centre_positions_and_mask(
        atom_coords,
        atom_exists_mask,
        token_centre_atom_index,
        token_exists_mask,
    )
    same_asym = token_asym_id[..., :, None] == token_asym_id[..., None, :]
    same_residue = (
        token_residue_index[..., :, None] == token_residue_index[..., None, :]
    )
    distances = jnp.linalg.norm(
        centre_coords[..., :, None, :] - centre_coords[..., None, :, :],
        axis=-1,
    )
    valid_pair = centre_mask[..., :, None] & centre_mask[..., None, :]
    distances = jnp.where(valid_pair & same_asym & same_residue, distances, jnp.inf)
    tokens = distances.shape[-1]
    distances = jnp.where(jnp.eye(tokens, dtype=bool), jnp.inf, distances)
    nearest = jnp.argsort(distances, axis=-1)[..., :2]
    a = nearest[..., 0]
    c = nearest[..., 1]
    b = jnp.broadcast_to(jnp.arange(tokens), a.shape)

    center_a = _gather_tokens(token_centre_atom_index, a)
    center_b = _gather_tokens(token_centre_atom_index, b)
    center_c = _gather_tokens(token_centre_atom_index, c)
    abc_atom_indices = jnp.stack((center_a, center_b, center_c), axis=-1)

    mask_a = _gather_tokens(centre_mask, a)
    mask_b = _gather_tokens(centre_mask, b)
    mask_c = _gather_tokens(centre_mask, c)
    abc_coords_mask = mask_a & mask_b & mask_c

    residue_a = _gather_tokens(token_residue_index, a)
    residue_b = _gather_tokens(token_residue_index, b)
    residue_c = _gather_tokens(token_residue_index, c)
    frame_same_residue = (residue_a == residue_b) & (residue_b == residue_c)
    asym_a = _gather_tokens(token_asym_id, a)
    asym_b = _gather_tokens(token_asym_id, b)
    asym_c = _gather_tokens(token_asym_id, c)
    frame_same_chain = (asym_a == asym_b) & (asym_b == asym_c)

    batch_indices = jnp.arange(atom_coords.shape[0])[:, None]
    colinear = abc_is_colinear(
        centre_coords[batch_indices, a],
        centre_coords[batch_indices, b],
        centre_coords[batch_indices, c],
    )

    atom_counts = jax.nn.one_hot(
        atom_token_index.astype(jnp.int32), tokens, dtype=jnp.int32
    ).sum(axis=-2)
    # Upstream only invalidates token IDs present in atom_token_index with a
    # count other than one; absent token IDs retain the initialized True value.
    is_single_atom_token = atom_counts <= 1
    valid_frame = (
        is_single_atom_token
        & ~token_backbone_frame_mask
        & frame_same_residue
        & frame_same_chain
        & ~colinear
        & abc_coords_mask
        & token_exists_mask
    )
    return abc_atom_indices, valid_frame


def get_frames_and_mask(
    atom_coords: jnp.ndarray,
    token_asym_id: jnp.ndarray,
    token_residue_index: jnp.ndarray,
    token_backbone_frame_mask: jnp.ndarray,
    token_centre_atom_index: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
    atom_exists_mask: jnp.ndarray,
    backbone_frame_idces: jnp.ndarray,
    atom_token_index: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Merge generated single-atom frames with supplied backbone frames."""

    single_indices, single_mask = get_single_atom_frames(
        atom_coords,
        token_asym_id,
        token_residue_index,
        token_backbone_frame_mask,
        token_centre_atom_index,
        token_exists_mask,
        atom_exists_mask,
        atom_token_index,
    )
    frame_indices = jnp.where(
        single_mask[..., None], single_indices, backbone_frame_idces
    )
    return frame_indices, single_mask | token_backbone_frame_mask
