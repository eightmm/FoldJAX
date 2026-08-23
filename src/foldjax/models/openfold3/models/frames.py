"""Predicted-geometry token frames used by pTM and ipTM."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp


def _encoded_atom_name(name: str) -> jnp.ndarray:
    """Encode a right-padded four-character atom name like the features do."""
    return jnp.asarray([ord(character) - 32 for character in name.ljust(4)])


def _named_atom_indices(
    batch: Mapping[str, jnp.ndarray], name: str, *, n_atom: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return an absolute atom index and presence mask for every token."""
    starts = batch["start_atom_index"].astype(jnp.int32)
    counts = batch["num_atoms_per_token"].astype(jnp.int32)
    slots = jnp.arange(23, dtype=jnp.int32)
    absolute = starts[..., :, None] + slots
    safe = jnp.clip(absolute, 0, n_atom - 1)

    # ``ref_atom_name_chars`` is one-hot over the printable-ASCII offset used
    # by upstream. Gather each token's at-most-23 atoms, then compare all four
    # characters; this derives the frame table from the archive itself and keeps
    # inference independent of upstream's chemistry modules.
    codes = jnp.argmax(batch["ref_atom_name_chars"], axis=-1)
    expanded = jnp.broadcast_to(
        codes[..., None, :, :], (*codes.shape[:-2], starts.shape[-1], n_atom, 4)
    )
    gathered = jnp.take_along_axis(expanded, safe[..., None], axis=-2)
    valid_slot = (
        (slots < counts[..., :, None])
        & (absolute < n_atom)
        & batch["token_mask"].astype(bool)[..., :, None]
    )
    matches = jnp.all(gathered == _encoded_atom_name(name), axis=-1) & valid_slot
    offset = jnp.argmax(matches, axis=-1).astype(jnp.int32)
    present = jnp.any(matches, axis=-1)
    return starts + offset, present


def token_frame_atoms(
    batch: Mapping[str, jnp.ndarray],
    coordinates: jnp.ndarray,
    atom_mask: jnp.ndarray,
    *,
    angle_threshold: float = 25.0,
    eps: float = 1e-8,
    has_atomized_tokens: bool = True,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Select frame atoms and their validity from predicted coordinates.

    This is the JAX equivalent of upstream ``get_token_frame_atoms``: standard
    proteins use ``(N, CA, C)``, standard nucleotides use
    ``(C3', C1', C4')``, and atomized ligand/modified-residue tokens use their
    first atom plus its two closest atoms in the same chain. The last case is
    geometry-dependent, so treating every real token as frame-valid changes
    pTM/ipTM for complexes containing chemistry outside the standard polymers.

    The distance calculation is ``[sample, token, atom]`` instead of upstream's
    full ``[sample, atom, atom]`` matrix; only rows for token start atoms are ever
    consumed, so the selected neighbors are identical with much lower memory.
    Production entry points derive ``has_atomized_tokens`` from masked host
    features. False makes the expensive neighbour-search graph disappear
    entirely; the conservative True default preserves generic direct callers.
    """
    n_atom = coordinates.shape[-2]
    token_mask = batch["token_mask"].astype(bool)
    atomized = batch["is_atomized"].astype(bool) & token_mask
    protein = batch["is_protein"].astype(bool) & ~atomized & token_mask
    nucleotide = (
        (batch["is_dna"].astype(bool) | batch["is_rna"].astype(bool))
        & ~atomized
        & token_mask
    )

    n_index, n_present = _named_atom_indices(batch, "N", n_atom=n_atom)
    ca_index, ca_present = _named_atom_indices(batch, "CA", n_atom=n_atom)
    c_index_standard, c_present = _named_atom_indices(batch, "C", n_atom=n_atom)
    c3_index, c3_present = _named_atom_indices(batch, "C3'", n_atom=n_atom)
    c1_index, c1_present = _named_atom_indices(batch, "C1'", n_atom=n_atom)
    c4_index, c4_present = _named_atom_indices(batch, "C4'", n_atom=n_atom)

    atom_owner = batch["atom_to_token_index"].astype(jnp.int32)
    safe_owner = jnp.clip(atom_owner, 0, token_mask.shape[-1] - 1)
    atom_chain = jnp.take_along_axis(batch["asym_id"], safe_owner, axis=-1)

    # This must be a Python branch. A dynamic ``lax.cond`` still leaves XLA's
    # arena sized for the expensive branch, defeating the point of proving on
    # the host that every active token is a standard polymer token.
    if has_atomized_tokens:
        start = batch["start_atom_index"].astype(jnp.int32)
        safe_start = jnp.clip(start, 0, n_atom - 1)
        start_position = jnp.take_along_axis(
            coordinates, safe_start[..., None], axis=-2
        )
        same_chain = batch["asym_id"][..., :, None] == atom_chain[..., None, :]
        candidate = atom_mask.astype(bool)[..., None, :] & same_chain
        square_distance = jnp.sum(
            (start_position[..., :, None, :] - coordinates[..., None, :, :]) ** 2,
            axis=-1,
        )
        distance = jnp.where(candidate, jnp.sqrt(square_distance + eps), jnp.inf)
        # The first result is the start atom itself; the next two are the frame's
        # neighbors. ``top_k`` on the negated distances is a bounded partial sort.
        # ``lax.top_k`` requires ``k <= n_atom``. Tiny ligands can contain only one
        # or two atoms, so pad the candidate axis rather than failing before their
        # (correctly invalid) frame mask can be returned. Dummy candidates remain
        # at infinity and are explicitly marked absent below.
        pad_atoms = max(0, 3 - n_atom)
        padded_distance = jnp.pad(
            distance,
            [(0, 0)] * (distance.ndim - 1) + [(0, pad_atoms)],
            constant_values=jnp.inf,
        )
        closest_values, closest = jax.lax.top_k(-padded_distance, 3)
        atomized_a = closest[..., 1]
        atomized_b = start
        atomized_c = closest[..., 2]
        atomized_a_present = jnp.isfinite(-closest_values[..., 1]) & (
            atomized_a < n_atom
        )
        atomized_c_present = jnp.isfinite(-closest_values[..., 2]) & (
            atomized_c < n_atom
        )
    else:
        # These values are never selected because the host-side specialization
        # proved ``atomized`` false for every active token. Shape-compatible
        # placeholders keep the common standard-polymer gather below simple.
        atomized_a = n_index
        atomized_b = ca_index
        atomized_c = c_index_standard
        atomized_a_present = jnp.zeros_like(atomized)
        atomized_c_present = jnp.zeros_like(atomized)

    a_index = jnp.where(atomized, atomized_a, jnp.where(protein, n_index, c3_index))
    b_index = jnp.where(atomized, atomized_b, jnp.where(protein, ca_index, c1_index))
    c_index = jnp.where(
        atomized, atomized_c, jnp.where(protein, c_index_standard, c4_index)
    )
    a_present = jnp.where(
        atomized,
        atomized_a_present,
        jnp.where(protein, n_present, jnp.where(nucleotide, c3_present, False)),
    )
    b_present = jnp.where(
        atomized,
        True,
        jnp.where(protein, ca_present, jnp.where(nucleotide, c1_present, False)),
    )
    c_present_selected = jnp.where(
        atomized,
        atomized_c_present,
        jnp.where(protein, c_present, jnp.where(nucleotide, c4_present, False)),
    )

    def gather(index: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        safe = jnp.clip(index, 0, n_atom - 1)
        position = jnp.take_along_axis(coordinates, safe[..., None], axis=-2)
        present = jnp.take_along_axis(atom_mask, safe, axis=-1).astype(bool)
        chain = jnp.take_along_axis(atom_chain, safe, axis=-1)
        return position, present, chain

    a, a_mask, a_chain = gather(a_index)
    b, b_mask, b_chain = gather(b_index)
    c, c_mask, c_chain = gather(c_index)

    if has_atomized_tokens:
        u = a - b
        v = c - b
        cosine = jnp.sum(u * v, axis=-1) / jnp.sqrt(
            (jnp.sum(u**2, axis=-1) + eps) * (jnp.sum(v**2, axis=-1) + eps)
        )
        radians = jnp.deg2rad(angle_threshold)
        valid_angle = (cosine < jnp.cos(radians)) & (cosine > jnp.cos(jnp.pi - radians))
        valid_angle = jnp.where(atomized, valid_angle, True)
    else:
        # ``coordinates`` may carry a diffusion-sample axis while host features
        # retain batch size one. The gathered frame already has the exact
        # broadcast leading shape the generic angle calculation would produce.
        valid_angle = jnp.ones_like(a[..., 0], dtype=bool)
    same_selected_chain = (a_chain == b_chain) & (b_chain == c_chain)
    valid = (
        token_mask
        & a_present
        & b_present
        & c_present_selected
        & a_mask
        & b_mask
        & c_mask
        & same_selected_chain
        & valid_angle
    )
    return (a, b, c), valid
