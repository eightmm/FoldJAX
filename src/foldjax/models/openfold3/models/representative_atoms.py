"""Per-token representative atoms (AF3's confidence-head geometry input).

The confidence heads embed pairwise distances between one representative atom per
token: CB for standard amino acids (CA for glycine), C4 for purines, C2 for
pyrimidines, and the single atom of anything tokenized per-atom.

Which atom that is depends on chemistry, not on the network, so the lookup
arrives as data in a :class:`RepresentativeAtomTable`. ``bridge.chemistry`` builds
one from upstream's own constants; keeping it out of this module is what lets the
inference path stay free of torch and of upstream's data stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp


class RepresentativeAtomTable(NamedTuple):
    """Chemistry lookups over the 32 one-hot residue-type columns.

    ``*_offset``/``*_mask`` give the within-residue index of an atom name and
    whether that residue has it at all. The indicator vectors select residue
    classes; glycine is indexed in the protein-only ordering upstream uses for it,
    which is not the same ordering as the nucleotide entries, so both come from
    the builder rather than being derived here.
    """

    cb_offset: jnp.ndarray
    cb_mask: jnp.ndarray
    ca_offset: jnp.ndarray
    ca_mask: jnp.ndarray
    c4_offset: jnp.ndarray
    c4_mask: jnp.ndarray
    c2_offset: jnp.ndarray
    c2_mask: jnp.ndarray
    glycine: jnp.ndarray
    purine_dna: jnp.ndarray
    purine_rna: jnp.ndarray
    pyrimidine_dna: jnp.ndarray
    pyrimidine_rna: jnp.ndarray


def token_representative_atoms(
    batch: Mapping[str, jnp.ndarray],
    x: jnp.ndarray,
    atom_mask: jnp.ndarray,
    table: RepresentativeAtomTable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gather one representative atom per token.

    Args:
        batch: needs ``restype``, ``start_atom_index``, ``token_mask``,
            ``is_protein``, ``is_dna``, ``is_rna`` and ``is_atomized``.
        x: ``[..., N_atom, 3]`` atom positions; may carry a leading sample axis.
        atom_mask: ``[..., N_atom]``.
        table: chemistry lookups.

    Returns:
        ``(rep_x, rep_mask)`` of shapes ``[..., N_token, 3]`` and
        ``[..., N_token]``.
    """
    restype = batch["restype"].astype(jnp.float32)

    def column(indicator: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...k,k->...", restype, indicator.astype(restype.dtype))

    atomized = batch["is_atomized"].astype(restype.dtype)
    standard = 1.0 - atomized
    is_protein = batch["is_protein"].astype(restype.dtype) * standard
    is_dna = batch["is_dna"].astype(restype.dtype) * standard
    is_rna = batch["is_rna"].astype(restype.dtype) * standard

    is_glycine = is_protein * column(table.glycine)
    is_purine = is_dna * column(table.purine_dna) + is_rna * column(table.purine_rna)
    is_pyrimidine = is_dna * column(table.pyrimidine_dna) + is_rna * column(
        table.pyrimidine_rna
    )
    not_glycine = is_protein * (1.0 - is_glycine)

    start = batch["start_atom_index"].astype(restype.dtype)
    index = (
        (start + column(table.cb_offset)) * not_glycine
        + (start + column(table.ca_offset)) * is_glycine
        + (start + column(table.c4_offset)) * is_purine
        + (start + column(table.c2_offset)) * is_pyrimidine
        + start * atomized
    )
    atom_present = (
        column(table.cb_mask) * not_glycine
        + column(table.ca_mask) * is_glycine
        + column(table.c4_mask) * is_purine
        + column(table.c2_mask) * is_pyrimidine
        + atomized
    )

    # Upstream gathers with an index broadcast against x's batch dims, so a
    # leading sample axis on x has to be honoured here too.
    gather_index = jnp.broadcast_to(
        index.astype(jnp.int32)[..., None], (*x.shape[:-2], index.shape[-1], 1)
    )
    rep_x = jnp.take_along_axis(x, gather_index, axis=-2)

    mask_index = jnp.broadcast_to(
        index.astype(jnp.int32), (*atom_mask.shape[:-1], index.shape[-1])
    )
    rep_mask = (
        jnp.take_along_axis(atom_mask, mask_index, axis=-1)
        * batch["token_mask"]
        * atom_present
    )
    return rep_x, rep_mask
