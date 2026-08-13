"""ESMFold2's feature dictionary for a whole job, in NumPy.

Upstream ships `prepare_protein_features`, which covers a single protein chain
with no alignment; everything else its `forward` accepts -- several chains, a
real MSA, symmetry -- is left to the caller. This is that assembly, written
against upstream's conventions rather than around them:

* residue types index a 33-entry vocabulary and are *not* one-hot -- the model
  one-hots them itself;
* atoms are each residue's heavy atoms in upstream's order, carrying its
  reference conformer coordinates, and the atom axis is padded to a multiple
  of 32;
* elements are atomic numbers and atom names are the four-character `ord - 32`
  encoding, neither of which is the AlphaFold-3 one-hot the other ports build;
* the MSA carries residue-type indices with gaps at `MSA_GAP_TOKEN_ID`.

Nothing here imports torch. The tables come from `chemistry`, which is
generated from upstream rather than transcribed, and
`tests/models/esmfold2/test_features_parity.py` checks the single-chain,
no-alignment case against `prepare_protein_features` tensor for tensor.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from foldjax.models.esmfold2.data import chemistry

#: The atom axis is padded to a multiple of this. Upstream's sliding-window
#: attention is happier on a round length, and the padding is masked anyway.
ATOM_BLOCK = 32


def build_features(
    chains: Sequence[tuple[str, str, int, int]],
    alignments: dict[int, Path] | None = None,
    *,
    msa_depth: int | None = None,
) -> dict[str, np.ndarray]:
    """Featurize a job, batch axis included.

    `chains` is one entry per chain copy: `(sequence, chain id, entity index,
    symmetry index)`. `alignments` maps an entity index to its a3m; a chain
    whose entity has none contributes only the query row.
    """
    alignments = alignments or {}

    res_type: list[int] = []
    input_ids: list[int] = []
    asym_id: list[int] = []
    sym_id: list[int] = []
    entity_id: list[int] = []
    residue_index: list[int] = []
    atom_records: list[tuple[int, str, str, int, tuple[float, float, float]]] = []
    distogram_atom_idx: list[int] = []
    # Where each chain's tokens start, so its alignment lands in its own
    # columns and nowhere else.
    chain_spans: list[tuple[int, int, int]] = []

    for chain_index, (sequence, _chain_id, entity, symmetry) in enumerate(chains):
        start = len(res_type)
        for position, letter in enumerate(sequence):
            residue = chemistry.PROTEIN_1TO3.get(letter, "UNK")
            names = chemistry.PROTEIN_HEAVY_ATOMS[residue]
            token = len(res_type)
            atom_start = len(atom_records)
            for name in names:
                atom_records.append(
                    (
                        token,
                        name,
                        name[0],
                        chemistry.PROTEIN_CHARGED_ATOMS.get((residue, name), 0),
                        chemistry.PROTEIN_REF_POS[residue][name],
                    )
                )
            representative = "CB" if "CB" in names else "CA"
            distogram_atom_idx.append(atom_start + names.index(representative))
            res_type.append(
                chemistry.PROTEIN_RESIDUE_TO_RES_TYPE.get(
                    residue, chemistry.PROTEIN_UNK_RES_TYPE
                )
            )
            input_ids.append(
                chemistry.ESM_PROTEIN_VOCAB.get(
                    letter, chemistry.ESM_PROTEIN_VOCAB["X"]
                )
            )
            # Chain identity, following upstream's single-chain values: asym_id
            # counts chain copies from zero, entity_id is one-based so that
            # zero stays available as padding, sym_id counts copies within an
            # entity -- which is what tells a homodimer from a heterodimer.
            asym_id.append(chain_index)
            sym_id.append(symmetry)
            entity_id.append(entity + 1)
            # Restarts per chain: the index is a position in its own polymer.
            residue_index.append(position)
        chain_spans.append((start, len(res_type), entity))

    n_token = len(res_type)
    if n_token == 0:
        raise ValueError("the job names no protein residues")
    n_atoms = -(-len(atom_records) // ATOM_BLOCK) * ATOM_BLOCK

    ref_pos = np.zeros((n_atoms, 3), dtype=np.float32)
    ref_element = np.zeros(n_atoms, dtype=np.int64)
    ref_charge = np.zeros(n_atoms, dtype=np.int8)
    ref_atom_name_chars = np.zeros((n_atoms, 4), dtype=np.int64)
    ref_space_uid = np.zeros(n_atoms, dtype=np.int64)
    atom_attention_mask = np.zeros(n_atoms, dtype=bool)
    atom_to_token = np.zeros(n_atoms, dtype=np.int64)
    for index, (token, name, element, charge, position) in enumerate(atom_records):
        ref_pos[index] = position
        ref_element[index] = chemistry.PROTEIN_ELEMENT_TO_ATOMIC_NUM[element]
        ref_charge[index] = charge
        ref_atom_name_chars[index] = chemistry.encode_atom_name(name)
        ref_space_uid[index] = token
        atom_attention_mask[index] = True
        atom_to_token[index] = token

    msa, msa_mask, has_deletion, deletion_value = build_msa(
        chain_spans, res_type, alignments, msa_depth=msa_depth
    )

    features = {
        "token_index": np.arange(n_token, dtype=np.int64),
        "residue_index": np.asarray(residue_index, dtype=np.int64),
        "asym_id": np.asarray(asym_id, dtype=np.int64),
        "sym_id": np.asarray(sym_id, dtype=np.int64),
        "entity_id": np.asarray(entity_id, dtype=np.int64),
        "mol_type": np.full(n_token, chemistry.MOL_TYPE_PROTEIN, dtype=np.int64),
        "res_type": np.asarray(res_type, dtype=np.int64),
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "token_bonds": np.zeros((n_token, n_token, 1), dtype=np.float32),
        "token_attention_mask": np.ones(n_token, dtype=bool),
        "ref_pos": ref_pos,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "atom_attention_mask": atom_attention_mask,
        "atom_to_token": atom_to_token,
        "distogram_atom_idx": np.asarray(distogram_atom_idx, dtype=np.int64),
        "msa": msa,
        "msa_attention_mask": msa_mask,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_value.mean(axis=0),
    }
    return {name: value[None] for name, value in features.items()}


def build_msa(
    chain_spans: Sequence[tuple[int, int, int]],
    res_type: Sequence[int],
    alignments: dict[int, Path],
    *,
    msa_depth: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble one alignment block per chain, gaps everywhere else.

    Row 0 is the query: every chain's own sequence, which is what the model
    reads when there is no alignment at all. Each further row belongs to one
    chain and is a gap over the others -- the unpaired arrangement, which is
    what a per-chain a3m can honestly support. Pairing rows across chains would
    claim the alignments were searched together, and they were not.
    """
    n_token = len(res_type)
    rows: list[list[int]] = [list(res_type)]
    deletions: list[list[float]] = [[0.0] * n_token]

    for start, end, entity in chain_spans:
        path = alignments.get(entity)
        if path is None:
            continue
        query = "".join(
            chemistry.RES_TYPE_TO_LETTER.get(value, "X")
            for value in res_type[start:end]
        )
        for sequence, counts in read_a3m(path, query=query):
            row = [chemistry.MSA_GAP_TOKEN_ID] * n_token
            deletion = [0.0] * n_token
            for offset, (letter, count) in enumerate(zip(sequence, counts)):
                residue = (
                    chemistry.PROTEIN_1TO3.get(letter, "UNK") if letter != "-" else None
                )
                row[start + offset] = (
                    chemistry.MSA_GAP_TOKEN_ID
                    if residue is None
                    else chemistry.PROTEIN_RESIDUE_TO_RES_TYPE.get(
                        residue, chemistry.PROTEIN_UNK_RES_TYPE
                    )
                )
                deletion[start + offset] = float(count)
            rows.append(row)
            deletions.append(deletion)
            if msa_depth is not None and len(rows) >= msa_depth:
                break

    msa = np.asarray(rows, dtype=np.int64)
    values = np.asarray(deletions, dtype=np.float32)
    return msa, np.ones(msa.shape, dtype=bool), values > 0, values


def read_a3m(path: Path, *, query: str) -> list[tuple[str, list[int]]]:
    """Aligned rows and their insertion counts, without the query row.

    a3m writes insertions as lowercase letters, which occupy no query column;
    each is counted against the column that follows it, which is what
    `deletion_value` means. A row whose aligned length is not the query's is
    dropped rather than padded -- it is not an alignment to this sequence --
    and a row identical to the query is dropped because row 0 already is it.
    Identity is the test rather than position: a search that does not repeat
    the query first would otherwise lose a real hit.
    """
    rows: list[tuple[str, list[int]]] = []
    for block in Path(path).read_text(encoding="utf-8").split(">")[1:]:
        if not block.strip():
            continue
        body = "".join(block.split("\n")[1:])
        aligned: list[str] = []
        counts: list[int] = []
        pending = 0
        for character in body:
            if character.islower():
                pending += 1
                continue
            aligned.append(character.upper() if character != "-" else "-")
            counts.append(pending)
            pending = 0
        if len(aligned) != len(query):
            continue
        sequence = "".join(aligned)
        if sequence == query and not any(counts):
            continue
        rows.append((sequence, counts))
    return rows


__all__ = ["ATOM_BLOCK", "build_features", "build_msa", "read_a3m"]
