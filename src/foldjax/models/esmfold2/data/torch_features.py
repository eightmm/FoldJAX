"""Build ESMFold2's feature dictionary for a whole job, not one sequence.

Torch tensors, because it feeds upstream's torch model through
`foldjax.backends.esmfold2`. It lives under `models/` rather than beside that
backend because it is model featurization -- upstream's residue table, atom
order and reference conformers -- and not the request plumbing the backend
layer holds. The JAX port growing in `../models/` will want the same
assembly against `jnp`, and this is the file it starts from.

Upstream ships `prepare_protein_features`, which covers a single protein chain
with no alignment -- everything else its `forward` accepts (several chains, a
real MSA, symmetry) is documented by that function's own shapes and by the
model's use of them, and left to the caller to assemble. This module is that
assembly, written against upstream's conventions rather than around them:

* residue types index a 33-entry vocabulary (`PROTEIN_RESIDUE_TO_RES_TYPE`),
  they are not one-hot -- the model one-hots them itself;
* atoms are the heavy atoms of each residue in upstream's order, with its
  reference conformer coordinates, and the atom axis is padded to a multiple
  of 32;
* elements are atomic numbers and atom names are its four-character encoding,
  neither of which is the AlphaFold-3 one-hot the FoldJAX ports build;
* the MSA carries residue-type indices with gaps at `MSA_GAP_TOKEN_ID`.

The single-chain, no-MSA case is checked against `prepare_protein_features`
tensor for tensor in `tests/test_esmfold2_features.py`, so the shared half of
this code cannot drift from upstream's; the parts that function does not reach
-- chain identity and the alignment -- follow the rules stated on each below.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

ATOM_BLOCK = 32

#: res_type index -> one-letter code, for recognising the query row of an a3m.
#: Built once from upstream's own table so the two cannot disagree.
_RES_TYPE_LETTER: dict[int, str] = {}


def _res_type_letters() -> dict[int, str]:
    if not _RES_TYPE_LETTER:
        up = _upstream()
        for letter, residue in up.PROTEIN_1TO3.items():
            index = up.PROTEIN_RESIDUE_TO_RES_TYPE.get(residue)
            if index is not None:
                _RES_TYPE_LETTER.setdefault(index, letter)
    return _RES_TYPE_LETTER


def _upstream() -> Any:
    from transformers.models.esmfold2 import protein_utils

    return protein_utils


def build_features(
    chains: Sequence[tuple[str, str, int, int]],
    alignments: dict[int, Path] | None = None,
    *,
    msa_depth: int | None = None,
) -> dict[str, Any]:
    """Featurize a job.

    `chains` is one entry per chain copy: (sequence, chain id, entity index,
    symmetry index). `alignments` maps an entity index to its a3m; a chain
    whose entity has none contributes only the query row.
    """
    import torch

    up = _upstream()
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
            residue = up.PROTEIN_1TO3.get(letter, "UNK")
            names = up.PROTEIN_HEAVY_ATOMS[residue]
            token = len(res_type)
            atom_start = len(atom_records)
            for name in names:
                atom_records.append(
                    (
                        token,
                        name,
                        name[0],
                        up.PROTEIN_CHARGED_ATOMS.get((residue, name), 0),
                        up.PROTEIN_REF_POS[residue][name],
                    )
                )
            representative = "CB" if "CB" in names else "CA"
            distogram_atom_idx.append(atom_start + names.index(representative))
            res_type.append(
                up.PROTEIN_RESIDUE_TO_RES_TYPE.get(residue, up.PROTEIN_UNK_RES_TYPE)
            )
            input_ids.append(
                up.ESM_PROTEIN_VOCAB.get(letter, up.ESM_PROTEIN_VOCAB["X"])
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
    n_real_atoms = len(atom_records)
    n_atoms = -(-n_real_atoms // ATOM_BLOCK) * ATOM_BLOCK

    ref_pos = torch.zeros(n_atoms, 3, dtype=torch.float32)
    ref_element = torch.zeros(n_atoms, dtype=torch.int64)
    ref_charge = torch.zeros(n_atoms, dtype=torch.int8)
    ref_atom_name_chars = torch.zeros(n_atoms, 4, dtype=torch.int64)
    ref_space_uid = torch.zeros(n_atoms, dtype=torch.int64)
    atom_attention_mask = torch.zeros(n_atoms, dtype=torch.bool)
    atom_to_token = torch.zeros(n_atoms, dtype=torch.int64)
    for index, (token, name, element, charge, position) in enumerate(atom_records):
        ref_pos[index] = torch.tensor(position, dtype=torch.float32)
        ref_element[index] = up._PROTEIN_ELEMENT_TO_ATOMIC_NUM[element]
        ref_charge[index] = charge
        ref_atom_name_chars[index] = torch.tensor(
            up._encode_atom_name(name), dtype=torch.int64
        )
        ref_space_uid[index] = token
        atom_attention_mask[index] = True
        atom_to_token[index] = token

    msa, msa_mask, has_deletion, deletion_value = _msa(
        chain_spans, res_type, alignments, msa_depth=msa_depth
    )

    features = {
        "token_index": torch.arange(n_token, dtype=torch.int64),
        "residue_index": torch.tensor(residue_index, dtype=torch.int64),
        "asym_id": torch.tensor(asym_id, dtype=torch.int64),
        "sym_id": torch.tensor(sym_id, dtype=torch.int64),
        "entity_id": torch.tensor(entity_id, dtype=torch.int64),
        "mol_type": torch.full((n_token,), up.MOL_TYPE_PROTEIN, dtype=torch.int64),
        "res_type": torch.tensor(res_type, dtype=torch.int64),
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "token_bonds": torch.zeros(n_token, n_token, 1, dtype=torch.float32),
        "token_attention_mask": torch.ones(n_token, dtype=torch.bool),
        "ref_pos": ref_pos,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "atom_attention_mask": atom_attention_mask,
        "atom_to_token": atom_to_token,
        "distogram_atom_idx": torch.tensor(distogram_atom_idx, dtype=torch.int64),
        "msa": msa,
        "msa_attention_mask": msa_mask,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_value.float().mean(dim=0),
    }
    return {name: value.unsqueeze(0) for name, value in features.items()}


def _msa(
    chain_spans: Sequence[tuple[int, int, int]],
    res_type: Sequence[int],
    alignments: dict[int, Path],
    *,
    msa_depth: int | None,
) -> tuple[Any, Any, Any, Any]:
    """Assemble one alignment block per chain, gaps everywhere else.

    Row 0 is the query: every chain's own sequence, which is what the model
    reads when there is no alignment at all. Each further row belongs to one
    chain and is a gap over the others -- the unpaired arrangement, which is
    what a per-chain a3m can honestly support. Pairing rows across chains
    would claim the alignments were searched together, and they were not.
    """
    import torch

    up = _upstream()
    n_token = len(res_type)
    rows: list[list[int]] = [list(res_type)]
    deletions: list[list[float]] = [[0.0] * n_token]

    for start, end, entity in chain_spans:
        path = alignments.get(entity)
        if path is None:
            continue
        query = "".join(
            _res_type_letters().get(value, "X") for value in res_type[start:end]
        )
        for sequence, counts in _read_a3m(path, query=query):
            row = [up.MSA_GAP_TOKEN_ID] * n_token
            deletion = [0.0] * n_token
            for offset, (letter, count) in enumerate(zip(sequence, counts)):
                residue = up.PROTEIN_1TO3.get(letter, "UNK") if letter != "-" else None
                row[start + offset] = (
                    up.MSA_GAP_TOKEN_ID
                    if residue is None
                    else up.PROTEIN_RESIDUE_TO_RES_TYPE.get(
                        residue, up.PROTEIN_UNK_RES_TYPE
                    )
                )
                deletion[start + offset] = float(count)
            rows.append(row)
            deletions.append(deletion)
            if msa_depth is not None and len(rows) >= msa_depth:
                break

    msa = torch.tensor(rows, dtype=torch.int64)
    values = torch.tensor(deletions, dtype=torch.float32)
    return (
        msa,
        torch.ones(msa.shape, dtype=torch.bool),
        values > 0,
        values,
    )


def _read_a3m(path: Path, *, query: str) -> list[tuple[str, list[int]]]:
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
