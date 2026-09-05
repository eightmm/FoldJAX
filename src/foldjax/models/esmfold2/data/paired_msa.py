"""Taxonomy-paired ESMFold2 MSAs, adapted from Biohub's MIT-licensed source.

Copyright 2026 Chan Zuckerberg Biohub, Inc. See ../LICENSE.

Adapted from ``esm/models/esmfold2/paired_msa.py`` at Biohub ESM commit
bf343ba264b650dff7a073643725f9aaa1fdbe8d.  The small A3M reader replaces
the upstream ``MSA`` dependency so the installed FoldJAX package stays
Torch- and BioPython-free.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from foldjax.models.esmfold2.data.all_atom_constants import (
    MSA_GAP_TOKEN_ID,
    PROTEIN_1TO3,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
)

_KEY_RE = re.compile(r"key=(-?\d+)")


@dataclass(frozen=True, slots=True)
class MSAEntry:
    header: str
    sequence: str


@dataclass(frozen=True, slots=True)
class MSA:
    entries: tuple[MSAEntry, ...]

    @property
    def depth(self) -> int:
        return len(self.entries)


def read_a3m(path: Path, *, expected_columns: int) -> MSA:
    """Read a complete A3M without dropping query-like hits or insertions."""

    records: list[MSAEntry] = []
    header: str | None = None
    sequence: list[str] = []
    try:
        source = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"ESMFold2 could not read MSA {path}: {error}") from error
    with source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                if header is not None:
                    if not sequence:
                        raise ValueError(
                            f"ESMFold2 MSA {path} has no sequence for header {header!r}"
                        )
                    records.append(MSAEntry(header, "".join(sequence)))
                header = line[1:]
                sequence = []
            elif header is None:
                if line.strip():
                    raise ValueError(
                        f"ESMFold2 MSA {path} has sequence data before a FASTA header "
                        f"at line {line_number}"
                    )
            elif line.strip():
                sequence.append(line.strip())
        if header is not None:
            if not sequence:
                raise ValueError(
                    f"ESMFold2 MSA {path} has no sequence for header {header!r}"
                )
            records.append(MSAEntry(header, "".join(sequence)))
    if not records:
        raise ValueError(f"ESMFold2 MSA {path} contains no FASTA records")

    widths = [_match_columns(entry.sequence) for entry in records]
    if any(width != widths[0] for width in widths[1:]):
        raise ValueError(
            f"ESMFold2 MSA {path} has inconsistent A3M match-column counts: {widths}"
        )
    if widths[0] != expected_columns:
        raise ValueError(
            f"ESMFold2 MSA {path} has {widths[0]} match columns; expected "
            f"{expected_columns} for its protein sequence"
        )
    return MSA(tuple(records))


def _match_columns(sequence: str) -> int:
    return sum(character != "." and not character.islower() for character in sequence)


def _taxonomy_from_header(header: str) -> int:
    match = _KEY_RE.search(header)
    return int(match.group(1)) if match else -1


def _res_types_and_deletions(msa: MSA) -> tuple[np.ndarray, np.ndarray]:
    length = _match_columns(msa.entries[0].sequence)
    residues = np.full((msa.depth, length), MSA_GAP_TOKEN_ID, dtype=np.int64)
    deletions = np.zeros((msa.depth, length), dtype=np.float32)
    for row, entry in enumerate(msa.entries):
        column = 0
        insertions = 0
        for character in entry.sequence:
            if character == "." or character.islower():
                insertions += 1
                continue
            if column >= length:
                break
            if character != "-":
                residues[row, column] = PROTEIN_RESIDUE_TO_RES_TYPE.get(
                    PROTEIN_1TO3.get(character.upper(), "UNK"), PROTEIN_UNK_RES_TYPE
                )
            if insertions:
                deletions[row, column] = float(insertions)
                insertions = 0
            column += 1
    return residues, deletions


def construct_paired_msa(
    chain_msas: Mapping[int, MSA | None],
    chain_query_res_types: Mapping[int, np.ndarray],
    token_asym_ids: np.ndarray,
    token_res_ids: np.ndarray,
    *,
    max_seqs: int = 16384,
    max_pairs: int = 8192,
    max_total: int = 16384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct Biohub's taxonomy-paired MSA rows exactly."""

    if any(
        type(limit) is not int or limit < 1
        for limit in (max_seqs, max_pairs, max_total)
    ):
        raise ValueError("MSA row limits must be positive integers")

    chain_ids = sorted(chain_msas)
    chain_residues: dict[int, np.ndarray] = {}
    chain_deletions: dict[int, np.ndarray] = {}
    chain_taxonomies: dict[int, list[int]] = {}
    for chain_id in chain_ids:
        msa = chain_msas[chain_id]
        if msa is None or not msa.depth:
            query = chain_query_res_types[chain_id]
            chain_residues[chain_id] = query[None, :]
            chain_deletions[chain_id] = np.zeros((1, len(query)), dtype=np.float32)
            chain_taxonomies[chain_id] = [-1]
        else:
            residues, deletions = _res_types_and_deletions(msa)
            chain_residues[chain_id] = residues
            chain_deletions[chain_id] = deletions
            chain_taxonomies[chain_id] = [
                _taxonomy_from_header(entry.header) for entry in msa.entries
            ]

    taxonomy_map: dict[int, list[tuple[int, int]]] = {}
    for chain_id in chain_ids:
        for sequence_index, taxon in enumerate(chain_taxonomies[chain_id]):
            if sequence_index and taxon != -1:
                taxonomy_map.setdefault(taxon, []).append((chain_id, sequence_index))
    taxonomy_map = {
        taxon: entries for taxon, entries in taxonomy_map.items() if len(entries) > 1
    }
    sorted_taxa = sorted(
        taxonomy_map.items(),
        key=lambda item: len({chain_id for chain_id, _ in item[1]}),
        reverse=True,
    )
    visited = {entry for entries in taxonomy_map.values() for entry in entries}
    available = {
        chain_id: [
            index
            for index in range(1, len(chain_taxonomies[chain_id]))
            if (chain_id, index) not in visited
        ]
        for chain_id in chain_ids
    }

    pairing: list[dict[int, int]] = [{chain_id: 0 for chain_id in chain_ids}]
    is_paired: list[dict[int, int]] = [{chain_id: 1 for chain_id in chain_ids}]
    for _, entries in sorted_taxa:
        per_chain: dict[int, list[int]] = {}
        for chain_id, sequence_index in entries:
            per_chain.setdefault(chain_id, []).append(sequence_index)
        for occurrence in range(max(map(len, per_chain.values()))):
            row = {}
            paired = {}
            for chain_id, indices in per_chain.items():
                row[chain_id] = indices[occurrence % len(indices)]
                paired[chain_id] = 1
            for chain_id in chain_ids:
                if chain_id not in row:
                    paired[chain_id] = 0
                    row[chain_id] = (
                        available[chain_id].pop(0) if available[chain_id] else -1
                    )
            pairing.append(row)
            is_paired.append(paired)
            if len(pairing) >= max_pairs:
                break
        if len(pairing) >= max_pairs:
            break

    remaining_rows = min(
        max_total - len(pairing), max(map(len, available.values()), default=0)
    )
    for _ in range(remaining_rows):
        row = {}
        paired = {}
        for chain_id in chain_ids:
            paired[chain_id] = 0
            row[chain_id] = available[chain_id].pop(0) if available[chain_id] else -1
        pairing.append(row)
        is_paired.append(paired)
        if len(pairing) >= max_total:
            break

    pairing = pairing[:max_seqs]
    is_paired = is_paired[:max_seqs]
    rows = len(pairing)
    msa = np.full((rows, len(token_asym_ids)), MSA_GAP_TOKEN_ID, dtype=np.int64)
    deletions = np.zeros((rows, len(token_asym_ids)), dtype=np.float32)
    paired_mask = np.zeros((rows, len(token_asym_ids)), dtype=np.float32)
    for chain_id in chain_ids:
        residues = chain_residues[chain_id]
        deletion_counts = chain_deletions[chain_id]
        chain_pairing = np.asarray([row[chain_id] for row in pairing], dtype=np.int64)
        chain_paired = np.asarray(
            [row[chain_id] for row in is_paired], dtype=np.float32
        )
        token_mask = token_asym_ids == chain_id
        if not token_mask.any():
            continue
        columns = np.minimum(token_res_ids[token_mask], residues.shape[1] - 1)
        valid_rows = chain_pairing >= 0
        if valid_rows.any():
            valid_indices = np.flatnonzero(valid_rows)
            token_indices = np.flatnonzero(token_mask)
            msa[np.ix_(valid_indices, token_indices)] = residues[
                chain_pairing[valid_rows]
            ][:, columns]
            deletions[np.ix_(valid_indices, token_indices)] = deletion_counts[
                chain_pairing[valid_rows]
            ][:, columns]
        paired_mask[:, token_mask] = chain_paired[:, None]
    return msa, deletions, paired_mask
