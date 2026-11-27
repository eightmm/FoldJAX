"""Vendored from boltz.data.parse.a3m (self-contained, no `import boltz`).

Parses A3M MSA files into the MSA dataclass. Imports remapped to foldjax.models.boltz2.
"""

import gzip
from pathlib import Path
from typing import TextIO

import numpy as np

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.types import MSA, MSADeletion, MSAResidue, MSASequence

#: Rows and characters per vectorised block, keeping per-byte work arrays bounded.
_ROW_BLOCK = 1024
_BYTE_BLOCK = 1 << 20

_INSERTION_CODE = -1
_INVALID_CODE = -2
#: A row before any header: the loop read its characters, then failed on the
#: taxonomy it never assigned.
_NO_HEADER = object()


def _ascii_table() -> np.ndarray:
    """Per-byte Boltz-2 residue ids for ASCII rows; -1 insertion, -2 invalid."""
    table = np.full(128, _INVALID_CODE, dtype=np.int8)
    for code in range(128):
        char = chr(code)
        if char != "-" and char.islower():
            table[code] = _INSERTION_CODE
        elif char in const.prot_letter_to_token:
            table.put(code, const.token_ids[const.prot_letter_to_token[char]])
    return table


_ASCII_TABLE = _ascii_table()


#: Deletes every character the byte table accepts; what remains marks a row
#: the character loop must read, so its KeyError is raised where it was.
_DROP_TABLE_CHARACTERS = str.maketrans(
    "",
    "",
    "".join(chr(code) for code in range(128) if _ASCII_TABLE[code] != _INVALID_CODE),
)


def _flush_ascii_block(
    block_lines: list[str],
    block_taxonomies: list[int | str],
    seq_idx_start: int,
    res_offset: int,
    del_offset: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple]]:
    """One bounded block of kept, valid ASCII rows.

    Residue types come back as their structured array, which cannot overflow;
    deletions and sequence rows come back as the Python tuples the loop built,
    so the final ``np.array`` calls refuse an out-of-range value exactly as,
    and only when, the loop's did.
    """
    n_rows = len(block_lines)
    lengths = np.fromiter(map(len, block_lines), dtype=np.int64, count=n_rows)
    raw_codes = np.frombuffer("".join(block_lines).encode("ascii"), dtype=np.uint8)
    mapped = _ASCII_TABLE[raw_codes]

    starts = np.empty(n_rows, dtype=np.int64)
    starts[0] = 0
    np.cumsum(lengths[:-1], out=starts[1:])

    positions = np.flatnonzero(mapped >= 0)
    row_pos_start = np.searchsorted(positions, starts, side="left")
    row_pos_end = np.empty(n_rows, dtype=np.int64)
    row_pos_end[:-1] = row_pos_start[1:]
    row_pos_end[-1] = positions.size

    block_residues = mapped[positions].view(MSAResidue)

    if positions.size:
        prev_pos = np.empty(positions.size, dtype=np.int64)
        prev_pos[0] = -1
        prev_pos[1:] = positions[:-1]
        non_empty = row_pos_start < row_pos_end
        prev_pos[row_pos_start[non_empty]] = starts[non_empty] - 1
        counts = positions - prev_pos - 1
        del_k = np.flatnonzero(counts)
    else:
        counts = positions
        del_k = positions

    row_del_start = np.searchsorted(del_k, row_pos_start, side="left")
    row_del_end = np.empty(n_rows, dtype=np.int64)
    row_del_end[:-1] = row_del_start[1:]
    row_del_end[-1] = del_k.size

    block_deletions: list[tuple[int, int]] = []
    if del_k.size:
        del_row = np.searchsorted(row_pos_start, del_k, side="right") - 1
        block_deletions = list(
            zip(
                (del_k - row_pos_start[del_row]).tolist(),
                counts[del_k].tolist(),
                strict=True,
            )
        )

    block_sequences = list(
        zip(
            range(seq_idx_start, seq_idx_start + n_rows),
            block_taxonomies,
            (res_offset + row_pos_start).tolist(),
            (res_offset + row_pos_end).tolist(),
            (del_offset + row_del_start).tolist(),
            (del_offset + row_del_end).tolist(),
            strict=True,
        )
    )
    return block_residues, block_deletions, block_sequences


def _parse_row_loop(
    line: str,
    taxonomy_id: int | str,
    seq_idx: int,
    res_offset: int,
    del_offset: int,
) -> tuple[list[int], list[tuple[int, int]], list[tuple]]:
    """The released per-character loop, for a non-ASCII or invalid row."""
    residue = []
    deletion = []
    count = 0
    res_idx = 0
    for c in line:
        if c != "-" and c.islower():
            count += 1
            continue
        residue.append(const.token_ids[const.prot_letter_to_token[c]])
        if count > 0:
            deletion.append((res_idx, count))
            count = 0
        res_idx += 1

    res_end = res_offset + len(residue)
    del_end = del_offset + len(deletion)
    if taxonomy_id is _NO_HEADER:
        raise UnboundLocalError(
            "cannot access local variable 'taxonomy_id' where it is not "
            "associated with a value"
        )
    sequence = [(seq_idx, taxonomy_id, res_offset, res_end, del_offset, del_end)]
    return residue, deletion, sequence


def _parse_a3m(
    lines: TextIO,
    taxonomy: dict[str, str] | None,
    max_seqs: int | None = None,
) -> MSA:
    """Process an MSA file.

    Parameters
    ----------
    lines : TextIO
        The lines of the MSA file.
    taxonomy : dict[str, str]
        The taxonomy database, if available.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    visited: set[str] = set()
    # One byte per residue type (the table's ids are 0..127), appended by both
    # paths in row order and viewed as the structured array at the end.
    residue_bytes = bytearray()
    deletions: list[tuple[int, int]] = []
    sequences: list[tuple] = []
    res_count = 0
    del_count = 0

    block_lines: list[str] = []
    block_taxonomies: list[int | str] = []
    block_bytes = 0
    seq_idx = 0
    taxonomy_id: object = _NO_HEADER

    def flush_block() -> None:
        nonlocal block_bytes, del_count, res_count
        if not block_lines:
            return
        res_chunk, del_chunk, seq_chunk = _flush_ascii_block(
            block_lines,
            block_taxonomies,
            seq_idx - len(block_lines),
            res_count,
            del_count,
        )
        residue_bytes.extend(res_chunk.tobytes())
        res_count += int(res_chunk.size)
        deletions.extend(del_chunk)
        del_count += len(del_chunk)
        sequences.extend(seq_chunk)
        block_lines.clear()
        block_taxonomies.clear()
        block_bytes = 0

    for line in lines:
        line = line.strip()  # noqa: PLW2901
        if not line or line.startswith("#"):
            continue

        # Get taxonomy, if annotated
        if line.startswith(">"):
            header = line.split()[0]
            if taxonomy and header.startswith(">UniRef100"):
                uniref_id = header.split("_")[1]
                found = taxonomy.get(uniref_id)
                taxonomy_id = -1 if found is None else found
            else:
                taxonomy_id = -1
            continue

        # Skip if duplicate sequence
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Anything the byte table cannot reproduce row for row -- a non-ASCII
        # or unknown character, a row before any header, or a row larger than
        # the byte budget, which the loop holds no worse -- takes the loop.
        if (
            taxonomy_id is not _NO_HEADER
            and len(line) <= _BYTE_BLOCK
            and line.isascii()
            and not line.translate(_DROP_TABLE_CHARACTERS)
        ):
            size = len(line)
            if block_lines and (
                len(block_lines) >= _ROW_BLOCK or block_bytes + size > _BYTE_BLOCK
            ):
                flush_block()
            block_lines.append(line)
            block_taxonomies.append(taxonomy_id)
            block_bytes += size
            seq_idx += 1
        else:
            flush_block()
            res_chunk, del_chunk, seq_chunk = _parse_row_loop(
                line, taxonomy_id, seq_idx, res_count, del_count
            )
            residue_bytes.extend(res_chunk)
            res_count += len(res_chunk)
            deletions.extend(del_chunk)
            del_count += len(del_chunk)
            sequences.extend(seq_chunk)
            # Released now: a row's residue list must not outlive it into the
            # next fallback row.
            del res_chunk, del_chunk, seq_chunk
            seq_idx += 1

        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    flush_block()

    # Built in the loop's order, after every row has been read, so an
    # out-of-range deletion or offset raises the loop's OverflowError.
    return MSA(
        residues=np.frombuffer(residue_bytes, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def parse_a3m(
    path: Path,
    taxonomy: dict[str, str] | None,
    max_seqs: int | None = None,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    taxonomy : Redis
        The taxonomy database.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read the file
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            msa = _parse_a3m(f, taxonomy, max_seqs)
    else:
        with path.open("r") as f:
            msa = _parse_a3m(f, taxonomy, max_seqs)

    return msa
