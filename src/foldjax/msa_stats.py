"""How deep and how diverse each chain's alignment is, for the run manifest.

`depth` is the number of rows a chain's alignment file holds, query included,
before the model's own selection (``max_msa_depth``, deduplication, a native
cap) reads it: the same file run through two models can reach them at two
depths. `neff` is the effective number of sequences at 80% identity, the
diversity measure a structure is usually judged against, because a deep
alignment of near-copies carries little more than the query does.

Definition (`NEFF_DEFINITION`, also written into each record): over the
query's match columns (A3M upper-case letters and ``-``; lower-case
insertions and ``.`` removed), two rows' identity is the fraction of those
columns at which they carry the same symbol, a gap counting as a symbol (the
plmDCA / EVcouplings convention). Each row is weighted by one over the number
of rows, itself included, at identity >= 0.8 to it; Neff is the sum of the
weights. A row whose match-column count differs from the query's is not an
A3M row of this alignment and is left out of Neff (``rows_skipped``).

NumPy only; nothing here imports JAX.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

#: The identity at or above which two rows count as one sequence.
NEFF_IDENTITY = 0.8
#: At most this many rows enter the Neff computation, the first ones in file
#: order; deeper alignments record ``neff_rows`` below ``depth``. The cost is
#: quadratic in rows (a 16k-row, 500-column alignment takes seconds).
NEFF_MAX_ROWS = 20_000
NEFF_DEFINITION = (
    "sum over rows of 1 / #rows at >= 80% identity (itself included); identity "
    "= fraction of the query's match columns with the same symbol, gap counted "
    "as a symbol; A3M insertions removed"
)
_BLOCK = 2048
#: 26 letters plus the gap.
_ALPHABET = 27


def _match_columns(row: str) -> str:
    return "".join(c for c in row if c.isupper() or c == "-")


def _rows(path: Path) -> list[str]:
    """The alignment's rows: A3M/FASTA records, or a Boltz keyed CSV."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        return [
            str(row.get("sequence") or "").strip()
            for row in reader
            if str(row.get("sequence") or "").strip()
        ]
    from foldjax.input import _a3m_rows

    return _a3m_rows(text.replace("\x00", ""))


def neff(rows: list[str], *, identity: float = NEFF_IDENTITY) -> tuple[float, int]:
    """Neff of A3M ``rows`` (the first is the query), and the rows it used."""
    import numpy as np

    if not rows:
        return 0.0, 0
    columns = [_match_columns(row) for row in rows[:NEFF_MAX_ROWS]]
    length = len(columns[0])
    kept = [row for row in columns if len(row) == length]
    if length == 0 or not kept:
        return 0.0, 0
    codes = np.frombuffer("".join(kept).encode("ascii"), dtype=np.uint8).reshape(
        len(kept), length
    )
    codes = np.where(codes == ord("-"), 26, codes - ord("A")).astype(np.intp)
    codes = np.clip(codes, 0, _ALPHABET - 1)
    # Integer match counts at or above this are >= `identity` of the columns.
    threshold = math.ceil(identity * length - 1e-9)
    count = len(kept)
    neighbours = np.zeros(count, dtype=np.int64)

    def one_hot(block: Any) -> Any:
        encoded = np.zeros((block.shape[0], length * _ALPHABET), dtype=np.float32)
        offsets = np.arange(length) * _ALPHABET
        encoded[np.arange(block.shape[0])[:, None], offsets[None, :] + block] = 1.0
        return encoded

    for start in range(0, count, _BLOCK):
        left = one_hot(codes[start : start + _BLOCK])
        for other in range(0, count, _BLOCK):
            right = one_hot(codes[other : other + _BLOCK])
            matches = left @ right.T
            neighbours[start : start + _BLOCK] += (
                matches >= threshold - 0.5
            ).sum(axis=1)
    return float((1.0 / neighbours).sum()), count


def alignment_stats(path: str | Path) -> dict[str, Any]:
    """``depth`` (rows) and ``neff`` of one alignment file, or why not.

    Memoized on the file's identity, so the seeds of one request, which each
    translate the same job, pay for Neff once.
    """
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as error:
        return {"path": str(path), "depth": None, "neff": None, "error": str(error)}
    return dict(_alignment_stats(str(path), stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=256)
def _alignment_stats(raw: str, _mtime: int, _size: int) -> dict[str, Any]:
    path = Path(raw)
    try:
        rows = _rows(path)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return {"path": str(path), "depth": None, "neff": None, "error": str(error)}
    value, used = neff(rows)
    record: dict[str, Any] = {
        "path": str(path),
        "depth": len(rows),
        "neff": round(value, 2),
        "neff_rows": used,
    }
    skipped = min(len(rows), NEFF_MAX_ROWS) - used
    if skipped:
        record["rows_skipped"] = skipped
    return record


def job_msa_stats(
    job: Mapping[str, Any], base: Path, *, skip: set[str] | None = None
) -> list[dict[str, Any]]:
    """One record per chain that carries an alignment, as the model reads it.

    ``skip`` holds resolved alignment paths the native input leaves out (the
    run's ``ignored_msas``); those are not what the model saw.
    """
    from foldjax.input import _ids, _path

    skip = skip or set()
    records = []
    for entity in job.get("entities") or []:
        if not isinstance(entity, Mapping):
            continue
        record: dict[str, Any] = {}
        for field in ("unpaired_msa", "paired_msa"):
            value = entity.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            resolved = _path(value, base)
            if resolved in skip:
                continue
            record[field] = alignment_stats(resolved)
        if record:
            records.append(
                {
                    "chains": _ids(entity),
                    "type": entity.get("type"),
                    **record,
                    "neff_identity": NEFF_IDENTITY,
                }
            )
    return records
