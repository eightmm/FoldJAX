"""`build_msa`'s ``msa_depth`` is one cap on the whole alignment, shared fairly.

The cap used to be checked against the shared row list inside the per-chain
loop: the first chain filled it, and every later chain still appended one row
past it -- so the cap was exceeded and the later chains were starved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from foldjax.models.esmfold2.data import chemistry
from foldjax.models.esmfold2.data.features import build_features

FIRST = "MKTAYIAK"
SECOND = "GSHMASMT"


def _a3m(path: Path, query: str, hits: int) -> Path:
    lines = [f">query\n{query}\n"]
    for index in range(hits):
        row = list(query)
        row[index % len(row)] = "W"
        lines.append(f">hit{index}\n{''.join(row)}\n")
    path.write_text("".join(lines))
    return path


def _rows_per_chain(msa: np.ndarray) -> tuple[int, int]:
    aligned = msa[1:] != chemistry.MSA_GAP_TOKEN_ID
    first = int(aligned[:, : len(FIRST)].any(axis=1).sum())
    second = int(aligned[:, len(FIRST) :].any(axis=1).sum())
    return first, second


def _features(tmp_path: Path, *, depth: int | None, first: int = 6, second: int = 6):
    alignments = {
        0: _a3m(tmp_path / "a.a3m", FIRST, first),
        1: _a3m(tmp_path / "b.a3m", SECOND, second),
    }
    chains = [(FIRST, "A", 0, 0), (SECOND, "B", 1, 0)]
    return build_features(chains, alignments, msa_depth=depth)["msa"][0]


def test_the_cap_counts_every_row_and_is_never_exceeded(tmp_path: Path) -> None:
    msa = _features(tmp_path, depth=4)
    assert msa.shape[0] == 4
    # Query plus three rows: the first chain takes the odd one, the second
    # chain is not starved.
    assert _rows_per_chain(msa) == (2, 1)


def test_a_short_chain_leaves_its_share_to_the_other(tmp_path: Path) -> None:
    msa = _features(tmp_path, depth=6, first=1, second=6)
    assert msa.shape[0] == 6
    assert _rows_per_chain(msa) == (1, 4)


def test_a_cap_above_the_alignment_is_the_uncapped_result(tmp_path: Path) -> None:
    capped = _features(tmp_path, depth=100)
    uncapped = _features(tmp_path, depth=None)
    np.testing.assert_array_equal(capped, uncapped)
    assert uncapped.shape[0] == 13


def test_a_cap_of_one_is_the_query_alone(tmp_path: Path) -> None:
    assert _features(tmp_path, depth=1).shape[0] == 1
