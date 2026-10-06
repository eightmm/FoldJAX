"""`max_msa_depth` above upstream's 8,192-row parse cap reaches the features.

Preprocessing reads at most `--max_msa_seqs` rows of each alignment file
(upstream ``main.py``, released 8192) before the featurizer applies its own
cap, so a deeper `max_msa_depth` used to change nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.boltz2 import api
from foldjax.models.boltz2.data.featurize import parse_cap
from foldjax.paths import weights_dir

_SEQUENCE = "MKTAYIAKQR"
_ROWS = 8300


def test_the_parse_cap_follows_a_deeper_max_msa_depth_only() -> None:
    assert parse_cap(None) == 8192
    assert parse_cap(1024) == 8192
    assert parse_cap(8192) == 8192
    assert parse_cap(12000) == 12000


def _alignment(path: Path) -> int:
    """Write the alignment; return its distinct rows, which is what is parsed."""
    letters = "ACDEFGHIKLMNPQRSTVWY"
    sequences = [_SEQUENCE]
    for index in range(_ROWS):
        head = "".join(letters[(index // 20**k) % 20] for k in range(4))
        sequences.append(f"{head}{_SEQUENCE[4:]}")
    path.write_text(
        "\n".join(f">row{i}\n{s}" for i, s in enumerate(sequences)) + "\n"
    )
    return len(set(sequences))


@pytest.mark.slow
def test_a_deeper_max_msa_depth_reads_past_8192_rows(tmp_path: Path) -> None:
    mols = weights_dir("boltz2") / "mols"
    if not mols.is_dir():
        pytest.skip("boltz2 molecule directory not fetched")
    alignment = tmp_path / "deep.a3m"
    distinct = _alignment(alignment)
    assert distinct > 8192
    job = tmp_path / "job.yaml"
    job.write_text(
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {_SEQUENCE}\n"
        f"      msa: {alignment}\n"
    )

    def depth(name: str, max_msa_depth: int | None) -> int:
        feats, _, _ = api.featurize(
            input=job,
            mols=mols,
            out_dir=tmp_path / name,
            max_msa_depth=max_msa_depth,
        )
        return int(np.asarray(feats["msa"]).shape[1])

    assert depth("released", None) == 8192
    assert depth("deep", 9000) == distinct
