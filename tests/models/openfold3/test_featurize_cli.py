"""The featurize CLI writes a loadable .npz and refuses half-given padding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.openfold3.cli.featurize import main
from foldjax.models.openfold3.data import MODEL_FEATURES

pytestmark = pytest.mark.torch_parity

UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "query.json"
    path.write_text(
        json.dumps(
            {
                "queries": {
                    "ubq": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": UBIQUITIN,
                            }
                        ]
                    }
                }
            }
        )
    )
    return path


def test_writes_every_model_feature(
    openfold3_source: Path, spec_file: Path, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "ubq.npz"
    assert main([str(spec_file), "-o", str(out)]) == 0
    loaded = np.load(out, allow_pickle=False)
    for name in MODEL_FEATURES:
        assert name in loaded, name
    printed = capsys.readouterr().out
    assert "76 tokens" in printed
    # A single-sequence MSA is a real accuracy problem, so it must be said out loud.
    assert "only the query sequence" in printed


def test_padding_changes_the_written_shapes(
    openfold3_source: Path, spec_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "padded.npz"
    assert (
        main(
            [
                str(spec_file),
                "-o",
                str(out),
                "--pad-tokens",
                "96",
                "--pad-atoms",
                "800",
            ]
        )
        == 0
    )
    loaded = np.load(out, allow_pickle=False)
    assert loaded["token_mask"].shape == (1, 96)
    assert loaded["atom_mask"].shape == (1, 800)
    assert float(loaded["token_mask"].sum()) == 76.0


def test_half_given_padding_is_a_usage_error(spec_file: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([str(spec_file), "-o", str(tmp_path / "x.npz"), "--pad-tokens", "96"])
    assert excinfo.value.code == 2
