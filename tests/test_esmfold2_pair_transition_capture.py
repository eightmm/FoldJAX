import json
import sys

import pytest

from bench.esmfold2_pair_transition_capture import main


@pytest.mark.parametrize("engine,full", [("jax", True), ("native", False)])
def test_requires_full_native_reference_before_import(
    tmp_path, monkeypatch, engine, full
):
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "engine": engine,
                "full_msa_block": full,
            }
        )
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture",
            "--reference",
            str(tmp_path),
            "--weights",
            str(tmp_path / "weights"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="full MSA block"):
        main()
    assert not output.exists()
