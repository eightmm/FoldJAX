import json
import sys

import numpy as np
import pytest

from bench.esmfold2_msa_prefix_report import main
from bench.esmfold2_tape import _sha256


@pytest.mark.parametrize("missing", [False, True])
def test_stack_requires_matching_block_coverage(tmp_path, monkeypatch, missing):
    for engine in ("native", "jax"):
        root = tmp_path / engine
        root.mkdir()
        keys = ["blocks.0.msa", "blocks.0.pair"]
        if missing and engine == "jax":
            keys.pop()
        np.savez(root / "prefix.npz", **{k: np.ones((1, 2), np.float32) for k in keys})
        (root / "report.json").write_text(
            json.dumps(
                {
                    "engine": engine,
                    "stack": True,
                    "bindings": {},
                    "dtypes": {k: "bfloat16" for k in keys},
                    "archive_sha256": _sha256(root / "prefix.npz"),
                }
            )
        )
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--stack",
            "--native",
            str(tmp_path / "native"),
            "--candidate",
            str(tmp_path / "jax"),
            "--output",
            str(output),
        ],
    )
    if missing:
        with pytest.raises(ValueError, match="coverage differs"):
            main()
        assert not output.exists()
    else:
        main()
        report = json.loads(output.read_text())
        assert len(report["comparisons"]) == 2
        assert report["full_model_admission"] is False


@pytest.mark.parametrize("wrong_shape", [False, True])
def test_explicit_input_mapping_and_shape_guard(tmp_path, monkeypatch, wrong_shape):
    key = "blocks.0.msa_transition.input"
    target = "blocks.0.msa_transition.norm.input"
    for engine, name, shape in (
        ("native", target, (1, 2, 3)),
        ("jax", key, (1, 3, 3) if wrong_shape else (1, 2, 3)),
    ):
        root = tmp_path / engine
        root.mkdir()
        np.savez(root / "prefix.npz", **{name: np.ones(shape, np.float32)})
        (root / "report.json").write_text(
            json.dumps(
                {
                    "engine": engine,
                    "full_msa_block": True,
                    "bindings": {},
                    "dtypes": {name: "bfloat16"},
                    "archive_sha256": _sha256(root / "prefix.npz"),
                }
            )
        )
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--native",
            str(tmp_path / "native"),
            "--candidate",
            str(tmp_path / "jax"),
            "--output",
            str(output),
        ],
    )
    if wrong_shape:
        with pytest.raises(ValueError, match="shape differs"):
            main()
        assert not output.exists()
    else:
        main()
        report = json.loads(output.read_text())
        assert report["comparisons"][key]["native_key"] == target
        assert report["comparisons"][key]["comparison"]["bitwise_equal"]
        assert report["full_model_admission"] is False
