from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from foldjax.models.chai.data.templates import load_native_template_context


def test_export_template_context_from_official_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.npz"
    destination = tmp_path / "templates.npz"
    identity_path = tmp_path / "identity.json"
    shape = (1, 4, 3)
    arrays = {
        "inputs/template_restype": np.full(shape, 31, np.int32),
        "inputs/template_pseudo_beta_mask": np.zeros(shape, np.bool_),
        "inputs/template_backbone_frame_mask": np.zeros(shape, np.bool_),
        "inputs/template_distances": np.zeros(shape + (3,), np.float32),
        "inputs/template_unit_vector": np.zeros(shape + (3, 3), np.float32),
    }
    arrays["inputs/template_restype"][0, 0] = np.asarray([0, 1, 2])
    np.savez(fixture, **arrays)
    identity = {"entity_names": ["protein"], "sequence_sha256": ["a" * 64]}
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/export_template_context.py",
            "--fixture",
            str(fixture),
            "--destination",
            str(destination),
            "--query-identity",
            str(identity_path),
        ],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = load_native_template_context(destination, expected_query_identity=identity)
    assert loaded.context.template_restype.shape == (4, 3)
    np.testing.assert_array_equal(
        loaded.context.template_restype[0], np.asarray([0, 1, 2])
    )
