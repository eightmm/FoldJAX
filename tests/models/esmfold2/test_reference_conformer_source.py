"""A chain's reference conformers do not depend on whether it has an MSA.

Every ESMFold2 job is featurized by Biohub's all-atom pipeline, whose
reference positions are the CCD's ideal conformers. A bare single chain used to
take the transformers fork's protein-only featurizer and its own conformer
table instead, so attaching an alignment changed `ref_pos`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.paths import weights_dir

_SEQUENCE = "MKTAYIAKQR"


def _ccd() -> Path:
    path = weights_dir("esmfold2") / "ccd.pkl"
    if not path.is_file():
        pytest.skip("esmfold2 ccd.pkl not fetched")
    return path


def test_an_alignment_does_not_change_the_reference_conformers(tmp_path) -> None:
    ccd = _ccd()
    alignment = tmp_path / "chain.a3m"
    alignment.write_text(f">query\n{_SEQUENCE}\n>hit\nMKTAYIAKQK\n")
    bare = {"type": "protein", "id": "A", "sequence": _SEQUENCE}

    def ref_pos(entity) -> np.ndarray:
        features = inference.build_common_job_features(
            {"entities": [entity]}, base_dir=tmp_path, ccd_path=ccd, seed=0
        )
        return np.asarray(features["ref_pos"])

    with_msa = ref_pos({**bare, "unpaired_msa": "chain.a3m"})
    np.testing.assert_array_equal(ref_pos(bare), with_msa)
    # And they are the CCD's, not the protein-only table the bare chain used.
    legacy = np.asarray(
        inference.build_job_features([(_SEQUENCE, "A", 0, 0)], {})["ref_pos"]
    )
    assert legacy.shape == with_msa.shape
    assert not np.array_equal(legacy, with_msa)
