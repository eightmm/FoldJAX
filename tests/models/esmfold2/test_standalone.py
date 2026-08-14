"""ESMFold2 folds with torch and transformers made unimportable.

The other tests in this directory all *require* torch -- they compare against
it. So none of them can tell you whether the prediction path needs it, and
"there is no import" is not the same claim as "it runs without one": a lazy
import inside a function is invisible to a grep and fatal at run time.

This blocks both modules at the import system and walks the whole path:
featurisation, both checkpoint readers, the model, and the writers.
"""

from __future__ import annotations

import builtins
import sys

import numpy as np
import pytest

BLOCKED = ("torch", "transformers", "esm")


@pytest.fixture
def without_torch(monkeypatch):
    """Make the blocked packages raise on import, however they are reached."""
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED:
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked: this path must not need it")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    return guarded


def test_the_blocker_actually_blocks(without_torch) -> None:
    """Otherwise every assertion below is vacuous."""
    with pytest.raises(ImportError):
        import torch  # noqa: F401


def test_featurisation_and_writing_need_no_torch(without_torch) -> None:
    from foldjax.models.esmfold2.data import features, pdb

    built = features.build_features([("MQIFVKTLTG", "A", 0, 0)])
    assert built["res_type"].shape == (1, 10)

    n_atoms = built["ref_pos"].shape[1]
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    assert pdb.to_pdb(coords, built).startswith("ATOM")
    assert "_atom_site" in pdb.to_mmcif(coords, built)


@pytest.mark.slow
def test_the_released_checkpoint_folds_with_no_torch(without_torch, tmp_path) -> None:
    """The real weights, the real model, the real writers -- torch unimportable.

    This is the claim that matters: not that the source has no import, but that
    a machine without torch installed can fold.
    """
    from foldjax.models.esmfold2 import inference, output
    from foldjax.models.esmfold2.bridge import checkpoint
    from foldjax.paths import weights_dir

    directory = weights_dir("esmfold2")
    if not (directory / checkpoint.WEIGHTS_NAME).exists():
        pytest.skip("esmfold2 weights are not in the store")

    model = inference.load(directory, language_model=False)
    prediction, features = inference.predict_job(
        inference.seed_key(0),
        [("MQIFVKTLTGKTITLE", "A", 0, 0)],
        None,
        model,
        num_loops=0,
        num_samples=1,
        num_steps=2,
    )
    written = output.write_prediction_outputs(
        prediction, features, tmp_path, name="job"
    )
    assert written["structures"][0].read_text().lstrip().startswith("data_")
    assert 0.0 <= written["summary"][0]["plddt"] <= 1.0
