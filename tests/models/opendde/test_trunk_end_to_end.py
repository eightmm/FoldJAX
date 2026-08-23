"""Run OpenDDE's residue trunk for real, on the released weights.

Every other test in this directory checks a signature, a shape rule, or a
monkeypatched stand-in. None of them runs the trunk, and that hole is not
theoretical: a change that removed four feature keys the trunk reads passed all
150 of them and then raised `KeyError: 'template_distogram'` on the first real
prediction. The read is at
`protenix/models/trunk_blocks/template.py:39`, reached through
`pairformer_output_from_s_inputs` -- OpenDDE's residue trunk is Protenix's, so
no amount of reading `models/opendde/` shows it.

This is the cheapest thing that would have failed: nine residues, one recycle,
stopping at the trunk so the diffusion sampler and the heads are skipped. About
ten seconds, almost all of it compilation. It asserts very little about the
values on purpose -- parity is `test_parity_matched_tape.py`'s job. What it
asserts is that the trunk *ran*, which is the property nothing else here has.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.api import predict
from foldjax.paths import weights_dir
from foldjax.schema import PredictionRequest

JOB = Path(__file__).resolve().parent / "examples" / "tiny.json"


def _weights() -> Path:
    path = weights_dir("opendde") / "opendde.jax"
    if not path.exists():
        pytest.skip(f"OpenDDE native weights are not in the store: {path}")
    return path


@pytest.mark.slow
def test_the_residue_trunk_runs_on_the_released_weights(tmp_path) -> None:
    """A missing feature the trunk reads fails here, and nowhere else.

    `stop_after="trunk"` is what makes this affordable: the residue trunk is
    where the shared Protenix modules are reached, and the diffusion sampler
    that dominates the runtime contributes nothing to that question.
    """
    _weights()

    result = predict(
        PredictionRequest(
            model="opendde",
            input=JOB,
            output_dir=tmp_path,
            seed=0,
            num_samples=1,
            num_recycles=1,
            stop_after="trunk",
            representations=("single", "pair"),
        )
    )
    assert result is not None

    written = tmp_path / "representations.npz"
    assert written.is_file(), (
        "the trunk produced no representations; "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )
    with np.load(written) as arrays:
        names = set(arrays.files)
        assert {"single", "pair"} <= names, names
        single, pair = np.asarray(arrays["single"]), np.asarray(arrays["pair"])

    # Shape rather than value: the point is that the trunk ran, not what it
    # concluded. `single` is per token, `pair` is per token pair.
    assert single.ndim >= 2 and pair.ndim >= 3, (single.shape, pair.shape)
    assert pair.shape[-2] == pair.shape[-3] == single.shape[-2], (
        single.shape,
        pair.shape,
    )
    assert np.all(np.isfinite(single)), "trunk single representation is not finite"
    assert np.all(np.isfinite(pair)), "trunk pair representation is not finite"
