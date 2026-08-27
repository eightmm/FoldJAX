"""ESMFold2 stops returning confidence logits nothing reads.

`write_prediction_outputs` consumes three arrays and four scalars. The
confidence head returns seven more, and no path hands them to a caller: the
backend takes its scores from the writer's summary, `representations` offers
only `single` and `pair`, and `raw` carries overrides, language-model presence
and padding. Measured entry outputs at 1,003 tokens were 2,748 MiB at 5 samples
and 16,264 MiB at 32, against 2,745 and 16,261 MiB of exactly these arrays
computed from their shapes -- they are the whole term.

The saving is in *not being an entry output of the compiled program*, which XLA
cannot eliminate however unread it is. That is why the flag is a static setting
and why the test that guards the bytes has to be compile-only and lives with
the arena probes. What is checkable without a GPU is everything else: that the
writer still gets what it reads, that the scores survive, and that withholding
the arrays did not perturb the arithmetic that produced them.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from foldjax.models.esmfold2.bridge import checkpoint
from foldjax.models.esmfold2.models import model as jax_model
from foldjax.models.esmfold2.output import SAMPLE_SCORES

from .test_checkpoint_load import N_TOKENS, _features, _weights

#: What the writer reads, and therefore what must survive the flag.
_WRITER_READS = ("sample_atom_coords", "plddt_per_atom", "plddt")


def _settings(directory, *, logits: bool):
    return dataclasses.replace(
        jax_model.with_overrides(
            checkpoint.load_settings(directory),
            num_recycles=1,
            num_samples=2,
            num_steps=1,
        ),
        return_confidence_logits=logits,
    )


def _run(directory, parameters, *, logits: bool):
    return jax_model.predict(
        jax.random.key(0),
        _features(),
        parameters,
        settings=_settings(directory, logits=logits),
        lm_hidden_states=np.zeros((1, N_TOKENS, 81, 2560), dtype=np.float32),
    )


def test_the_flag_is_off_by_default() -> None:
    """Mirrors Boltz-2's `return_confidence_logits`, name and default both."""
    assert jax_model.ModelSettings().return_confidence_logits is False


@pytest.mark.slow
def test_withholding_the_logits_changes_nothing_the_writer_reads() -> None:
    directory = _weights()
    parameters = checkpoint.load_parameters(directory)

    without = _run(directory, parameters, logits=False)
    with_them = _run(directory, parameters, logits=True)

    # Gone when off, present when on -- and absent means absent, not zeroed.
    for name in jax_model.CONFIDENCE_LOGIT_OUTPUTS:
        assert name not in without, f"{name} still returned with the flag off"
        assert name in with_them, f"{name} missing with the flag on"

    # Everything the writer and the scores need survives, bit for bit. `ptm`
    # is the load-bearing one: it is computed *from* `pae_logits`, so if
    # withholding the array had disturbed the arithmetic it would move here.
    for name in (*_WRITER_READS, *SAMPLE_SCORES):
        assert name in without, f"{name} was dropped and the writer needs it"
        np.testing.assert_array_equal(
            np.asarray(without[name]),
            np.asarray(with_them[name]),
            err_msg=f"{name} changed when the logits were withheld",
        )

    # `distogram_logits` is deliberately outside the flag: 245 MiB, flat in
    # sample count, and `test_model_parity` reads it.
    assert "distogram_logits" in without


@pytest.mark.slow
def test_the_writer_still_produces_structures_and_scores(tmp_path) -> None:
    """End to end: the flag must not strand the output path."""
    from foldjax.models.esmfold2 import output as output_module

    directory = _weights()
    parameters = checkpoint.load_parameters(directory)
    prediction = _run(directory, parameters, logits=False)

    written = output_module.write_prediction_outputs(
        prediction, _features(), tmp_path, name="flag"
    )
    assert len(written["structures"]) == 2
    assert written["scores"].is_file()
    for entry in written["summary"]:
        for name in SAMPLE_SCORES:
            assert name in entry, f"{name} missing from the confidence JSON"
        assert 0.0 <= entry["plddt"] <= 1.0
