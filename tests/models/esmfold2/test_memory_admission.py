"""ESMFold2's peak is one arena, and this is where its law meets a run.

The two fitted rows are 14,733.3 MiB at 1,003 tokens and 46,041.8 at 2,096,
both at 5 samples; the released count is 32 and the same 2,096-token peak
reads 46,284.8 there, which is why the sample axis is not in the law and why
32 samples is inside its profile rather than off it.

`inference.predict` takes the budget as an argument and does nothing with
`None`, so every direct caller here -- and every parity run and tape replay --
keeps the contract it had. The tests that matter are therefore: the call site
fires when a budget is given, it fires *before* the language model, and it is
silent when there is nothing to compare.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from foldjax import memory_policy
from foldjax.models.esmfold2 import inference

_GIB = 2**30


@pytest.fixture(autouse=True)
def _forget_one_time_warnings():
    memory_policy.reset_warnings()
    memory_policy.clear_record()
    yield
    memory_policy.reset_warnings()
    memory_policy.clear_record()


def _budget(pool: int | None, *, override: float | None = None):
    return memory_policy.resolve_budget(
        pool_bytes=pool,
        card_bytes=None if pool is None else pool + 6 * _GIB,
        override_gib=override,
    )


def _features(n_token: int) -> dict[str, np.ndarray]:
    return {
        "token_attention_mask": np.ones((1, n_token), dtype=bool),
        "asym_id": np.zeros((1, n_token), dtype=np.int64),
    }


def _admit(n_token: int, *, budget, mode="refuse", num_samples=32, **kwargs):
    return memory_policy.admit(
        model="esmfold2",
        n_token=n_token,
        msa_rows=None,
        candidates=(("released", memory_policy.ESMFOLD2_PEAK),),
        budget=budget,
        mode=mode,
        off_profile=memory_policy.off_profile_reason(
            num_samples=num_samples, samples_validated=32
        ),
        **kwargs,
    )


def test_a_measured_row_fits_the_card_it_was_measured_on() -> None:
    """Both fitted rows ran inside a 95.6 GiB pool, so both must be admitted."""
    budget = _budget(int(0.9 * 95.6 * _GIB))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for n_token in (1003, 2096):
            assert _admit(n_token, budget=budget).state == "fits"


def test_the_released_sample_count_still_refuses() -> None:
    """32 samples is the released count; an advisory-only law never refuses.

    The peak moved 0.53% between 5 and 32 samples, so the estimate binds at
    both -- and this is the assertion that keeps the widening honest, because
    without it every default run would be warned rather than refused.
    """
    budget = _budget(32 * _GIB)
    with pytest.raises(MemoryError) as error:
        _admit(2096, budget=budget, num_samples=32)
    assert "2096 tokens" in str(error.value)
    assert "advisory rather than binding" not in str(error.value)

    # One sample, on the other hand, is below everything measured.
    with pytest.warns(RuntimeWarning, match="advisory rather than binding"):
        _admit(2096, budget=budget, num_samples=1)


def test_warn_proceeds_and_refuse_does_not() -> None:
    budget = _budget(32 * _GIB)
    with pytest.warns(RuntimeWarning) as caught:
        decision = _admit(2096, budget=budget, mode="warn")
    assert decision.state == "over_budget"
    assert "Running anyway" in str(caught[0].message)


@pytest.mark.parametrize("n_token", [1002, 2097])
def test_outside_the_domain_is_unknown(n_token: int) -> None:
    """3,012 tokens is censored, not fitted: the law reads 87.3 GiB there and
    the run's allocator asked 86 GiB before failing, but no completed run
    measured it, so the answer above 2,096 is `unknown`."""
    with pytest.warns(RuntimeWarning, match="outside the fitted range"):
        decision = _admit(n_token, budget=_budget(32 * _GIB))
    assert decision.state == "unknown"


def test_no_budget_is_no_check() -> None:
    with pytest.warns(RuntimeWarning, match="proceeds without the check"):
        decision = _admit(2096, budget=_budget(None))
    assert decision.state == "unknown"


def test_predict_without_a_budget_neither_admits_nor_warns(monkeypatch) -> None:
    """The contract every direct caller has: a budget of `None` means unasked.

    Asserted by reaching the seam right after the admission point with
    warnings promoted to errors -- a parity run, a tape replay and a unit
    test all arrive here, and none of them asked for a check.
    """

    class _StopError(Exception):
        pass

    def _stop(*_args, **_kwargs):
        raise _StopError

    monkeypatch.setattr(inference.atom_model, "_resolve_rows_per_block", _stop)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(_StopError):
            inference.predict(
                None,
                _features(2096),
                inference.LoadedModel(
                    parameters={},
                    settings=inference.structure_model.ModelSettings(),
                ),
            )
    assert memory_policy.recorded() is None


def test_predict_refuses_before_the_language_model(monkeypatch) -> None:
    """Admission has to sit above ESMC, which is a 3B encoder forward.

    The seam is `_resolve_rows_per_block`, the first statement after the
    admission call: reaching it would mean the refusal came too late, and
    `language_model_states` is further down still.
    """

    def _never(*_args, **_kwargs):
        raise AssertionError("the run got past a refused admission")

    monkeypatch.setattr(inference.atom_model, "_resolve_rows_per_block", _never)
    monkeypatch.setattr(inference, "language_model_states", _never)

    model = inference.LoadedModel(
        parameters={},
        settings=inference.structure_model.ModelSettings(),
    )
    with pytest.raises(MemoryError, match="2096 tokens"):
        inference.predict(
            None,
            _features(2096),
            model,
            memory_budget=_budget(32 * _GIB),
        )
