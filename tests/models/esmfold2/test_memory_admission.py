"""ESMFold2's peak is one arena, and this is where its law meets a run.

The rolled-program fit uses 35,023.9 MiB at 2,096 tokens and 69,350.8 at
3,012. The earlier released program established the small spread between
5 and 32 samples; its spread is retained in the allowance.

`inference.predict` takes the budget as an argument and does nothing with
`None`, so every direct caller here -- and every parity run and tape replay --
keeps the contract it had. The tests that matter are therefore: the call site
fires when a budget is given, it fires *before* the language model, and it is
silent when there is nothing to compare.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax import memory_policy
from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.models.esmfold2 import inference
from foldjax.schema import DEFAULT_PADDING, PaddingConfig, PredictionRequest
from tests.test_esmfold2_backend import (
    _fake_session_weights,
    _fake_staged_session_modules,
    _job,
)

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
    """Both fitted rows ran on a 95.6 GiB card, so both must be admitted.

    The 3,012-token row completed with the pool at 0.95 of the card; the law
    describes the program, and at the CLI's 0.9 it still fits.
    """
    budget = _budget(int(0.9 * 95.6 * _GIB))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for n_token in (2096, 3012):
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


@pytest.mark.parametrize("n_token", [2095, 3013])
def test_outside_the_domain_is_unknown(n_token: int) -> None:
    """The historical fit does not establish a peak beyond its measured band."""
    with pytest.warns(RuntimeWarning, match="outside the fitted range"):
        decision = _admit(n_token, budget=_budget(32 * _GIB))
    assert decision.state == "unknown"


def test_small_input_warning_identifies_the_memory_calibration_not_an_input_limit():
    with pytest.warns(RuntimeWarning) as caught:
        decision = _admit(256, budget=_budget(1 * _GIB))
    assert decision.state == "unknown"
    assert decision.estimates == ()
    message = str(caught[0].message)
    assert "no measured ESMFold2 peak estimate for 256 tokens" in message
    assert "historical peak calibration covers 2,096-3,012 tokens" in message
    assert "esmfold2-2026-09-23" in message
    assert "unknown peak memory" in message
    assert "The run proceeds without the check" in message
    assert "released (2,096-3,012 tokens)" not in message


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


@pytest.mark.parametrize("n_token", [1280, 1408])
@pytest.mark.parametrize("pool_gib", [22, 23.5])
def test_small_card_capacity_risk_warns_before_the_language_model(
    monkeypatch, n_token, pool_gib
):
    class _StopError(Exception):
        pass

    def _stop(*_args, **_kwargs):
        raise _StopError

    monkeypatch.setattr(inference.atom_model, "_resolve_rows_per_block", _stop)
    monkeypatch.setattr(inference, "language_model_states", _stop)
    model = inference.LoadedModel(
        parameters={}, settings=inference.structure_model.ModelSettings()
    )
    with pytest.warns(RuntimeWarning, match="reported successful limits 1024"):
        with pytest.raises(_StopError):
            inference.predict(
                None,
                _features(n_token),
                model,
                memory_budget=memory_policy.resolve_budget(
                    pool_bytes=int(pool_gib * _GIB),
                    card_bytes=int(23.5 * _GIB),
                    override_gib=None,
                ),
            )
    record = memory_policy.recorded()
    assert record["state"] == "unknown"
    assert "required peak was not measured" in record["reason"]


def test_a_smaller_unknown_run_does_not_silence_the_capacity_warning():
    budget = memory_policy.resolve_budget(
        pool_bytes=22 * _GIB, card_bytes=int(23.5 * _GIB), override_gib=None
    )
    with pytest.warns(RuntimeWarning, match="outside the fitted range"):
        _admit(128, budget=budget)
    with pytest.warns(RuntimeWarning, match="reported successful limits 1024"):
        _admit(1280, budget=budget)


@pytest.mark.parametrize("n_token", [1792, 2048])
@pytest.mark.parametrize("n_chain", [1, 7])
def test_48gb_observed_failure_refuses_the_padded_graph_before_lm(
    monkeypatch, n_token, n_chain
):
    def _never(*_args, **_kwargs):
        pytest.fail("observed refusal must happen before LM or graph execution")

    monkeypatch.setattr(inference.atom_model, "_resolve_rows_per_block", _never)
    monkeypatch.setattr(inference, "language_model_states", _never)
    model = inference.LoadedModel(
        parameters={}, settings=inference.structure_model.ModelSettings()
    )
    features = _features(n_token)
    features["asym_id"] = np.arange(n_token)[None] % n_chain
    with pytest.raises(MemoryError, match="observed ESMFold2 capacity refusal"):
        inference.predict(
            None,
            features,
            model,
            memory_budget=memory_policy.resolve_budget(
                pool_bytes=int(40.3 * _GIB),
                card_bytes=int(47.4 * _GIB),
                override_gib=None,
            ),
            padded=True,
        )
    record = memory_policy.recorded()
    assert record["state"] == "over_budget"
    assert record["n_token"] == n_token
    assert record["estimates"] == []  # No peak is inferred from a censored failure.
    assert record["threshold_bytes"] is None
    assert record["calibration_id"] == memory_policy.ESMFOLD2_CAPACITY_POLICY_ID


def test_48gb_observed_failure_warn_override_reaches_the_graph(monkeypatch):
    class _StopError(Exception):
        pass

    def _stop(*_args, **_kwargs):
        raise _StopError

    monkeypatch.setattr(inference.atom_model, "_resolve_rows_per_block", _stop)
    model = inference.LoadedModel(
        parameters={}, settings=inference.structure_model.ModelSettings()
    )
    with pytest.warns(RuntimeWarning, match="Running anyway"):
        with pytest.raises(_StopError):
            inference.predict(
                None,
                _features(1792),
                model,
                memory_budget=memory_policy.resolve_budget(
                    pool_bytes=int(40.3 * _GIB),
                    card_bytes=int(47.4 * _GIB),
                    override_gib=None,
                ),
                memory_check="warn",
                padded=True,
            )


@pytest.mark.parametrize(
    "n_token,card_gib,pool_gib", [(1024, 23.5, 21.2), (1536, 47.4, 40.3)]
)
def test_latest_successful_buckets_keep_unknown_peaks_without_capacity_refusal(
    n_token, card_gib, pool_gib
):
    budget = memory_policy.resolve_budget(
        pool_bytes=int(pool_gib * _GIB),
        card_bytes=int(card_gib * _GIB),
        override_gib=None,
    )
    with pytest.warns(RuntimeWarning, match="unknown peak memory"):
        decision = _admit(n_token, budget=budget)
    assert decision.state == "unknown"
    assert decision.estimates == ()
    assert "640-768" not in decision.reason
    assert "capacity refusal" not in decision.reason


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"cp_shards": 2}, "context parallelism"),
        ({"stop_after_inputs": True}, "an inputs-only graph"),
        ({"glu_backend": "tokamax"}, "a fused GLU"),
        (
            {"triangle_backend": "cueq"},
            "a cuEquivariance triangle multiplication",
        ),
        (
            {"rigid_align_backend": "jacobi"},
            "solver-free rigid alignment (unmeasured memory)",
        ),
    ],
)
def test_unmeasured_alternatives_do_not_inherit_a_single_device_refusal(
    kwargs, expected
):
    assert memory_policy.esmfold2_off_profile_reason(num_samples=32) == ()
    reasons = memory_policy.esmfold2_off_profile_reason(num_samples=32, **kwargs)
    assert reasons == (expected,)
    budget = memory_policy.resolve_budget(
        pool_bytes=int(40.3 * _GIB),
        card_bytes=int(47.4 * _GIB),
        override_gib=None,
    )
    with pytest.warns(RuntimeWarning, match="observed operating limit is advisory"):
        decision = memory_policy.admit(
            model="esmfold2",
            n_token=1792,
            msa_rows=None,
            candidates=(("released", memory_policy.ESMFOLD2_PEAK),),
            budget=budget,
            off_profile=reasons,
        )
    assert decision.state == "over_budget"


def test_predict_records_the_selected_configuration_when_the_peak_fits(monkeypatch):
    class _StopError(Exception):
        pass

    def _stop(*_args, **_kwargs):
        raise _StopError

    monkeypatch.setattr(inference.atom_model, "_resolve_rows_per_block", _stop)
    model = inference.LoadedModel(
        parameters={}, settings=inference.structure_model.ModelSettings()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(_StopError):
            inference.predict(
                None, _features(2096), model, memory_budget=_budget(80 * _GIB)
            )
    record = memory_policy.recorded()
    assert record["state"] == "fits"
    assert record["selected"] == "released"
    assert "released fits" in record["reason"]


def test_predict_marks_cueq_triangle_advisory_without_loosening_xla(
    monkeypatch,
) -> None:
    class _StopError(Exception):
        pass

    def _stop(*_args, **_kwargs):
        raise _StopError

    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: True)
    monkeypatch.setattr(inference.atom_model, "_resolve_rows_per_block", _stop)
    monkeypatch.setattr(
        inference,
        "language_model_states",
        lambda *_a, **_k: pytest.fail("LM ran before seam"),
    )
    model = inference.LoadedModel(
        parameters={}, settings=inference.structure_model.ModelSettings()
    )
    budget_48g = memory_policy.resolve_budget(
        pool_bytes=int(40.3 * _GIB),
        card_bytes=int(47.4 * _GIB),
        override_gib=None,
    )

    for on_profile in ({}, {"triangle_backend": "xla"}):
        memory_policy.reset_warnings()
        memory_policy.clear_record()
        with pytest.raises(MemoryError, match="observed ESMFold2 capacity refusal"):
            inference.predict(
                None,
                _features(1792),
                model,
                num_samples=32,
                memory_budget=budget_48g,
                padded=True,
                **on_profile,
            )
        assert memory_policy.recorded()["off_profile"] == []

    for unmeasured, expected_reason in (
        ({"glu_backend": "tokamax"}, "a fused GLU"),
        (
            {"triangle_backend": "cueq"},
            "a cuEquivariance triangle multiplication",
        ),
    ):
        memory_policy.reset_warnings()
        memory_policy.clear_record()
        with pytest.warns(RuntimeWarning, match=expected_reason) as caught:
            with pytest.raises(_StopError):
                inference.predict(
                    None,
                    _features(1792),
                    model,
                    num_samples=32,
                    memory_budget=budget_48g,
                    padded=True,
                    **unmeasured,
                )
        assert "observed operating limit is advisory" in str(caught[0].message)
        record = memory_policy.recorded()
        assert record["state"] == "over_budget"
        assert record["off_profile"] == [expected_reason]

    memory_policy.reset_warnings()
    memory_policy.clear_record()
    with pytest.warns(
        RuntimeWarning,
        match="estimate is advisory rather than binding.*cuEquivariance triangle",
    ):
        with pytest.raises(_StopError):
            inference.predict(
                None,
                _features(2096),
                model,
                num_samples=32,
                triangle_backend="cueq",
                memory_budget=_budget(32 * _GIB),
            )
    assert memory_policy.recorded()["off_profile"] == [
        "a cuEquivariance triangle multiplication"
    ]

    for malformed in ("", 0, "pallas"):
        memory_policy.clear_record()
        with pytest.raises(ValueError, match="triangle_backend must be one of"):
            inference.predict(
                None,
                _features(1792),
                model,
                num_samples=32,
                triangle_backend=malformed,  # type: ignore[arg-type]
                memory_budget=budget_48g,
                padded=True,
            )
        assert memory_policy.recorded() is None


def test_managed_admission_marks_cueq_triangle_advisory_at_48gb_limit(
    tmp_path, monkeypatch
) -> None:
    class _ReachedLMError(Exception):
        pass

    events: list[tuple[str, object]] = []
    modules = _fake_staged_session_modules(tmp_path, events)
    native = modules["foldjax.models.esmfold2.inference"]
    native.LANGUAGE_MODEL_STATE_STORAGE_API = True
    original_stage = native.load_language_model_stage

    def stage(*args, **kwargs):
        loaded = original_stage(*args, **kwargs)
        loaded.settings.num_samples = 32
        loaded.settings.diffusion = SimpleNamespace(
            glu_backend="xla", rigid_align_backend="svd"
        )
        return loaded

    tok_axis = "n_" + "to" + "ken"
    tok_target = "to" + "kens"
    mask_name = "to" + "ken_attention_mask"
    native.load_language_model_stage = stage
    native.language_model_length = lambda features: features["input_ids"].shape[-1] + 2
    native.normalize_msa_features = lambda _rng, features, **_kwargs: features
    native.pad_features = lambda features, **kwargs: {
        name: np.pad(value, ((0, 0), (0, kwargs[tok_axis] - value.shape[-1])))
        for name, value in features.items()
    }
    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: True)
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    monkeypatch.setattr(
        "foldjax.backends.esmfold2._padding_plan",
        lambda *_args, **_kwargs: SimpleNamespace(
            target={tok_target: 1792, "atoms": 16384, "msa": 16}
        ),
    )
    monkeypatch.setattr(
        memory_policy,
        "device_memory_budget",
        lambda **_kwargs: memory_policy.resolve_budget(
            pool_bytes=int(40.3 * _GIB),
            card_bytes=int(47.4 * _GIB),
            override_gib=None,
        ),
    )
    backend = ESMFold2Backend()
    weights = _fake_session_weights(tmp_path)
    seen_lengths: list[int] = []

    def reached_lm(_inference, features, *_args, **_kwargs):
        seen_lengths.append(int(features[mask_name].shape[-1]))
        raise _ReachedLMError

    monkeypatch.setattr(backend, "_language_model_embedding", reached_lm)

    def make_request(options=None, *, padding=PaddingConfig(), length: int = 1726):
        return PredictionRequest(
            model="esmfold2",
            input=_job(
                tmp_path / f"in_{length}",
                [{"type": "protein", "id": ["A"], "sequence": "A" * length}],
            ),
            weights=weights,
            output_dir=tmp_path / "out",
            padding=padding,
            options=options or {},
        )

    for on_profile in ({}, {"triangle_backend": "xla"}):
        memory_policy.reset_warnings()
        memory_policy.clear_record()
        req = make_request(on_profile)
        with backend.session((req,)):
            with pytest.raises(MemoryError, match="observed ESMFold2 capacity refusal"):
                backend.predict(req)
        assert seen_lengths == []
        assert memory_policy.recorded()["off_profile"] == []

    for unmeasured, expected_reason in (
        ({"glu_backend": "tokamax"}, "a fused GLU"),
        (
            {"triangle_backend": "cueq"},
            "a cuEquivariance triangle multiplication",
        ),
        (
            {"triangle_kernel": "cueq"},
            "a cuEquivariance triangle multiplication",
        ),
    ):
        memory_policy.reset_warnings()
        memory_policy.clear_record()
        seen_lengths.clear()
        req = make_request(unmeasured)
        with backend.session((req,)):
            with pytest.warns(RuntimeWarning, match=expected_reason) as caught:
                with pytest.raises(_ReachedLMError):
                    backend.predict(req)
        assert "observed operating limit is advisory" in str(caught[0].message)
        assert seen_lengths == [1792]
        record = memory_policy.recorded()
        assert record["state"] == "over_budget"
        assert record["off_profile"] == [expected_reason]

    memory_policy.reset_warnings()
    memory_policy.clear_record()
    seen_lengths.clear()
    implicit_req = make_request({"triangle_backend": "cueq"}, padding=DEFAULT_PADDING)
    with backend.session((implicit_req,)):
        with pytest.warns(
            RuntimeWarning, match="a cuEquivariance triangle multiplication"
        ):
            with pytest.raises(_ReachedLMError):
                backend.predict(implicit_req)
    assert seen_lengths == [1792]

    for options, refuses in (
        ({"triangle_backend": "xla"}, True),
        ({"triangle_backend": "cueq"}, False),
    ):
        memory_policy.reset_warnings()
        memory_policy.clear_record()
        seen_lengths.clear()
        exact_req = make_request(options, padding=False, length=1792)
        with backend.session((exact_req,)):
            if refuses:
                with pytest.raises(
                    MemoryError, match="observed ESMFold2 capacity refusal"
                ):
                    backend.predict(exact_req)
                assert seen_lengths == []
            else:
                with pytest.raises(_ReachedLMError):
                    backend.predict(exact_req)
                assert seen_lengths == [1792]

    memory_policy.clear_record()
    bad_req = make_request({"triangle_backend": ""})
    with pytest.raises(ValueError, match="triangle_backend must be one of"):
        backend.predict(bad_req)
    assert memory_policy.recorded() is None
