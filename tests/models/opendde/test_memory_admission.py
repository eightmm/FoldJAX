"""Admission must be silent on jobs that run and refuse the ones that do not.

This file used to test `_preflight_arena`, a warning built on the temp-arena
law. The arena is about 91% of the peak and a warning cannot stop a run, so
the estimate is now a `memory_policy.PeakLaw` over the whole peak and the
verdict goes through `memory_policy.admit`: refuse, warn, or `unknown`.

The sizes are the ones this port was bracketed against by measurement:
1,003 residues (1,902 structural tokens) runs in both dtypes, 1,531 residues
(2,978) runs in bfloat16 and OOMs in float32 on a 95.6 GiB card, and 2,096
residues (about 4,034) runs on no layout but the 2x2 grid.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from foldjax import memory_policy
from foldjax.backends.opendde import _PARSER_DEFAULTS
from foldjax.models.opendde import runner as predict_runner

#: What `foldjax predict` asks the allocator for on this card, and the card.
POOL_BYTES = int(0.9 * 95.6 * 2**30)
CARD_BYTES = int(95.6 * 2**30)

N_ST_1003, N_ST_1531, N_ST_2096 = 1902, 2978, 4034

BF16 = np.dtype("bfloat16") if hasattr(np, "bfloat16") else object()


def _features(n_structural: int, *, n_residue: int = 8) -> dict[str, np.ndarray]:
    return {
        "structural_token_index": np.arange(n_structural, dtype=np.int64),
        "restype": np.zeros((1, n_residue), dtype=np.int64),
    }


@pytest.fixture(autouse=True)
def _forget_one_time_warnings():
    """`unknown` warns once per model per process, so each test starts clean."""
    memory_policy.reset_warnings()
    memory_policy.clear_record()
    yield
    memory_policy.reset_warnings()
    memory_policy.clear_record()


@pytest.fixture
def pool(monkeypatch):
    """Pin the device budget so the test does not need a GPU."""
    from foldjax import oom

    monkeypatch.setattr(oom, "device_budget", lambda: (POOL_BYTES, CARD_BYTES))
    return POOL_BYTES


def _admit(n_structural: int, trunk_dtype, **kwargs):
    """What `run_prediction` does, at the one seam worth calling directly."""
    bf16 = trunk_dtype is not None
    return memory_policy.admit(
        model="opendde",
        n_token=n_structural,
        msa_rows=None,
        candidates=(
            ("bf16 trunk", memory_policy.OPENDDE_BF16_PEAK)
            if bf16
            else ("fp32 trunk", memory_policy.OPENDDE_FP32_PEAK),
        ),
        budget=memory_policy.device_memory_budget(),
        levers=(
            predict_runner._BF16_TRUNK_LEVER
            if bf16
            else predict_runner._FP32_TRUNK_LEVER,
        ),
        token_label="structural tokens",
        **kwargs,
    )


@pytest.mark.parametrize(
    "n_structural,trunk_dtype",
    [
        (N_ST_1003, None),  # 1,003 residues, float32 -- measured 43,090 MiB, runs
        (N_ST_1003, BF16),  # 1,003 residues, bfloat16 -- measured 21,492 MiB, runs
        (N_ST_1531, BF16),  # 1,531 residues, bfloat16 -- measured 46,858 MiB, runs
    ],
)
def test_silent_when_the_job_fits(pool, n_structural, trunk_dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decision = _admit(n_structural, trunk_dtype)
    assert decision.state == "fits"


def test_refuses_on_the_float32_wall(pool):
    """1,531 residues in float32 is the measured OOM: it asked 93.40 GiB.

    A warning is what this used to be; a refusal is what the measurement
    supports, and the dtype lever is what the refusal has to name.
    """
    with pytest.raises(MemoryError) as error:
        _admit(N_ST_1531, None)
    message = str(error.value)
    assert f"{N_ST_1531} structural tokens" in message
    assert "--trunk-dtype bf16" in message
    # The lever carries its own evidence, not just its name.
    assert "43,090 MiB against 21,492" in message
    assert "upstream's own trunk precision" in message
    assert "--memory-check=warn" in message


def test_the_size_that_only_a_mesh_runs_is_refused_with_the_mesh_named(pool):
    """2,096 residues under bfloat16: the dtype lever is spent, cards are not.

    This is the case the whole law is for. Serial, 1-D on two cards and 1-D on
    four all died here; the 2x2 grid completed it at 32,068 MiB per device.
    """
    with pytest.raises(MemoryError) as error:
        _admit(N_ST_2096, BF16)
    message = str(error.value)
    assert "--trunk-dtype bf16" not in message
    assert "already bfloat16" in message
    assert "--cp-devices 4" in message


def test_warn_runs_it_anyway_and_says_so(pool):
    with pytest.warns(RuntimeWarning) as caught:
        decision = _admit(N_ST_2096, BF16, mode="warn")
    assert decision.state == "over_budget"
    assert "Running anyway" in str(caught[0].message)


def test_unknown_without_a_pool(pool, monkeypatch):
    """No readable ceiling means nothing was compared; the run proceeds.

    Inventing a ceiling is worse than saying so, and this is the shape a CPU
    run takes -- `device_budget` answers `(None, None)` on every platform
    that reports no memory statistics.
    """
    from foldjax import oom

    monkeypatch.setattr(oom, "device_budget", lambda: (None, None))
    with pytest.warns(RuntimeWarning, match="proceeds without the check"):
        decision = _admit(N_ST_2096, BF16)
    assert decision.state == "unknown"


def test_a_size_below_the_fitted_domain_is_unknown_and_never_refused(pool):
    """The parity sizes. A refusal from an extrapolation is the worst outcome.

    The float32 arm has no intercept -- one fitted size determines one
    parameter -- so below its domain its estimate is not a number to stop a
    run on, and `covers` is what keeps it from being used as one.
    """
    for n_structural in (257, 948):
        memory_policy.reset_warnings()
        with pytest.warns(RuntimeWarning, match="outside the fitted range"):
            decision = _admit(n_structural, None)
        assert decision.state == "unknown"
        # The bf16 law was checked against 144 completed runs down to 246
        # structural tokens (`admits_from`), so it may say "fits" there -- and
        # still never refuses: a budget it does not fit is `unknown`.
        assert _admit(n_structural, BF16).state == "fits"
    assert memory_policy.OPENDDE_BF16_PEAK.admits_from == 246
    memory_policy.reset_warnings()
    with pytest.warns(RuntimeWarning, match="outside the fitted range"):
        assert _admit(245, BF16).state == "unknown"


def test_a_mesh_run_is_warned_rather_than_refused(pool, monkeypatch, tmp_path):
    """The refusal must not refuse the run that takes its own advice.

    The bfloat16 lever names `--cp-devices 4`, because the 2x2 grid is the one
    layout measured to complete 2,096 residues. The laws are fitted on the
    serial arena, which a mesh splits, so a distributed run keeps the
    estimate -- it is still the best number available -- and loses the
    refusal. Without this the port would print "use four cards" and then
    refuse the four-card run.
    """
    job = {"name": "tiny", "modelSeeds": [101]}
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    document = tmp_path / "tiny.json"
    document.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(predict_runner, "_load_jobs", lambda _path: [job])
    monkeypatch.setattr(
        predict_runner,
        "_featurize",
        lambda _job, **_kwargs: _features(N_ST_2096, n_residue=2096),
    )

    class _StopError(Exception):
        pass

    def _stop(*_args, **_kwargs):
        raise _StopError

    monkeypatch.setattr(predict_runner, "_predict", _stop)

    fields = dict(_PARSER_DEFAULTS)
    fields.update(
        input_json=document,
        weights=weights,
        out=tmp_path / "out",
        trunk_dtype="bf16",
        confidence_dtype="fp32",
        diffusion_dtype="fp32",
        cp_devices=4,
    )
    config = predict_runner.PredictionConfig(**fields)

    with pytest.warns(RuntimeWarning, match="advisory rather than binding"):
        with pytest.raises(_StopError):
            predict_runner.run_prediction(
                config, _prepared_params_loader=lambda *_args: object()
            )


def test_no_structural_token_count_means_no_admission(pool):
    """A job whose features cannot say how big it is gets no verdict."""
    assert predict_runner._structural_token_count({}) is None
    assert predict_runner._structural_token_count(
        {"structural_token_index": np.zeros((0,), dtype=np.int64)}
    ) is None
    assert predict_runner._structural_token_count(_features(N_ST_1003)) == N_ST_1003


def test_the_estimates_match_the_measured_peaks(pool):
    """The laws must still reproduce the measurements they were fitted to."""
    for n_st, dtype, measured in (
        (N_ST_1003, BF16, 21492.0),
        (N_ST_1531, BF16, 46858.2),
        (N_ST_1003, None, 42690.6),
    ):
        law = (
            memory_policy.OPENDDE_BF16_PEAK
            if dtype is not None
            else memory_policy.OPENDDE_FP32_PEAK
        )
        assert law.estimate(n_st) / 2**20 == pytest.approx(measured, rel=1e-4), (
            n_st,
            law.calibration_id,
        )


def test_the_refusal_actually_comes_out_of_the_runner(pool, monkeypatch, tmp_path):
    """The laws being right is not the same as the call site firing.

    Everything above goes through `memory_policy.admit` directly. This one
    drives `run_prediction`, which is where a user meets it, and pins three
    things the direct tests cannot: that the runner admits at all, that it
    reads the job's own structural token count, and that it refuses *before*
    the first trace -- the seam below is `_predict`, and reaching it is the
    failure this test is guarding against.
    """
    job = {"name": "tiny", "modelSeeds": [101]}
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    document = tmp_path / "tiny.json"
    document.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(predict_runner, "_load_jobs", lambda _path: [job])
    monkeypatch.setattr(
        predict_runner,
        "_featurize",
        lambda _job, **_kwargs: _features(N_ST_2096, n_residue=2096),
    )
    monkeypatch.setattr(
        predict_runner,
        "project_generated_output_features",
        lambda features: features,
        raising=False,
    )

    def _never(*_args, **_kwargs):
        raise AssertionError("the run reached the model after being refused")

    monkeypatch.setattr(predict_runner, "_predict", _never)

    fields = dict(_PARSER_DEFAULTS)
    fields.update(
        input_json=document,
        weights=weights,
        out=tmp_path / "out",
        # The released trunk dtype, which selects the bfloat16 law; the other
        # two are pinned to fp32 only so the stand-in parameter tree is never
        # handed to a cast it has no fields for.
        trunk_dtype="bf16",
        confidence_dtype="fp32",
        diffusion_dtype="fp32",
    )
    config = predict_runner.PredictionConfig(**fields)

    with pytest.raises(MemoryError, match="4034 structural tokens"):
        predict_runner.run_prediction(
            config, _prepared_params_loader=lambda *_args: object()
        )


@pytest.mark.parametrize("implicit", [True, False])
def test_the_runner_admits_the_padded_shape_before_padding(
    pool, monkeypatch, tmp_path, implicit
):
    """5DEI: 4,040 structural tokens are measured to complete, and the default
    padding pads them to 4,096, which the law refuses. The runner weighs the
    padded shape against the exact one before anything is padded, so the
    default steps aside and the job runs at 4,040; padding asked for by name
    is refused, with the exact shape named in the refusal.
    """
    from foldjax.schema import DEFAULT_PADDING, PaddingConfig

    job = {"name": "tiny", "modelSeeds": [101]}
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    document = tmp_path / "tiny.json"
    document.write_text("[]", encoding="utf-8")
    features = _features(4040, n_residue=2096)
    features["parent_residue_idx"] = np.repeat(np.arange(2096), 2)[:4040]
    features["ref_pos"] = np.zeros((31_000, 3), dtype=np.float32)

    monkeypatch.setattr(predict_runner, "_load_jobs", lambda _path: [job])
    monkeypatch.setattr(predict_runner, "_featurize", lambda _job, **_kw: features)
    monkeypatch.setattr(
        predict_runner,
        "project_generated_output_features",
        lambda features: features,
        raising=False,
    )

    class _StopError(Exception):
        pass

    seen: dict[str, object] = {}

    def _stop(model_features, *_args, **_kwargs):
        seen["n_structural"] = predict_runner._structural_token_count(model_features)
        raise _StopError

    monkeypatch.setattr(predict_runner, "_predict", _stop)

    fields = dict(_PARSER_DEFAULTS)
    fields.update(
        input_json=document,
        weights=weights,
        out=tmp_path / "out",
        trunk_dtype="bf16",
        confidence_dtype="fp32",
        diffusion_dtype="fp32",
    )
    config = predict_runner.PredictionConfig(**fields)
    padding = DEFAULT_PADDING if implicit else PaddingConfig()
    profiles: list[dict[str, object]] = []

    if implicit:
        with pytest.warns(UserWarning, match="uses the exact input shapes"):
            with pytest.raises(_StopError):
                predict_runner.run_prediction(
                    config,
                    padding=padding,
                    padding_profiles=profiles,
                    _prepared_params_loader=lambda *_args: object(),
                )
        assert seen["n_structural"] == 4040
        assert profiles == [{"disabled_by": "memory admission"}]
        assert memory_policy.padding_disabled_by() == memory_policy.PADDING_STEP_ASIDE
        assert memory_policy.recorded()["n_token"] == 4040
    else:
        with pytest.raises(MemoryError, match="4096 structural tokens") as info:
            predict_runner.run_prediction(
                config,
                padding=padding,
                padding_profiles=profiles,
                _prepared_params_loader=lambda *_args: object(),
            )
        assert "--no-padding (padding=False) runs the exact 4040" in str(info.value)
        assert seen == {}
        assert memory_policy.padding_disabled_by() is None


def test_the_step_aside_is_decided_per_job_in_one_call(pool, monkeypatch, tmp_path):
    """Three jobs, 2,000 -> 4,040 -> 2,000 structural tokens, in one call. The
    shape that fits padded runs padded both times, the middle one steps aside,
    and the third job -- a cache hit on the first job's verdict -- neither
    inherits the step-aside marker nor leaves it set for the adapter. Every job
    reports an entry: its profile, or the step-aside record.
    """
    from types import SimpleNamespace

    from foldjax.backends.opendde import _shape_profile
    from foldjax.models.opendde.data import padding as padding_module
    from foldjax.schema import DEFAULT_PADDING

    jobs = [{"name": name, "modelSeeds": [101]} for name in ("a", "b", "c")]
    sizes = {"a": 2000, "b": 4040, "c": 2000}

    def featurize(job, **_kwargs):
        features = _features(sizes[job["name"]], n_residue=2096)
        features["parent_residue_idx"] = np.arange(sizes[job["name"]])
        features["ref_pos"] = np.zeros((31_000, 3), dtype=np.float32)
        return features

    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    document = tmp_path / "tiny.json"
    document.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(predict_runner, "_load_jobs", lambda _path: jobs)
    monkeypatch.setattr(predict_runner, "_featurize", featurize)
    monkeypatch.setattr(
        predict_runner,
        "project_generated_output_features",
        lambda features: features,
        raising=False,
    )
    # The padded route, without padding anything: this test is about the
    # decision and what it reports, not the features.
    summary = {"target": {"structural_tokens": 2048}}
    plan = SimpleNamespace(
        actual={"atoms": 1},
        target={"atoms": 1},
        summary=lambda: dict(summary),
        message=lambda model: f"{model}: padded",
    )
    monkeypatch.setattr(
        padding_module,
        "pad_opendde_features",
        lambda features, sampled, padding, **_kwargs: (features, None, plan),
    )
    monkeypatch.setattr(padding_module, "select_opendde_model_features", lambda f: f)
    monkeypatch.setattr(
        "foldjax.models.opendde.models.msa_sampling.sample_opendde_msa_cycle_features",
        lambda features, **_kwargs: None,
    )
    monkeypatch.setattr(predict_runner, "_prefix_rng_is_supported", lambda: True)
    monkeypatch.setattr(predict_runner, "_collect_representations", lambda *_a: {})
    monkeypatch.setattr(
        predict_runner._representations, "save", lambda *_args, **_kwargs: None
    )
    seen: list[tuple[int | None, bool]] = []

    def predict(model_features, *_args, **kwargs):
        seen.append(
            (
                predict_runner._structural_token_count(model_features),
                kwargs.get("preserve_prefix_rng", False),
            )
        )
        return {}

    monkeypatch.setattr(predict_runner, "_predict", predict)

    fields = dict(_PARSER_DEFAULTS)
    fields.update(
        input_json=document,
        weights=weights,
        out=tmp_path / "out",
        trunk_dtype="bf16",
        confidence_dtype="fp32",
        diffusion_dtype="fp32",
        # Returns after the model call, before any output is written; unlike
        # `inputs`, the trunk stage keeps the pair-stack peak and so the refusal.
        stop_after="trunk",
    )
    config = predict_runner.PredictionConfig(**fields)
    profiles: list[dict[str, object]] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predict_runner.run_prediction(
            config,
            padding=DEFAULT_PADDING,
            padding_profiles=profiles,
            _prepared_params_loader=lambda *_args: object(),
        )

    assert seen == [(2000, True), (4040, False), (2000, True)]
    step_aside = {"disabled_by": "memory admission"}
    assert profiles == [summary, step_aside, summary]
    # Once per shape: the third job reused the first job's verdict.
    notices = [w for w in caught if "uses the exact input shapes" in str(w.message)]
    assert len(notices) == 1
    # The last job ran padded, so the marker the adapter reads is clear ...
    assert memory_policy.padding_disabled_by() is None
    # ... and the adapter's record keeps every job's outcome.
    assert _shape_profile(profiles, padded=True) == {
        "per_run": [summary, step_aside, summary]
    }
    assert _shape_profile([step_aside, step_aside], padded=False) is None
