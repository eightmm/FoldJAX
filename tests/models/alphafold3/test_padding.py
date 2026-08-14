from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.backends.alphafold3 import (
    _PREFIX_STABLE_NOISE_FEATURE,
    AlphaFold3Backend,
    _predict_featurized_structure,
    _prediction_buckets,
    _prepare_padded_jobs,
    _public_shape_profile,
    _resolved_token_padding_plan,
    _token_padding_plan,
    _validated_buckets,
)
from foldjax.padding import TOKEN_BUCKETS
from foldjax.schema import PaddingConfig, PredictionRequest


def _request(tmp_path: Path, **updates) -> PredictionRequest:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    values = {
        "model": "alphafold3",
        "input": input_path,
        "input_format": "native",
        "weights": weights,
        "output_dir": tmp_path / "out",
        "use_compile_cache": False,
    }
    values.update(updates)
    return PredictionRequest(**values)


@pytest.mark.parametrize("buckets", ([256, 256], [512, 256]))
def test_legacy_buckets_must_be_strictly_increasing_and_unique(buckets) -> None:
    with pytest.raises(ValueError, match="strictly increasing and unique"):
        _validated_buckets(buckets)


def test_default_off_keeps_af3_native_bucketing_disabled(tmp_path: Path) -> None:
    request = _request(tmp_path)
    options = {}

    assert _prediction_buckets(request, options) is None
    assert options == {}


def test_neutral_af3_padding_maps_auto_and_fixed_to_native_buckets(
    tmp_path: Path,
) -> None:
    automatic = _request(tmp_path, padding=PaddingConfig())
    fixed = _request(tmp_path, padding=PaddingConfig(tokens=1024))

    assert _prediction_buckets(automatic, {}) == TOKEN_BUCKETS
    assert _prediction_buckets(fixed, {}) == (1024,)


def test_neutral_and_legacy_af3_buckets_conflict(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        padding=PaddingConfig(),
        options={"buckets": [256, 512]},
    )

    with pytest.raises(ValueError, match="were both set"):
        AlphaFold3Backend().validate_request(request)


def test_neutral_padding_rejects_an_unpatched_external_af3_source(
    tmp_path: Path,
) -> None:
    request = _request(
        tmp_path,
        padding=PaddingConfig(),
        options={"source": tmp_path / "external"},
    )

    with pytest.raises(ValueError, match="managed runtime"):
        AlphaFold3Backend().validate_request(request)


def test_af3_reports_the_concrete_selected_token_shape() -> None:
    inference_result = SimpleNamespace(
        metadata={"token_chain_ids": ["A"] * 300}
    )
    results = (
        SimpleNamespace(inference_results=(inference_result,)),
    )

    plan = _token_padding_plan(results, TOKEN_BUCKETS)

    assert plan is not None
    assert plan.summary() == {
        "actual": {"tokens": 300},
        "storage": {"tokens": 300},
        "target": {"tokens": 512},
        "changed": True,
    }


def test_af3_rejects_a_too_small_fixed_bucket_before_model_execution() -> None:
    with pytest.raises(ValueError, match="exceeds the requested"):
        _resolved_token_padding_plan(
            actual=513,
            target=513,
            buckets=(512,),
            overflow="error",
            fixed_target=True,
        )


def test_af3_fixed_bucket_cannot_use_exact_overflow_to_shrink_input() -> None:
    with pytest.raises(ValueError, match="exceeds the requested"):
        _resolved_token_padding_plan(
            actual=513,
            target=513,
            buckets=(512,),
            overflow="exact",
            fixed_target=True,
        )


def test_af3_exact_overflow_reports_the_unpadded_native_shape() -> None:
    plan = _resolved_token_padding_plan(
        actual=5000,
        target=5000,
        buckets=TOKEN_BUCKETS,
        overflow="exact",
    )

    assert plan.summary() == {
        "actual": {"tokens": 5000},
        "storage": {"tokens": 5000},
        "target": {"tokens": 5000},
        "changed": False,
    }


def test_af3_preflights_every_neutral_job_before_returning_work(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def fake_featurize(fold_input, *, buckets, overflow, fixed_target=False):
        calls.append(fold_input)
        if fold_input == "late-overflow":
            raise ValueError("late overflow")
        return (f"features-{fold_input}",), SimpleNamespace(name=fold_input)

    monkeypatch.setattr(
        "foldjax.backends.alphafold3._featurize_padded_structure",
        fake_featurize,
    )
    jobs = (
        ("first", "first", tmp_path / "first"),
        ("late-overflow", "late", tmp_path / "late"),
    )

    with pytest.raises(ValueError, match="late overflow"):
        _prepare_padded_jobs(jobs, buckets=(256,), overflow="error")

    assert calls == ["first", "late-overflow"]


def test_padded_runner_selects_prefix_noise_without_changing_extraction_batch() -> None:
    seen = {}
    example = {"seq_length": np.asarray(3), "seq_mask": np.ones(8)}
    inference_result = SimpleNamespace(metadata={})

    class ModelRunner:
        def run_inference(self, model_batch, key):
            seen["model_batch"] = model_batch
            seen["key"] = key
            return {"result": np.asarray(1)}

        def extract_inference_results(self, *, batch, result, target_name):
            seen["extraction_batch"] = batch
            return (inference_result,)

        def extract_embeddings(self, *, result, num_tokens):
            return None

        def extract_distogram(self, *, result, num_tokens):
            return None

    runner = SimpleNamespace(
        ResultsForSeed=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    fold_input = SimpleNamespace(rng_seeds=(17,), name="job")

    results = _predict_featurized_structure(
        fold_input, (example,), ModelRunner(), runner
    )

    assert len(results) == 1
    assert bool(seen["model_batch"][_PREFIX_STABLE_NOISE_FEATURE])
    assert _PREFIX_STABLE_NOISE_FEATURE not in seen["extraction_batch"]


def test_legacy_buckets_do_not_change_the_public_result_envelope(
    tmp_path: Path,
) -> None:
    legacy = _request(tmp_path, options={"buckets": [512]})
    plan = {
        "actual": {"tokens": 300},
        "storage": {"tokens": 300},
        "target": {"tokens": 512},
        "changed": True,
    }

    assert _public_shape_profile(legacy, [plan]) is None


def test_af3_capability_and_cache_identity_separate_policy_from_legacy(
    tmp_path: Path,
) -> None:
    backend = AlphaFold3Backend()
    neutral = _request(tmp_path, padding=PaddingConfig(tokens=512))
    legacy = _request(tmp_path, options={"buckets": [512]})

    assert backend.capabilities().padding_axes == ("tokens",)
    assert "buckets" not in backend.cache_profile(neutral)
    assert backend.cache_profile(legacy)["buckets"] == [512]
