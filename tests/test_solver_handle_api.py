"""Unconfirmed CUDA library errors must survive the common API as non-OOMs."""

from __future__ import annotations

import dataclasses

import pytest

import foldjax
from foldjax import oom, warm_buckets
from foldjax.backends.base import Backend
from foldjax.registry import backend_override
from foldjax.schema import ModelCapabilities, PredictionRequest

_HANDLE_ERROR = "jaxlib/gpu/solver_handle_pool.cc:37: gpusolverDnCreate(&handle) failed"


def _failing_backend(error_type, *, explicit_oom):
    class HandleFailureBackend(Backend):
        name = "boltz2"

        def capabilities(self):
            return ModelCapabilities(model=self.name, input_formats=("native",))

        def predict(self, request):
            detail = _HANDLE_ERROR
            if explicit_oom:
                detail += ": CUDA_ERROR_OUT_OF_MEMORY"
            raise error_type(detail)

    return HandleFailureBackend


def _request(tmp_path, *, use_compile_cache=False):
    source = tmp_path / "native.json"
    source.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model="boltz2",
        input=source,
        input_format="native",
        weights=weights,
        seed=101,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache" if use_compile_cache else None,
        use_compile_cache=use_compile_cache,
    )


@pytest.mark.parametrize("error_type", [RuntimeError, SystemExit])
@pytest.mark.parametrize("explicit_oom", [False, True])
def test_predict_preserves_solver_oom_evidence_in_its_final_exception(
    tmp_path, error_type, explicit_oom
):
    backend = _failing_backend(error_type, explicit_oom=explicit_oom)
    expected_type = MemoryError if explicit_oom else foldjax.PredictionError
    with backend_override("boltz2", backend), pytest.raises(expected_type) as failure:
        foldjax.predict(_request(tmp_path))

    error = failure.value
    assert isinstance(error.__cause__, error_type)
    assert "outside the allocator's pool" in str(error)
    assert ("ran out of memory" in str(error)) is explicit_oom
    if not explicit_oom:
        assert not isinstance(error, MemoryError)
        assert not oom.is_out_of_memory(error)
        assert _HANDLE_ERROR in str(error)
    assert not (tmp_path / "out/foldjax_run.json").exists()


@pytest.mark.parametrize("error_type", [RuntimeError, SystemExit])
@pytest.mark.parametrize("explicit_oom", [False, True])
def test_bucket_warming_stops_only_for_confirmed_post_api_solver_oom(
    tmp_path, monkeypatch, error_type, explicit_oom
):
    backend = _failing_backend(error_type, explicit_oom=explicit_oom)
    failures = []

    def predict_through_common_api(request):
        try:
            return foldjax.predict(
                dataclasses.replace(
                    request,
                    input_format="native",
                    padding=None,
                    cache_dir=None,
                    use_compile_cache=False,
                )
            )
        except Exception as error:
            failures.append(error)
            raise

    monkeypatch.setattr(warm_buckets, "warm_cache", predict_through_common_api)
    monkeypatch.setattr(
        warm_buckets, "device_summary", lambda *_args: {"platform": "cpu"}
    )
    with backend_override("boltz2", backend):
        report = warm_buckets.warm_token_buckets(
            _request(tmp_path, use_compile_cache=True), "8,16", stop_on_oom=True
        )

    model_report = report["models"][0]
    assert model_report["stopped_on_oom"] is explicit_oom
    assert len(model_report["buckets"]) == (1 if explicit_oom else 2)
    assert len(failures) == (1 if explicit_oom else 2)
    assert all(row["oom"] is explicit_oom for row in model_report["buckets"])
    assert all(isinstance(error, MemoryError) is explicit_oom for error in failures)
    assert all(isinstance(error.__cause__, error_type) for error in failures)
    assert all("outside the allocator's pool" in str(error) for error in failures)
