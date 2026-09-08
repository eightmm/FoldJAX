"""Common model handles expose stages/settings, never a native feature ABI."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import foldjax
from foldjax import ExecutionConfig, Job, ModelConfig, Protein, get_model
from foldjax.models import _representations
from foldjax.registry import get_backend
from foldjax.schema import PredictionRequest, PredictionResult


@pytest.fixture(autouse=True)
def input_files(tmp_path):
    (tmp_path / "job.json").write_text('{"name":"test","entities":[]}')
    (tmp_path / "weights").mkdir()


@pytest.mark.parametrize("model", foldjax.available_models())
@pytest.mark.parametrize("passes", [1, 4])
def test_total_passes_translate_to_existing_request_contract(tmp_path, model, passes):
    handle = get_model(
        model, weights=tmp_path / "weights", config=ModelConfig(trunk_passes=passes)
    )
    request = handle.plan(tmp_path / "job.json")
    expected = passes if model in {"protenix", "opendde"} else passes - 1
    assert request.num_recycles == expected
    native = get_backend(model).apply_sampling(request)["num_recycles"]
    assert native == (
        passes if model in {"protenix", "opendde", "openfold3"} else passes - 1
    )


@pytest.mark.parametrize("model", foldjax.available_models())
def test_all_models_advertise_real_input_stage_and_resolve_all(tmp_path, model):
    handle = get_model(model, weights=tmp_path / "weights")
    assert handle.capabilities.input_representations == ("single_inputs",)
    request = handle.plan(tmp_path / "job.json", stage="inputs", outputs="all")
    assert request.stop_after == "inputs"
    assert request.representations == ("single_inputs",)
    with pytest.raises(ValueError, match="representation"):
        handle.plan(tmp_path / "job.json", stage="inputs", outputs="pair")
    with pytest.raises(ValueError, match="representation"):
        handle.plan(tmp_path / "job.json", stage="inputs", outputs="all,pair")


def test_model_padding_requires_explicit_scientific_depth(tmp_path):
    with pytest.raises(ValueError, match="explicit.*msa_depth"):
        get_model("boltz2", execution=ExecutionConfig(padding=True))
    handle = get_model(
        "boltz2",
        config=ModelConfig(msa_depth=1024),
        execution=ExecutionConfig(padding=True),
    )
    assert handle.config.msa_depth == 1024
    assert handle.execution.padding is not None
    # Preserve the older interface's serving preset.
    request = PredictionRequest(
        model="boltz2", input=tmp_path / "job.json", padding=True
    )
    assert get_backend("boltz2").apply_sampling(request)["max_msa_depth"] == 1024


@pytest.mark.parametrize("field", ["msa_depth", "samples", "steps", "trunk_passes"])
@pytest.mark.parametrize("value", [True, -1, 0, 1.5, "4"])
def test_config_rejects_invalid_counts(field, value):
    with pytest.raises(ValueError):
        ModelConfig(**{field: value})


def test_get_model_is_lazy_and_immutable(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("get_model must not resolve/run a prediction")

    monkeypatch.setattr(foldjax.api, "resolve_request", forbidden)
    first = get_model("af3")
    second = first.with_config(config=ModelConfig(trunk_passes=4))
    assert first.name == "alphafold3"
    assert first.config.trunk_passes is None
    assert second.config.trunk_passes == 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.name = "boltz2"
    assert not hasattr(first, "prepare_features")


def test_python_job_assets_are_resolved_before_storing(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    job = Job("example", [Protein("A", "ACD", unpaired_msa="a.a3m")])
    handle = get_model("protenix", weights=tmp_path / "weights")
    request = handle.plan(job, stage="inputs", base_dir=tmp_path)
    stored = json.loads(request.input.read_text())
    assert stored["entities"][0]["unpaired_msa"] == str(tmp_path / "a.a3m")
    assert job.to_document()["entities"][0]["unpaired_msa"] == "a.a3m"
    assert request.input == handle.plan(job, base_dir=tmp_path).input
    assert request.output_dir != handle.plan(job, base_dir=tmp_path).output_dir


def test_stage_results_keep_lazy_arrays_and_configuration(tmp_path, monkeypatch):
    seen = []

    def predict(request):
        seen.append(request)
        directory = Path(request.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.ones((3, 4), dtype=np.float32) for name in request.representations
        }
        if arrays:
            from foldjax.backends._representations import _representations_result

            _representations.save(
                directory,
                arrays,
                model=request.model,
                specs=_representations.specs_for(request.model),
            )
            reps = _representations_result(
                request.model, directory, request.representations
            )
        else:
            reps = None
        return PredictionResult(
            request.model, output_dir=directory, representations=reps
        )

    monkeypatch.setattr(foldjax.api, "predict", predict)
    handle = get_model(
        "boltz2",
        weights=tmp_path / "weights",
        config=ModelConfig(msa_depth=1024, trunk_passes=4),
    )
    for method, stage, names in (
        (handle.embed, "inputs", ("single_inputs",)),
        (handle.encode, "trunk", ("single",)),
        (handle.predict, "full", ()),
    ):
        result = method(tmp_path / "job.json", output_dir=tmp_path / stage)
        assert seen[-1].stop_after == stage
        assert seen[-1].representations == names
        assert result.configuration["requested"]["trunk_passes"] == 4
        assert (
            json.loads((result.output_dir / "model_config.json").read_text())[
                "sampling_bindings"
            ]["num_recycles"]["value"]
            == 3
        )
        if names:
            np.testing.assert_array_equal(
                result.representations[names[0]], np.ones((3, 4))
            )


def test_native_integer_bindings_remain_serializable(tmp_path, monkeypatch):
    def predict(request):
        return PredictionResult(request.model, output_dir=tmp_path)

    monkeypatch.setattr(foldjax.api, "predict", predict)
    handle = get_model(
        "boltz2", weights=tmp_path / "weights",
        native_options={"num_samples": np.int64(1)},
    )
    result = handle.predict(tmp_path / "job.json")
    recorded = json.loads((tmp_path / "model_config.json").read_text())
    assert recorded["sampling_bindings"]["num_samples"]["value"] == 1
    assert result.configuration == recorded
    assert isinstance(handle.native_options["num_samples"], np.int64)
