"""The one-file-in workflow: auto-detected input and model-neutral knobs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import foldjax
from foldjax.api import detect_input_format
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.backends.chai import ChaiBackend
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.backends.protenix import ProtenixBackend
from foldjax.input import read_job_document
from foldjax.registry import backend_override
from foldjax.schema import PredictionRequest

_JOB = {"entities": [{"type": "protein", "id": "A", "sequence": "ACD"}]}


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "weights.jax"
    path.touch()
    return path


# --------------------------------------------------------------------------
# input format
# --------------------------------------------------------------------------


def test_common_schema_is_detected_in_both_json_and_yaml(tmp_path: Path) -> None:
    as_json = tmp_path / "job.json"
    as_json.write_text(json.dumps(_JOB))
    as_yaml = tmp_path / "job.yaml"
    as_yaml.write_text("entities:\n  - type: protein\n    id: A\n    sequence: ACD\n")
    assert detect_input_format(as_json) == "foldjax"
    assert detect_input_format(as_yaml) == "foldjax"
    assert read_job_document(as_json) == read_job_document(as_yaml)


def test_native_dialects_are_not_mistaken_for_the_common_schema(
    tmp_path: Path,
) -> None:
    # Protenix-family native input is a list of jobs, Boltz native is a mapping
    # with `sequences`, and Chai native is a FASTA. None has `entities`.
    protenix = tmp_path / "native.json"
    protenix.write_text(json.dumps([{"name": "x", "sequences": []}]))
    boltz = tmp_path / "native.yaml"
    boltz.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n")
    fasta = tmp_path / "job.fasta"
    fasta.write_text(">protein|name=A\nACD\n")

    assert detect_input_format(protenix) == "native"
    assert detect_input_format(boltz) == "native"
    assert detect_input_format(fasta) == "native"


def test_unparseable_structured_file_is_treated_as_native(tmp_path: Path) -> None:
    broken = tmp_path / "job.json"
    broken.write_text("{not json at all")
    assert detect_input_format(broken) == "native"


def test_yaml_common_input_materializes_like_its_json_twin(tmp_path: Path) -> None:
    from foldjax.input import materialize_native_input
    from foldjax.registry import capabilities

    as_json = tmp_path / "job.json"
    as_json.write_text(json.dumps(_JOB))
    as_yaml = tmp_path / "job.yaml"
    as_yaml.write_text("entities:\n  - type: protein\n    id: A\n    sequence: ACD\n")

    caps = capabilities("protenix")
    from_json = materialize_native_input(as_json, caps, tmp_path / "a", seed=1)
    from_yaml = materialize_native_input(as_yaml, caps, tmp_path / "b", seed=1)
    assert from_json.read_text() == from_yaml.read_text()


# --------------------------------------------------------------------------
# sampling knobs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            Boltz2Backend(),
            {"diffusion_samples": 3, "steps": 40, "recycling": 2},
        ),
        (
            ChaiBackend(),
            {
                "num_diffusion_samples": 3,
                "num_diffusion_timesteps": 40,
                "num_trunk_recycles": 2,
            },
        ),
        (OpenDDEBackend(), {"n_sample": 3, "n_step": 40, "n_cycle": 2}),
        (ProtenixBackend(), {"n_sample": 3, "n_step": 40, "n_cycle": 2}),
    ],
    ids=["boltz2", "chai", "opendde", "protenix"],
)
def test_neutral_knobs_become_each_backends_own_option_names(
    tmp_path: Path, backend, expected
) -> None:
    request = PredictionRequest(
        model=backend.name,
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        num_samples=3,
        num_steps=40,
        num_recycles=2,
    )
    assert backend.apply_sampling(request) == expected


def test_every_translated_option_is_one_the_backend_really_accepts() -> None:
    """A knob that maps onto a name the backend rejects would fail at runtime."""
    for backend in (OpenDDEBackend(), ProtenixBackend()):
        from foldjax.backends import opendde, protenix

        accepted = (
            opendde._CLI_OPTIONS if backend.name == "opendde" else protenix._CLI_OPTIONS
        )
        assert set(backend.sampling_options.values()) <= accepted, backend.name
    for backend in (Boltz2Backend(), ChaiBackend()):
        assert set(backend.sampling_options.values()) <= set(backend.compile_options)


def test_setting_a_knob_and_its_native_name_together_is_rejected(
    tmp_path: Path,
) -> None:
    request = PredictionRequest(
        model="opendde",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        num_samples=3,
        options={"n_sample": 5},
    )
    with pytest.raises(ValueError, match="both set"):
        OpenDDEBackend().apply_sampling(request)


def test_a_knob_the_backend_cannot_express_is_rejected(tmp_path: Path) -> None:
    from foldjax.backends.alphafold3 import AlphaFold3Backend

    request = PredictionRequest(
        model="alphafold3",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        num_steps=8,
    )
    with pytest.raises(ValueError, match="does not support num_steps"):
        AlphaFold3Backend().apply_sampling(request)


def test_knobs_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_samples must be at least 1"):
        PredictionRequest(model="opendde", input=_job_file(tmp_path), num_samples=0)


def test_capabilities_report_the_knobs_each_backend_honours() -> None:
    for name in ("boltz2", "chai", "opendde", "protenix"):
        sampling = foldjax.capabilities(name).sampling
        assert {"num_samples", "num_steps", "num_recycles"} <= set(sampling), name
    # Chai has no reachable MSA depth cap: it reads alignments through its own
    # context objects rather than a featurizer argument. Reporting the knob it
    # cannot honour would be worse than reporting none.
    assert "max_msa_depth" not in foldjax.capabilities("chai").sampling
    for name in ("boltz2", "opendde", "protenix"):
        assert "max_msa_depth" in foldjax.capabilities(name).sampling, name
    # AlphaFold 3 has no diffusion-step count at all.
    assert set(foldjax.capabilities("alphafold3").sampling) == {
        "num_samples",
        "num_recycles",
    }


# --------------------------------------------------------------------------
# request resolution
# --------------------------------------------------------------------------


def _job_file(tmp_path: Path) -> Path:
    path = tmp_path / "job.json"
    if not path.exists():
        path.write_text(json.dumps(_JOB))
    return path


def test_a_bare_request_resolves_weights_output_and_cache(
    tmp_path: Path, monkeypatch
) -> None:
    store = tmp_path / "store"
    (store / "weights" / "opendde").mkdir(parents=True)
    (store / "weights" / "opendde" / "opendde.jax").touch()
    monkeypatch.setenv("FOLDJAX_HOME", str(store))
    monkeypatch.chdir(tmp_path)

    resolved = foldjax.resolve_request(
        PredictionRequest(model="opendde", input=_job_file(tmp_path))
    )
    assert resolved.weights == store / "weights" / "opendde" / "opendde.jax"
    assert resolved.output_dir == Path("foldjax-outputs") / "job"
    assert resolved.cache_dir == store / "compile"
    assert resolved.input_format == "foldjax"


def test_missing_weights_name_the_command_that_fetches_them(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="foldjax weights fetch --model chai"):
        foldjax.resolve_request(
            PredictionRequest(model="chai", input=_job_file(tmp_path))
        )


def test_explicit_values_are_never_overridden(tmp_path: Path) -> None:
    request = PredictionRequest(
        model="opendde",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path / "mine",
        cache_dir=tmp_path / "cache",
        input_format="native",
    )
    resolved = foldjax.resolve_request(request)
    assert resolved.weights == tmp_path / "weights.jax"
    assert resolved.output_dir == tmp_path / "mine"
    assert resolved.cache_dir == tmp_path / "cache"
    assert resolved.input_format == "native"


def test_sampling_knobs_survive_the_whole_predict_path(tmp_path: Path) -> None:
    """End to end: a neutral knob reaches the backend under its native name."""
    from tests.test_api import DummyBackend

    backend = DummyBackend()
    with backend_override("opendde", lambda: backend):
        foldjax.predict(
            PredictionRequest(
                model="opendde",
                input=_job_file(tmp_path),
                weights=_weights(tmp_path),
                output_dir=tmp_path / "out",
                input_format="native",
                num_samples=4,
            )
        )
    assert backend.seen is not None
    assert backend.seen.num_samples == 4
    assert OpenDDEBackend().apply_sampling(backend.seen)["n_sample"] == 4
