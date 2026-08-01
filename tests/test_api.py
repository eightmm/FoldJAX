import json
from pathlib import Path

import pytest

import foldjax
from foldjax.backends.base import Backend
from foldjax.registry import backend_override
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)


class DummyBackend(Backend):
    """Records the request its dispatcher handed over."""

    name = "boltz2"
    compile_options = ("steps",)

    def __init__(self, input_formats: tuple[str, ...] = ("native",)) -> None:
        self.input_formats = input_formats
        self.seen: PredictionRequest | None = None

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model=self.name, input_formats=self.input_formats)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        self.seen = request
        return PredictionResult(
            model=self.name,
            samples=(PredictionSample(seed=request.seed, scores={"iptm": 0.8}),),
            output_dir=request.output_dir,
            raw={"called": True},
        )


def _request(tmp_path: Path, **overrides) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("version: 1\n")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    fields: dict = {
        "model": "boltz",
        "input": input_path,
        "weights": weights,
        "output_dir": tmp_path / "out",
        "seed": 7,
    }
    fields.update(overrides)
    return PredictionRequest(**fields)


def test_registry_exposes_every_model_and_its_aliases() -> None:
    assert foldjax.available_models() == (
        "alphafold3",
        "boltz2",
        "chai",
        "opendde",
        "openfold3",
        "protenix",
    )
    assert foldjax.normalize_model_name("af3") == "alphafold3"
    assert foldjax.normalize_model_name("boltz") == "boltz2"
    assert foldjax.normalize_model_name("protenix-jax") == "protenix"
    assert foldjax.normalize_model_name("chai-1") == "chai"
    assert foldjax.normalize_model_name("chai-jax") == "chai"
    assert foldjax.normalize_model_name("opendde-jax") == "opendde"
    assert foldjax.normalize_model_name("open-dde") == "opendde"
    assert foldjax.normalize_model_name("of3") == "openfold3"
    assert foldjax.normalize_model_name("openfold3-jax") == "openfold3"
    with pytest.raises(ValueError, match="unknown model"):
        foldjax.normalize_model_name("unknown")


def test_every_backend_declares_its_own_capabilities() -> None:
    for model in foldjax.available_models():
        capabilities = foldjax.capabilities(model)
        assert capabilities.model == model
        assert "foldjax" in capabilities.input_formats
        assert "native" in capabilities.input_formats


def test_predict_dispatches_a_validated_request(tmp_path: Path) -> None:
    backend = DummyBackend()
    with backend_override("boltz2", lambda: backend):
        result = foldjax.predict(_request(tmp_path))

    assert result.model == "boltz2"
    assert result.samples[0].seed == 7
    assert result.samples[0].scores["iptm"] == 0.8
    assert result.raw == {"called": True}
    assert (tmp_path / "out").is_dir()


def test_predict_namespaces_the_compilation_cache_per_backend(tmp_path: Path) -> None:
    backend = DummyBackend()
    with backend_override("boltz2", lambda: backend):
        foldjax.predict(_request(tmp_path, cache_dir=tmp_path / "cache"))

    assert backend.seen is not None
    cache_dir = backend.seen.cache_dir
    assert cache_dir is not None
    # The backend never sees the shared root, only its own subtree.
    assert cache_dir != tmp_path / "cache"
    assert cache_dir.is_relative_to(tmp_path / "cache" / "boltz2" / "weights")


def test_predict_defaults_to_the_shared_compile_cache(tmp_path: Path) -> None:
    """A request that names no cache still gets one: warm runs are the point."""
    backend = DummyBackend()
    with backend_override("boltz2", lambda: backend):
        foldjax.predict(_request(tmp_path))
    assert backend.seen is not None
    cache_dir = backend.seen.cache_dir
    assert cache_dir is not None
    assert cache_dir.is_relative_to(foldjax.paths.compile_cache_dir() / "boltz2")


def test_predict_honours_an_explicit_cache_opt_out(tmp_path: Path) -> None:
    backend = DummyBackend()
    with backend_override("boltz2", lambda: backend):
        foldjax.predict(_request(tmp_path, use_compile_cache=False))
    assert backend.seen is not None
    assert backend.seen.cache_dir is None


def test_predict_converts_common_input_before_dispatch(tmp_path: Path) -> None:
    source = tmp_path / "common.json"
    source.write_text(
        json.dumps({"entities": [{"type": "protein", "id": "A", "sequence": "ACD"}]})
    )
    backend = DummyBackend(input_formats=("native", "foldjax"))
    with backend_override("boltz2", lambda: backend):
        foldjax.predict(_request(tmp_path, input=source, input_format="foldjax"))

    assert backend.seen is not None
    assert backend.seen.input_format == "native"
    # Boltz only parses .yaml/.fasta, so the materialized job must not be .json.
    assert backend.seen.input == tmp_path / "out" / "inputs" / "boltz2_input.yaml"


def test_predict_rejects_an_unsupported_input_format(tmp_path: Path) -> None:
    with backend_override("boltz2", lambda: DummyBackend()):
        with pytest.raises(ValueError, match="does not support input format"):
            foldjax.predict(_request(tmp_path, input_format="alphafold3"))


def test_request_rejects_missing_input_and_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="input"):
        PredictionRequest(
            model="boltz2",
            input=tmp_path / "missing.json",
            weights=tmp_path / "weights",
            output_dir=tmp_path / "out",
        )
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    with pytest.raises(FileNotFoundError, match="weights"):
        PredictionRequest(
            model="boltz2",
            input=input_path,
            weights=tmp_path / "weights",
            output_dir=tmp_path / "out",
        )


def test_request_rejects_a_negative_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        _request(tmp_path, seed=-1)


def test_request_coerces_string_paths(tmp_path: Path) -> None:
    request = _request(tmp_path, output_dir=str(tmp_path / "out"))
    assert isinstance(request.output_dir, Path)


def test_result_summary_is_json_serializable(tmp_path: Path) -> None:
    result = PredictionResult(
        model="alphafold3",
        samples=(
            PredictionSample(
                seed=1,
                structure_path=tmp_path / "model.cif",
                scores={"ranking_score": 0.9},
            ),
        ),
        output_dir=tmp_path,
        raw={"large": object()},
    )
    assert result.summary() == {
        "model": "alphafold3",
        "output_dir": str(tmp_path),
        "samples": [
            {
                "seed": 1,
                "structure_path": str(tmp_path / "model.cif"),
                "scores": {"ranking_score": 0.9},
                "metadata": {},
            }
        ],
    }
    assert json.loads(json.dumps(result.summary()))["model"] == "alphafold3"
