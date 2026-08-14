import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
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
            samples=(
                PredictionSample(
                    seed=request.seed,
                    coordinates=((0.0, 0.0, 0.0),),
                    scores={"iptm": 0.8},
                ),
            ),
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
        "esmfold2",
        "opendde",
        "openfold3",
        "protenix",
    )
    assert foldjax.normalize_model_name("af3") == "alphafold3"
    assert foldjax.normalize_model_name("boltz") == "boltz2"
    assert foldjax.normalize_model_name("protenix-jax") == "protenix"
    assert foldjax.normalize_model_name("opendde-jax") == "opendde"
    assert foldjax.normalize_model_name("open-dde") == "opendde"
    assert foldjax.normalize_model_name("esmfold-2") == "esmfold2"
    assert foldjax.normalize_model_name("of3") == "openfold3"
    assert foldjax.normalize_model_name("openfold3-jax") == "openfold3"
    with pytest.raises(ValueError, match="unknown model"):
        foldjax.normalize_model_name("unknown")


def test_model_info_exposes_typed_runtime_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import build

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    monkeypatch.setattr(build, "is_ready", lambda: False)
    monkeypatch.setattr(build, "runtime_blocker", lambda: None)

    alphafold = foldjax.model_info("af3")
    assert isinstance(alphafold.runtime, foldjax.RuntimeInfo)
    assert alphafold.runtime.ready is False
    assert alphafold.runtime.setup == (
        "foldjax runtime prepare --model alphafold3"
    )
    assert alphafold.runtime.requires_network is True

    boltz = foldjax.model_info("boltz")
    assert boltz.runtime.ready is True
    assert boltz.runtime.setup is None
    assert boltz.runtime.requires_network is False


def test_every_backend_declares_its_own_capabilities() -> None:
    for model in foldjax.available_models():
        capabilities = foldjax.capabilities(model)
        assert capabilities.model == model
        assert "foldjax" in capabilities.input_formats
        assert set(capabilities.input_requirements) == set(
            capabilities.input_formats
        )
        assert all(
            isinstance(requirement, foldjax.InputRequirement)
            for requirement in capabilities.input_requirements.values()
        )
        if model == "esmfold2":
            # The one backend with no native dialect to accept: upstream's
            # entry point takes a sequence string (or its SDK's objects), not
            # a job file, so "native" here would name a format that does not
            # exist. Every other backend has a published input format and must
            # keep taking it.
            continue
        assert "native" in capabilities.input_formats


def test_model_capabilities_keep_positional_construction_compatible() -> None:
    capabilities = ModelCapabilities(
        "third-party", ("native",), ("protein",), False, False, False, {}
    )

    assert capabilities.entity_types == ("protein",)
    assert capabilities.input_requirements == {}


@pytest.mark.parametrize(
    ("options", "expected_profile"),
    [
        ({}, "released"),
        ({"no_language_model": False}, "released"),
        ({"no_language_model": True}, "structure-only"),
        ({"esmc_weights": "/tmp/external-esmc"}, "structure-only"),
    ],
)
def test_esmfold2_request_resolution_selects_the_matching_asset_profile(
    tmp_path: Path,
    monkeypatch,
    options: dict,
    expected_profile: str,
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('{"entities": []}')
    resolved_weights = tmp_path / "weights" / "model.safetensors"
    resolved_weights.parent.mkdir()
    resolved_weights.write_bytes(b"weights")
    calls = []

    def fake_resolve(model: str, *, profile: str | None = None) -> Path:
        calls.append((model, profile))
        return resolved_weights

    monkeypatch.setattr(foldjax.assets, "resolve_weights", fake_resolve)
    resolved = foldjax.resolve_request(
        PredictionRequest(
            model="esmfold2",
            input=input_path,
            output_dir=tmp_path / "out",
            options=options,
        )
    )

    assert resolved.weights == resolved_weights
    assert resolved.profile == expected_profile
    assert calls == [("esmfold2", expected_profile)]


@pytest.mark.parametrize(
    ("model_name", "expected_profile"),
    [
        ("protenix_base_default_v1.0.0", "released"),
        ("protenix_mini_esm_v0.5.0", "mini-esm-v0.5.0"),
        ("protenix_mini_ism_v0.5.0", "mini-ism-v0.5.0"),
    ],
)
def test_protenix_request_resolution_selects_variant_managed_assets(
    tmp_path: Path,
    monkeypatch,
    model_name: str,
    expected_profile: str,
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('[{"sequences": []}]')
    resolved_weights = tmp_path / expected_profile / "weights.jax"
    resolved_weights.parent.mkdir()
    resolved_weights.write_bytes(b"weights")
    calls = []

    def fake_resolve(model: str, *, profile: str | None = None) -> Path:
        calls.append((model, profile))
        return resolved_weights

    monkeypatch.setattr(foldjax.assets, "resolve_weights", fake_resolve)
    resolved = foldjax.resolve_request(
        PredictionRequest(
            model="protenix",
            input=input_path,
            output_dir=tmp_path / "out",
            options={"model_name": model_name},
        )
    )

    assert resolved.weights == resolved_weights
    assert resolved.profile == expected_profile
    assert calls == [("protenix", expected_profile)]


@pytest.mark.parametrize(
    ("profile", "model_name"),
    [
        ("mini-esm-v0.5.0", "protenix_mini_esm_v0.5.0"),
        ("mini-ism-v0.5.0", "protenix_mini_ism_v0.5.0"),
    ],
)
def test_protenix_first_class_profile_configures_the_complete_bundle(
    tmp_path: Path,
    monkeypatch,
    profile: str,
    model_name: str,
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('[{"sequences": []}]')
    resolved_weights = tmp_path / profile / "weights.jax"
    resolved_weights.parent.mkdir()
    resolved_weights.write_bytes(b"weights")
    calls = []

    def fake_resolve(model: str, *, profile: str | None = None) -> Path:
        calls.append((model, profile))
        return resolved_weights

    monkeypatch.setattr(foldjax.assets, "resolve_weights", fake_resolve)
    resolved = foldjax.resolve_request(
        PredictionRequest(
            model="protenix",
            input=input_path,
            profile=profile,
            output_dir=tmp_path / "out",
        )
    )

    assert calls == [("protenix", profile)]
    assert resolved.weights == resolved_weights
    assert resolved.profile == profile
    assert resolved.options["model_name"] == model_name
    assert resolved.options["esm_checkpoint_dir"] == resolved_weights.parent


def test_protenix_profile_can_describe_an_external_complete_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('[{"sequences": []}]')
    external_weights = tmp_path / "external" / "mini.jax"
    external_weights.parent.mkdir()
    external_weights.write_bytes(b"external")

    monkeypatch.setattr(
        foldjax.assets,
        "resolve_weights",
        lambda *args, **kwargs: pytest.fail("explicit weights were replaced"),
    )
    resolved = foldjax.resolve_request(
        PredictionRequest(
            model="protenix",
            input=input_path,
            weights=external_weights,
            profile="mini-esm-v0.5.0",
            output_dir=tmp_path / "out",
        )
    )

    assert resolved.weights == external_weights
    assert resolved.options == {
        "model_name": "protenix_mini_esm_v0.5.0",
        "esm_checkpoint_dir": external_weights.parent,
    }
    assert foldjax.resolve_request(resolved) == resolved


@pytest.mark.parametrize(
    "options",
    [
        {"model_name": "protenix_mini_ism_v0.5.0"},
        {"esm_checkpoint_dir": "/tmp/not-the-managed-bundle"},
    ],
)
def test_protenix_profile_rejects_conflicting_native_options(
    tmp_path: Path, monkeypatch, options: dict
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('[{"sequences": []}]')
    resolved_weights = tmp_path / "managed" / "mini.jax"
    resolved_weights.parent.mkdir()
    resolved_weights.write_bytes(b"weights")
    monkeypatch.setattr(
        foldjax.assets,
        "resolve_weights",
        lambda *args, **kwargs: resolved_weights,
    )

    with pytest.raises(ValueError, match="conflict"):
        foldjax.resolve_request(
            PredictionRequest(
                model="protenix",
                input=input_path,
                profile="mini-esm-v0.5.0",
                output_dir=tmp_path / "out",
                options=options,
            )
        )


def test_esmfold2_structure_only_profile_disables_the_language_model(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('{"entities": []}')
    resolved_weights = tmp_path / "weights" / "model.safetensors"
    resolved_weights.parent.mkdir()
    resolved_weights.write_bytes(b"weights")
    calls = []

    def fake_resolve(model: str, *, profile: str | None = None) -> Path:
        calls.append((model, profile))
        return resolved_weights

    monkeypatch.setattr(foldjax.assets, "resolve_weights", fake_resolve)
    resolved = foldjax.resolve_request(
        PredictionRequest(
            model="esmfold2",
            input=input_path,
            profile="structure-only",
            output_dir=tmp_path / "out",
        )
    )

    assert calls == [("esmfold2", "structure-only")]
    assert resolved.profile == "structure-only"
    assert resolved.options == {"no_language_model": True}


@pytest.mark.parametrize(
    ("profile", "options"),
    [
        ("structure-only", {"no_language_model": False}),
        ("released", {"no_language_model": True}),
    ],
)
def test_esmfold2_profile_rejects_conflicting_native_options(
    tmp_path: Path, profile: str, options: dict
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('{"entities": []}')

    with pytest.raises(ValueError, match="conflict|cannot be combined"):
        foldjax.resolve_request(
            PredictionRequest(
                model="esmfold2",
                input=input_path,
                profile=profile,
                output_dir=tmp_path / "out",
                options=options,
            )
        )


def test_esmfold2_structure_profile_accepts_an_external_language_model(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('{"entities": []}')
    resolved_weights = tmp_path / "weights" / "model.safetensors"
    resolved_weights.parent.mkdir()
    resolved_weights.write_bytes(b"weights")
    monkeypatch.setattr(
        foldjax.assets,
        "resolve_weights",
        lambda *args, **kwargs: resolved_weights,
    )

    resolved = foldjax.resolve_request(
        PredictionRequest(
            model="esmfold2",
            input=input_path,
            profile="structure-only",
            output_dir=tmp_path / "out",
            options={"esmc_weights": "/tmp/external-esmc"},
        )
    )

    assert resolved.profile == "structure-only"
    assert resolved.options == {"esmc_weights": "/tmp/external-esmc"}
    assert foldjax.resolve_request(resolved) == resolved


def test_request_profile_is_trimmed_and_validated(tmp_path: Path) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('{}')

    request = PredictionRequest(
        model="protenix", input=input_path, profile=" mini-esm-v0.5.0 "
    )
    assert request.profile == "mini-esm-v0.5.0"
    with pytest.raises(ValueError, match="profile must be a non-empty string"):
        PredictionRequest(model="protenix", input=input_path, profile="  ")


def test_protenix_external_weights_are_not_replaced_by_variant_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('[{"sequences": []}]')
    external_weights = tmp_path / "external" / "mini.jax"
    external_weights.parent.mkdir()
    external_weights.write_bytes(b"external")
    checkpoint_dir = tmp_path / "external" / "esm"

    monkeypatch.setattr(
        foldjax.assets,
        "resolve_weights",
        lambda *args, **kwargs: pytest.fail("explicit weights were replaced"),
    )
    resolved = foldjax.resolve_request(
        PredictionRequest(
            model="protenix",
            input=input_path,
            weights=external_weights,
            output_dir=tmp_path / "out",
            options={
                "model_name": "protenix_mini_esm_v0.5.0",
                "esm_checkpoint_dir": checkpoint_dir,
            },
        )
    )

    assert resolved.weights == external_weights
    assert resolved.options["esm_checkpoint_dir"] == checkpoint_dir


@pytest.mark.parametrize(
    "options",
    [
        {"no_language_model": "true"},
        {"no_language_model": True, "esmc_weights": "/tmp/esmc"},
    ],
)
def test_esmfold2_variant_is_validated_even_with_explicit_weights(
    tmp_path: Path, options: dict
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('{"entities": []}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")

    with pytest.raises(ValueError):
        foldjax.resolve_request(
            PredictionRequest(
                model="esmfold2",
                input=input_path,
                weights=weights,
                output_dir=tmp_path / "out",
                options=options,
            )
        )


@pytest.mark.parametrize(
    ("input_format", "options", "message"),
    [
        ("native", {}, "does not support input format"),
        ("foldjax", {"num_stpes": 20}, "unsupported esmfold2 options"),
    ],
)
def test_resolution_rejects_a_plan_that_prediction_cannot_run(
    tmp_path: Path, input_format: str, options: dict, message: str
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('{"entities": []}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    output = tmp_path / "out"

    with pytest.raises(ValueError, match=message):
        foldjax.resolve_request(
            PredictionRequest(
                model="esmfold2",
                input=input_path,
                input_format=input_format,
                weights=weights,
                output_dir=output,
                options=options,
            )
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("model", "input_name", "input_format", "options", "message"),
    [
        (
            "opendde",
            "job.json",
            "native",
            {"include_raw": "yes"},
            "include_raw must be a boolean",
        ),
        (
            "openfold3",
            "features.npz",
            "openfold3-features",
            {"no_compile": "maybe"},
            "no_compile must be a boolean",
        ),
    ],
)
def test_resolution_validates_native_values_before_model_or_output_work(
    tmp_path: Path,
    model: str,
    input_name: str,
    input_format: str,
    options: dict,
    message: str,
) -> None:
    input_path = tmp_path / input_name
    input_path.write_bytes(b"{}")
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"weights")
    output = tmp_path / "out"

    with pytest.raises(ValueError, match=message):
        foldjax.resolve_request(
            PredictionRequest(
                model=model,
                input=input_path,
                input_format=input_format,
                weights=weights,
                output_dir=output,
                options=options,
            )
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("model", "input_name", "input_format", "native_sampling"),
    [
        ("alphafold3", "job.json", "native", "diffusion_steps"),
        ("boltz2", "job.yaml", "native", "steps"),
        ("esmfold2", "job.json", "foldjax", "num_steps"),
        ("opendde", "job.json", "native", "n_step"),
        (
            "openfold3",
            "features.npz",
            "openfold3-features",
            "no_rollout_steps",
        ),
        ("protenix", "job.json", "native", "n_step"),
    ],
)
def test_every_backend_validates_native_sampling_during_resolution(
    tmp_path: Path,
    model: str,
    input_name: str,
    input_format: str,
    native_sampling: str,
) -> None:
    input_path = tmp_path / input_name
    input_path.write_text('{"entities": []}')
    weights = tmp_path / "weights"
    weights.mkdir()

    with pytest.raises(ValueError, match=f"{native_sampling} must be an integer"):
        foldjax.resolve_request(
            PredictionRequest(
                model=model,
                input=input_path,
                input_format=input_format,
                weights=weights,
                output_dir=tmp_path / "out",
                options={native_sampling: "bad"},
            )
        )


def test_alphafold3_bucket_elements_are_validated_during_resolution(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()

    with pytest.raises(ValueError, match="each bucket must be an integer"):
        foldjax.resolve_request(
            PredictionRequest(
                model="alphafold3",
                input=input_path,
                input_format="native",
                weights=weights,
                output_dir=tmp_path / "out",
                options={"buckets": [256, "bad"]},
            )
        )


def test_alphafold3_kernel_autotuning_is_validated_during_resolution(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()

    with pytest.raises(ValueError, match="kernel_autotuning must be one of"):
        foldjax.resolve_request(
            PredictionRequest(
                model="alphafold3",
                input=input_path,
                input_format="native",
                weights=weights,
                output_dir=tmp_path / "out",
                options={"kernel_autotuning": "typo"},
            )
        )


def test_openfold3_npz_auto_reports_the_jax_only_feature_dialect(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "features.npz"
    archive.write_bytes(b"not loaded during planning")
    weights = tmp_path / "of3_ft3_v1.pt"
    weights.write_bytes(b"weights")

    resolved = foldjax.resolve_request(
        PredictionRequest(
            model="openfold3",
            input=archive,
            weights=weights,
            output_dir=tmp_path / "out",
        )
    )

    assert resolved.input_format == "openfold3-features"


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


def test_default_scalar_output_preserves_the_compatible_layout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = foldjax.resolve_request(
        _request(
            tmp_path,
            output_dir=None,
            use_compile_cache=False,
            model="boltz",
        )
    )

    assert resolved.model == "boltz2"
    assert resolved.output_dir == Path("foldjax-outputs") / "job"


def test_default_generated_output_symlink_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    generated = tmp_path / "foldjax-outputs" / "job"
    generated.parent.mkdir()
    generated.symlink_to(outside, target_is_directory=True)
    backend = DummyBackend()

    with backend_override("boltz2", lambda: backend):
        with pytest.raises(
            foldjax.PredictionOutputError, match="output directory is a symlink"
        ):
            foldjax.predict(
                _request(
                    tmp_path,
                    output_dir=None,
                    use_compile_cache=False,
                )
            )

    assert backend.seen is None


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


def test_a_backend_without_a_dialect_keeps_the_common_format(tmp_path: Path) -> None:
    """ESMFold2 reads the common schema itself, so there is no `native` to claim.

    Relabelling the materialized file `native` for such a backend made it fail
    the capability check on a format the conversion had just invented, which
    is how `foldjax predict --model esmfold2` came to be impossible.
    """
    source = tmp_path / "common.json"
    source.write_text(
        json.dumps({"entities": [{"type": "protein", "id": "A", "sequence": "ACD"}]})
    )
    backend = DummyBackend(input_formats=("foldjax",))
    backend.name = "esmfold2"
    with backend_override("esmfold2", lambda: backend):
        foldjax.predict(
            _request(tmp_path, model="esmfold2", input=source, input_format="foldjax")
        )

    assert backend.seen is not None
    assert backend.seen.input_format == "foldjax"
    assert backend.seen.input == tmp_path / "out" / "inputs" / "esmfold2_input.json"


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("seed", 1.5),
        ("num_seeds", "2"),
        ("num_samples", False),
        ("num_steps", 20.5),
        ("num_recycles", "3"),
        ("max_msa_depth", 64.5),
    ],
)
def test_request_rejects_values_that_are_not_integers(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _request(tmp_path, **{field: value})


def test_request_normalizes_numpy_integer_knobs_for_json(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        seed=np.int64(4),
        num_seeds=np.int32(2),
        num_steps=np.int64(20),
    )

    assert request.seed == 4 and type(request.seed) is int
    assert request.num_seeds == 2 and type(request.num_seeds) is int
    assert request.num_steps == 20 and type(request.num_steps) is int
    json.dumps({"seeds": request.resolved_seeds, "sampling": request.sampling})


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
                "coordinate_shape": None,
                "scores": {"ranking_score": 0.9},
                "metadata": {},
            }
        ],
    }
    assert json.loads(json.dumps(result.summary()))["model"] == "alphafold3"


def test_coordinate_only_summary_reports_shape_and_redacts_metadata() -> None:
    result = PredictionResult(
        model="opendde",
        samples=(
            PredictionSample(
                seed=3,
                coordinates=[(0.0, 1.0, 2.0), (3.0, 4.0, 5.0)],
                metadata={"accessToken": "do-not-log", "sample": 0},
            ),
        ),
    )

    sample = result.summary()["samples"][0]
    assert sample["coordinate_shape"] == [2, 3]
    assert sample["metadata"] == {"accessToken": "[REDACTED]", "sample": 0}


def test_numpy_scores_are_normalized_before_json_outputs(tmp_path: Path) -> None:
    class NumpyScoreBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=np.int64(request.seed),
                        coordinates=((0.0, 0.0, 0.0),),
                        scores={"confidence": np.float32(0.75)},
                    ),
                ),
                output_dir=request.output_dir,
            )

    output = tmp_path / "out"
    with backend_override("boltz2", NumpyScoreBackend):
        result = foldjax.predict(
            _request(tmp_path, output_dir=output, use_compile_cache=False)
        )

    assert result.samples[0].seed == 7
    assert result.samples[0].scores == {"confidence": 0.75}
    json.dumps(result.summary())
    manifest = json.loads((output / "foldjax_run.json").read_text())
    assert manifest["samples"][0]["scores"] == {"confidence": 0.75}


def test_one_declaration_runs_every_model_on_every_input(tmp_path: Path) -> None:
    """The plural spellings fan out the cross product, namespaced by both axes.

    Two inputs through one backend: two full predictions, each into
    `out/<model>/<input stem>`, returned in declaration order. The scalar
    request keeps returning a single result -- fanning is something the caller
    asks for by spelling the field in the plural, never a surprise.
    """
    first = tmp_path / "alpha.yaml"
    second = tmp_path / "beta.yaml"
    first.write_text("version: 1\n")
    second.write_text("version: 1\n")

    request = _request(
        tmp_path,
        model=None,
        models=("boltz",),
        input=None,
        inputs=(first, second),
    )
    resolved = foldjax.resolve_requests(request)
    assert [item.model for item in resolved] == ["boltz2", "boltz2"]
    assert [item.output_dir.name for item in resolved] == ["alpha", "beta"]

    backend = DummyBackend()
    with backend_override("boltz2", lambda: backend):
        results = foldjax.predict(request)

    assert isinstance(results, tuple) and len(results) == 2
    assert [r.output_dir.name for r in results] == ["alpha", "beta"]
    assert all(r.output_dir.parent.name == "boltz2" for r in results)


def test_plural_spelling_returns_a_tuple_even_for_one_run(tmp_path: Path) -> None:
    source = tmp_path / "only.yaml"
    source.write_text("version: 1\n")
    request = _request(
        tmp_path,
        model=None,
        models=("boltz",),
        input=None,
        inputs=(source,),
    )

    backend = DummyBackend()
    with backend_override("boltz2", lambda: backend):
        results = foldjax.predict(request)

    assert isinstance(results, tuple) and len(results) == 1
    assert results[0].model == "boltz2"


def test_plural_inputs_with_the_same_stem_are_refused(tmp_path: Path) -> None:
    first = tmp_path / "first" / "job.yaml"
    second = tmp_path / "second" / "job.yaml"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("version: 1\n")
    second.write_text("version: 1\n")
    request = _request(
        tmp_path,
        model=None,
        models=("boltz2",),
        input=None,
        inputs=(first, second),
    )

    with pytest.raises(ValueError, match="share output"):
        foldjax.resolve_requests(request)


def test_plural_generated_output_symlink_is_refused_before_dispatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "job.yaml"
    source.write_text("version: 1\n")
    root = tmp_path / "out"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "boltz2").mkdir(parents=True)
    (root / "boltz2" / "job").symlink_to(outside, target_is_directory=True)
    request = _request(
        tmp_path,
        model=None,
        models=("boltz2",),
        input=None,
        inputs=(source,),
        output_dir=root,
        use_compile_cache=False,
    )
    backend = DummyBackend()

    with backend_override("boltz2", lambda: backend):
        with pytest.raises(
            foldjax.PredictionOutputError, match="output directory is a symlink"
        ):
            foldjax.predict(request)

    assert backend.seen is None


def test_plural_models_refuse_one_ambiguous_explicit_weight_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "job.yaml"
    source.write_text("version: 1\n")
    request = _request(
        tmp_path,
        model=None,
        models=("boltz2", "protenix"),
        input=source,
    )

    with pytest.raises(ValueError, match="one explicit weights path"):
        foldjax.resolve_requests(request)


def test_a_backend_returning_no_samples_fails_without_a_manifest(
    tmp_path: Path,
) -> None:
    class EmptyBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(model=self.name, output_dir=request.output_dir)

    output = tmp_path / "out"
    with backend_override("boltz2", EmptyBackend):
        with pytest.raises(
            foldjax.PredictionOutputError, match="no prediction samples"
        ):
            foldjax.predict(
                _request(tmp_path, output_dir=output, use_compile_cache=False)
            )

    assert not (output / "foldjax_run.json").exists()


@pytest.mark.parametrize(
    "coordinates",
    [
        object(),
        [[]],
        (value for value in ((0.0, 0.0, 0.0),)),
        ((0.0, float("nan"), 2.0),),
        ((0.0, 1.0),),
        (("x", "y", "z"),),
    ],
    ids=[
        "opaque-object",
        "empty-nested-list",
        "generator",
        "non-finite",
        "not-xyz",
        "non-numeric",
    ],
)
def test_malformed_coordinate_placeholders_are_not_success(
    tmp_path: Path, coordinates
) -> None:
    class MalformedBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(seed=request.seed, coordinates=coordinates),
                ),
                output_dir=request.output_dir,
            )

    with backend_override("boltz2", MalformedBackend):
        with pytest.raises(foldjax.PredictionOutputError, match="no coordinates"):
            foldjax.predict(_request(tmp_path, use_compile_cache=False))


def test_prediction_samples_must_follow_the_tuple_contract(tmp_path: Path) -> None:
    class ListBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=[
                    PredictionSample(
                        seed=request.seed, coordinates=((0.0, 0.0, 0.0),)
                    )
                ],
                output_dir=request.output_dir,
            )

    with backend_override("boltz2", ListBackend):
        with pytest.raises(foldjax.PredictionOutputError, match="must be a tuple"):
            foldjax.predict(_request(tmp_path, use_compile_cache=False))


def test_an_array_sample_container_gets_the_public_contract_error(
    tmp_path: Path,
) -> None:
    class ArrayBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=np.asarray([PredictionSample(seed=request.seed)], dtype=object),
                output_dir=request.output_dir,
            )

    with backend_override("boltz2", ArrayBackend):
        with pytest.raises(foldjax.PredictionOutputError, match="must be a tuple"):
            foldjax.predict(_request(tmp_path, use_compile_cache=False))


@pytest.mark.parametrize("score", [float("nan"), True, "high"])
def test_sample_scores_must_be_finite_numbers(tmp_path: Path, score) -> None:
    class BadScoreBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed,
                        coordinates=((0.0, 0.0, 0.0),),
                        scores={"confidence": score},
                    ),
                ),
                output_dir=tmp_path / "wrong-directory",
            )

    with backend_override("boltz2", BadScoreBackend):
        with pytest.raises(
            foldjax.PredictionOutputError, match="non-numeric or non-finite scores"
        ):
            foldjax.predict(_request(tmp_path, use_compile_cache=False))


def test_sample_seed_must_match_the_run(tmp_path: Path) -> None:
    class WrongSeedBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed + 1,
                        coordinates=((0.0, 0.0, 0.0),),
                    ),
                ),
                output_dir=request.output_dir,
            )

    with backend_override("boltz2", WrongSeedBackend):
        with pytest.raises(foldjax.PredictionOutputError, match="run used seed"):
            foldjax.predict(_request(tmp_path, use_compile_cache=False))


def test_structure_path_must_be_path_like_even_with_coordinates(
    tmp_path: Path,
) -> None:
    class BadPathBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed,
                        structure_path=object(),
                        coordinates=((0.0, 0.0, 0.0),),
                    ),
                ),
                output_dir=request.output_dir,
            )

    with backend_override("boltz2", BadPathBackend):
        with pytest.raises(foldjax.PredictionOutputError, match="path-like"):
            foldjax.predict(_request(tmp_path, use_compile_cache=False))


def test_coordinate_fallback_drops_a_missing_structure_path(tmp_path: Path) -> None:
    class CoordinateBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed,
                        structure_path=tmp_path / "missing.cif",
                        coordinates=((0.0, 0.0, 0.0),),
                    ),
                ),
                output_dir=request.output_dir,
            )

    with backend_override("boltz2", CoordinateBackend):
        result = foldjax.predict(_request(tmp_path, use_compile_cache=False))

    assert result.samples[0].structure_path is None


def test_common_result_reports_the_resolved_output_directory(tmp_path: Path) -> None:
    class WrongDirectoryBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed, coordinates=((0.0, 0.0, 0.0),)
                    ),
                ),
                output_dir=tmp_path / "backend-internal",
            )

    requested = tmp_path / "public-output"
    with backend_override("boltz2", WrongDirectoryBackend):
        result = foldjax.predict(
            _request(
                tmp_path,
                output_dir=requested,
                use_compile_cache=False,
            )
        )

    assert result.output_dir == requested


def test_a_backend_returning_a_missing_structure_fails_without_a_manifest(
    tmp_path: Path,
) -> None:
    class MissingBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed,
                        structure_path=request.output_dir / "missing.cif",
                    ),
                ),
                output_dir=request.output_dir,
            )

    output = tmp_path / "out"
    with backend_override("boltz2", MissingBackend):
        with pytest.raises(foldjax.PredictionOutputError, match="missing or empty"):
            foldjax.predict(
                _request(tmp_path, output_dir=output, use_compile_cache=False)
            )

    assert not (output / "foldjax_run.json").exists()


@pytest.mark.parametrize("same_path", [False, True], ids=["same-slot", "same-file"])
def test_duplicate_structure_destinations_are_refused_before_normalization(
    tmp_path: Path, same_path: bool
) -> None:
    class DuplicateBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            first = request.output_dir / "first.cif"
            second = first if same_path else request.output_dir / "second.cif"
            first.write_text("data_first\n#\n")
            if second != first:
                second.write_text("data_second\n#\n")
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed,
                        structure_path=first,
                        metadata={"sample": 0},
                    ),
                    PredictionSample(
                        seed=request.seed,
                        structure_path=second,
                        metadata={"sample": 1 if same_path else 0},
                    ),
                ),
                output_dir=request.output_dir,
            )

    output = tmp_path / "out"
    with backend_override("boltz2", DuplicateBackend):
        with pytest.raises(foldjax.PredictionOutputError, match="same structure|slot"):
            foldjax.predict(
                _request(tmp_path, output_dir=output, use_compile_cache=False)
            )

    assert (output / "first.cif").exists(), "validation must run before moving files"
    assert not (output / "foldjax_run.json").exists()


def test_hard_linked_structure_outputs_are_recognized_as_the_same_file(
    tmp_path: Path,
) -> None:
    class HardLinkBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            first = request.output_dir / "first.cif"
            second = request.output_dir / "second.cif"
            first.write_text("data_first\n#\n")
            second.hardlink_to(first)
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed,
                        structure_path=first,
                        metadata={"sample": 0},
                    ),
                    PredictionSample(
                        seed=request.seed,
                        structure_path=second,
                        metadata={"sample": 1},
                    ),
                ),
                output_dir=request.output_dir,
            )

    with backend_override("boltz2", HardLinkBackend):
        with pytest.raises(foldjax.PredictionOutputError, match="same structure"):
            foldjax.predict(_request(tmp_path, use_compile_cache=False))


def test_native_system_exit_becomes_a_prediction_error(tmp_path: Path) -> None:
    class ExitingBackend(DummyBackend):
        def predict(self, request: PredictionRequest) -> PredictionResult:
            raise SystemExit(7)

    output = tmp_path / "out"
    with backend_override("boltz2", ExitingBackend):
        with pytest.raises(
            foldjax.PredictionError, match="stopped its native runner: 7"
        ):
            foldjax.predict(
                _request(tmp_path, output_dir=output, use_compile_cache=False)
            )

    assert not (output / "foldjax_run.json").exists()


def test_an_output_path_that_is_a_file_fails_before_the_backend_runs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output-file"
    output.write_text("not a directory")
    backend = DummyBackend()
    with backend_override("boltz2", lambda: backend):
        with pytest.raises(NotADirectoryError, match="output_dir is not a directory"):
            foldjax.predict(
                _request(tmp_path, output_dir=output, use_compile_cache=False)
            )
    assert backend.seen is None


def test_manifest_redacts_secret_backend_options(tmp_path: Path) -> None:
    output = tmp_path / "out"
    secret = "do-not-record-this"
    backend = DummyBackend()
    with backend_override("boltz2", lambda: backend):
        foldjax.predict(
            _request(
                tmp_path,
                output_dir=output,
                use_compile_cache=False,
                options={
                    "password": secret,
                    "ssh_passphrase": secret,
                    "pwd": secret,
                    "headers": {"Authorization": f"Bearer {secret}", "safe": "ok"},
                    "batch_size": 2,
                },
            )
        )

    text = (output / "foldjax_run.json").read_text()
    payload = json.loads(text)
    assert secret not in text
    assert payload["options"] == {
        "password": "[REDACTED]",
        "ssh_passphrase": "[REDACTED]",
        "pwd": "[REDACTED]",
        "headers": {"Authorization": "[REDACTED]", "safe": "ok"},
        "batch_size": 2,
    }


def test_both_spellings_of_the_same_axis_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model"):
        _request(tmp_path, models=("boltz",))
    with pytest.raises(ValueError, match="input"):
        _request(tmp_path, inputs=(tmp_path / "job.yaml",))


def test_plural_fields_reject_bare_strings_with_a_clear_message(tmp_path: Path) -> None:
    source = tmp_path / "job.yaml"
    source.write_text("version: 1\n")
    with pytest.raises(ValueError, match="models must be a sequence"):
        PredictionRequest(models="boltz2", input=source)
    with pytest.raises(ValueError, match="inputs must be a sequence"):
        PredictionRequest(model="boltz2", inputs=str(source))


def test_plural_models_accept_a_one_shot_generator(tmp_path: Path) -> None:
    source = tmp_path / "job.yaml"
    source.write_text("version: 1\n")
    request = PredictionRequest(
        models=(name for name in ("boltz", "protenix")), input=source
    )
    assert request.models == ("boltz", "protenix")


def test_scalar_resolver_points_plural_callers_to_plural_resolver(
    tmp_path: Path,
) -> None:
    source = tmp_path / "job.yaml"
    source.write_text("version: 1\n")
    request = PredictionRequest(models=("boltz",), input=source)
    with pytest.raises(ValueError, match=r"resolve_requests\(\).+predict\(\)"):
        foldjax.resolve_request(request)


def test_request_rejects_an_input_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="is not a file"):
        PredictionRequest(model="boltz2", input=tmp_path)


def test_request_normalizes_and_copies_option_mappings(tmp_path: Path) -> None:
    source = {"  n_step  ": 20}
    request = _request(tmp_path, options=MappingProxyType(source))

    source["  n_step  "] = 99
    source["new"] = True

    assert request.options == {"n_step": 20}
    assert isinstance(request.options, dict)


@pytest.mark.parametrize("options", [None, [("n_step", 20)], "n_step=20"])
def test_request_requires_options_to_be_a_mapping(
    tmp_path: Path, options: object
) -> None:
    with pytest.raises(ValueError, match="options must be a mapping"):
        _request(tmp_path, options=options)


@pytest.mark.parametrize("key", ["", "   ", 7])
def test_request_requires_nonblank_string_option_keys(
    tmp_path: Path, key: object
) -> None:
    with pytest.raises(ValueError, match="options keys must be non-empty strings"):
        _request(tmp_path, options={key: 1})


def test_request_rejects_option_keys_that_collide_after_trimming(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="set more than once after trimming"):
        _request(tmp_path, options={"n_step": 20, " n_step ": 40})


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_request_requires_a_real_compile_cache_boolean(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(ValueError, match="use_compile_cache must be a boolean"):
        _request(tmp_path, use_compile_cache=value)
