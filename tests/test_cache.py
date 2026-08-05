import dataclasses
from pathlib import Path

import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.base import Backend
from foldjax.cache import cache_namespace, runtime_profile, weight_identity
from foldjax.schema import ModelCapabilities, PredictionRequest, PredictionResult


class ProfiledBackend(Backend):
    name = "boltz2"
    compile_options = ("steps",)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model=self.name, input_formats=("native",))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        raise AssertionError("not called")


def _request(tmp_path: Path, **options) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("version: 1\n")
    weights = tmp_path / "boltz2_conf"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model="boltz2",
        input=input_path,
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=options,
    )


def test_cache_namespace_is_backend_and_profile_specific(tmp_path: Path) -> None:
    path = cache_namespace(
        tmp_path,
        model="boltz2",
        weight_id="boltz2/conf weights",
        profile={"dtype": "bf16", "tokens": 512, "steps": 200},
    )
    assert path.parent.parent.name == "boltz2"
    assert path.parent.name == "boltz2_conf_weights"
    assert len(path.name) == 16
    assert path == cache_namespace(
        tmp_path,
        model="boltz2",
        weight_id="boltz2/conf weights",
        profile={"steps": 200, "tokens": 512, "dtype": "bf16"},
    )


def test_cache_namespace_falls_back_to_a_usable_directory_name(tmp_path: Path) -> None:
    path = cache_namespace(tmp_path, model="protenix", weight_id="///", profile={})
    assert path.parent.name == "weights"


def test_weight_identity_separates_same_named_checkpoints(tmp_path: Path) -> None:
    first = tmp_path / "a" / "model.pt"
    second = tmp_path / "b" / "model.pt"
    for index, path in enumerate((first, second)):
        path.parent.mkdir()
        path.write_bytes(b"x" * (index + 1))
    assert weight_identity(first)[0] == weight_identity(second)[0] == "model.pt"
    assert weight_identity(first)[1] != weight_identity(second)[1]


def test_weight_identity_of_a_directory_omits_size(tmp_path: Path) -> None:
    label, identity = weight_identity(tmp_path)
    assert label == tmp_path.name
    assert identity == str(tmp_path.resolve())


def test_runtime_profile_reports_the_active_jax_runtime() -> None:
    profile = runtime_profile()
    assert profile["jax"]
    assert profile["platform"]


def test_resolve_cache_dir_partitions_by_backend_and_compile_options(
    tmp_path: Path,
) -> None:
    backend = ProfiledBackend()
    base = resolve_cache_dir(_request(tmp_path, steps=200), backend)
    assert base.is_relative_to(tmp_path / "cache" / "boltz2" / "boltz2_conf")

    # A compile-relevant option changes the namespace.
    assert resolve_cache_dir(_request(tmp_path, steps=50), backend) != base
    # An option this backend does not compile against does not.
    unrelated = resolve_cache_dir(
        _request(tmp_path, steps=200, write_fmt="pdb"), backend
    )
    assert unrelated == base


def test_resolve_cache_dir_requires_a_cache_root(tmp_path: Path) -> None:
    request = dataclasses.replace(_request(tmp_path), cache_dir=None)
    with pytest.raises(ValueError, match="cache_dir is required"):
        resolve_cache_dir(request, ProfiledBackend())
