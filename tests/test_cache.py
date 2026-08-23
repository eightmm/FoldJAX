import dataclasses
from pathlib import Path

import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.base import Backend
from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.cache import (
    CacheSnapshot,
    cache_namespace,
    cache_snapshot,
    compilation_cache_scope,
    runtime_profile,
    weight_identity,
)
from foldjax.registry import available_models, get_backend
from foldjax.schema import ModelCapabilities, PredictionRequest, PredictionResult


class ProfiledBackend(Backend):
    name = "boltz2"
    compile_options = ("steps",)
    sampling_options = {"num_steps": "steps"}

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model=self.name, input_formats=("native",))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        raise AssertionError("not called")


def test_cache_snapshot_counts_regular_files_without_following_symlinks(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    nested = cache / "nested"
    nested.mkdir(parents=True)
    (cache / "one").write_bytes(b"123")
    (nested / "two").write_bytes(b"4567")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large").write_bytes(b"x" * 100)
    (cache / "linked").symlink_to(outside, target_is_directory=True)

    snapshot = cache_snapshot(cache)

    assert snapshot == CacheSnapshot(files=2, bytes=7)
    assert snapshot.summary() == {"files": 2, "bytes": 7}


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


def test_neutral_and_native_sampling_share_one_cache_namespace(
    tmp_path: Path,
) -> None:
    backend = ProfiledBackend()
    native = resolve_cache_dir(_request(tmp_path, steps=200), backend)
    neutral = resolve_cache_dir(
        dataclasses.replace(_request(tmp_path), num_steps=200), backend
    )

    assert neutral == native


@pytest.mark.parametrize("model", available_models())
def test_every_sampling_knob_participates_in_the_compile_profile(model: str) -> None:
    backend = get_backend(model)
    assert set(backend.sampling_options.values()) <= set(backend.compile_options)


def test_compile_option_types_are_part_of_the_cache_profile(tmp_path: Path) -> None:
    backend = ProfiledBackend()
    integer = resolve_cache_dir(_request(tmp_path, steps=1), backend)
    text = resolve_cache_dir(_request(tmp_path, steps="1"), backend)

    assert integer != text


def test_openfold3_triangle_kernel_changes_the_cache_namespace(tmp_path: Path) -> None:
    backend = OpenFold3Backend()
    request = dataclasses.replace(_request(tmp_path), model="openfold3")
    cueq = resolve_cache_dir(
        dataclasses.replace(request, options={"triangle_kernel": "cueq"}), backend
    )
    xla = resolve_cache_dir(
        dataclasses.replace(request, options={"triangle_kernel": "xla"}), backend
    )

    assert cueq != xla


def test_openfold3_cache_profile_names_every_static_runtime_route(
    tmp_path: Path,
) -> None:
    backend = OpenFold3Backend()
    request = dataclasses.replace(_request(tmp_path), model="openfold3")

    serial = backend.cache_profile(request)
    assert serial["cp_devices"] == 1
    assert serial["cp_layout"] == "serial"
    assert serial["triangle_kernel"] == "cueq"
    assert serial["representations"] == ()
    assert serial["stop_after"] == "full"
    assert serial["rng_route"] == "native"

    automatic = dataclasses.replace(
        request, options={"cp_devices": 4, "cp_layout": "auto"}
    )
    rows = dataclasses.replace(
        request, options={"cp_devices": 4, "cp_layout": "1d"}
    )
    grid = dataclasses.replace(
        request, options={"cp_devices": 4, "cp_layout": "2d"}
    )
    assert backend.cache_profile(automatic) == backend.cache_profile(rows)
    assert backend.cache_profile(automatic)["triangle_kernel"] == "xla"
    assert resolve_cache_dir(automatic, backend) != resolve_cache_dir(grid, backend)

    represented = dataclasses.replace(request, representations=("pair",))
    trunk = dataclasses.replace(represented, stop_after="trunk")
    padded = dataclasses.replace(request, padding=True)
    assert resolve_cache_dir(represented, backend) != resolve_cache_dir(
        request, backend
    )
    assert resolve_cache_dir(trunk, backend) != resolve_cache_dir(represented, backend)
    assert resolve_cache_dir(padded, backend) != resolve_cache_dir(request, backend)


def test_compilation_cache_scope_disables_and_restores_host_config(
    tmp_path: Path, monkeypatch
) -> None:
    import jax
    from jax.experimental.compilation_cache import compilation_cache

    names = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
    )
    original = {name: getattr(jax.config, name) for name in names}
    original_reset = compilation_cache.reset_cache
    resets: list[None] = []

    def reset_cache() -> None:
        resets.append(None)
        original_reset()

    monkeypatch.setattr(compilation_cache, "reset_cache", reset_cache)
    host = tmp_path / "host-cache"
    request = tmp_path / "request-cache"
    try:
        jax.config.update("jax_compilation_cache_dir", str(host))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 7.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 2048)
        configured = {name: getattr(jax.config, name) for name in names}

        with pytest.raises(RuntimeError, match="prediction failed"):
            with compilation_cache_scope(None):
                assert jax.config.jax_compilation_cache_dir is None
                assert jax.config.jax_persistent_cache_min_compile_time_secs == 7.0
                assert jax.config.jax_persistent_cache_min_entry_size_bytes == 2048
                raise RuntimeError("prediction failed")
        assert {name: getattr(jax.config, name) for name in names} == configured

        with compilation_cache_scope(request):
            assert jax.config.jax_compilation_cache_dir == str(request)
            assert jax.config.jax_persistent_cache_min_compile_time_secs == 1.0
            assert request.is_dir()
        assert {name: getattr(jax.config, name) for name in names} == configured
        assert len(resets) == 4
    finally:
        for name, value in original.items():
            jax.config.update(name, value)


def test_resolve_cache_dir_requires_a_cache_root(tmp_path: Path) -> None:
    request = dataclasses.replace(_request(tmp_path), cache_dir=None)
    with pytest.raises(ValueError, match="cache_dir is required"):
        resolve_cache_dir(request, ProfiledBackend())
