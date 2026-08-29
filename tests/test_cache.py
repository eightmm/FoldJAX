import dataclasses
from pathlib import Path

import numpy as np
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.alphafold3 import AlphaFold3Backend
from foldjax.backends.base import Backend
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.backends.opendde import OpenDDEBackend
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


def test_esmfold2_fixed_defaults_share_the_omitted_cache_namespace(
    tmp_path: Path,
) -> None:
    backend = ESMFold2Backend()
    omitted = dataclasses.replace(
        _request(tmp_path), model="esmfold2", input_format="foldjax"
    )
    explicit = dataclasses.replace(
        omitted,
        options={
            "cp_devices": 1,
            "no_language_model": False,
            "max_msa_depth": 1024,
        },
    )

    assert backend.cache_profile(omitted) == {}
    assert backend.cache_profile(explicit) == backend.cache_profile(omitted)
    assert resolve_cache_dir(explicit, backend) == resolve_cache_dir(omitted, backend)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("cp_devices", 2),
        ("no_language_model", True),
        ("max_msa_depth", 1023),
        ("num_samples", 32),
        ("num_steps", 14),
        ("num_recycles", 3),
        ("cp_devices", np.int64(1)),
        ("max_msa_depth", np.int64(1024)),
        ("no_language_model", 0),
    ],
)
def test_esmfold2_unproven_defaults_and_lookalikes_stay_distinct(
    tmp_path: Path, name: str, value: object
) -> None:
    backend = ESMFold2Backend()
    omitted = dataclasses.replace(
        _request(tmp_path), model="esmfold2", input_format="foldjax"
    )
    changed = dataclasses.replace(omitted, options={name: value})

    assert backend.cache_profile(changed) != backend.cache_profile(omitted)
    assert resolve_cache_dir(changed, backend) != resolve_cache_dir(omitted, backend)


def test_alphafold3_released_defaults_share_the_omitted_cache_namespace(
    tmp_path: Path,
) -> None:
    backend = AlphaFold3Backend()
    omitted = dataclasses.replace(
        _request(tmp_path), model="alphafold3", input_format="native"
    )
    native = dataclasses.replace(
        omitted,
        options={
            "num_samples": 5,
            "num_steps": 200,
            "num_recycles": 10,
            "max_msa_depth": 1024,
            "buckets": [],
            "attention_backend": "triton",
            "return_embeddings": False,
            "return_distogram": False,
            "kernel_autotuning": "autotune",
        },
    )
    neutral = dataclasses.replace(
        omitted,
        num_samples=5,
        num_steps=200,
        num_recycles=10,
        max_msa_depth=1024,
        options={
            "buckets": (),
            "attention_kernel": "auto",
            "return_embeddings": False,
            "return_distogram": False,
            "kernel_autotuning": "autotune",
        },
    )

    assert backend.cache_profile(omitted) == {}
    assert backend.cache_profile(native) == backend.cache_profile(omitted)
    assert backend.cache_profile(neutral) == backend.cache_profile(omitted)
    assert resolve_cache_dir(native, backend) == resolve_cache_dir(omitted, backend)
    assert resolve_cache_dir(neutral, backend) == resolve_cache_dir(omitted, backend)


def test_alphafold3_external_source_keeps_nested_config_defaults_explicit(
    tmp_path: Path,
) -> None:
    backend = AlphaFold3Backend()
    request = dataclasses.replace(
        _request(tmp_path), model="alphafold3", input_format="native"
    )
    source = tmp_path / "external-alphafold3"
    omitted = dataclasses.replace(request, options={"source": source})
    explicit = dataclasses.replace(
        request,
        options={
            "source": source,
            "num_samples": 5,
            "num_steps": 200,
            "num_recycles": 10,
            "max_msa_depth": 1024,
            "attention_backend": "triton",
            "return_embeddings": False,
            "return_distogram": False,
            "kernel_autotuning": "autotune",
        },
    )

    assert backend.cache_profile(omitted) == {}
    assert backend.cache_profile(explicit) == {
        "num_steps": 200,
        "max_msa_depth": 1024,
    }
    assert resolve_cache_dir(explicit, backend) != resolve_cache_dir(omitted, backend)


@pytest.mark.parametrize(
    "options",
    (
        {"num_samples": 6},
        {"num_steps": 199},
        {"num_recycles": 11},
        {"max_msa_depth": 512},
        {"buckets": [256]},
        {"attention_backend": "xla"},
        {"return_embeddings": True},
        {"return_distogram": True},
        {"kernel_autotuning": "heuristics"},
        {"kernel_autotuning": "error"},
        {"num_samples": np.int64(5)},
    ),
)
def test_alphafold3_nondefault_and_type_routes_keep_distinct_cache_namespaces(
    tmp_path: Path,
    options: dict[str, object],
) -> None:
    backend = AlphaFold3Backend()
    omitted = dataclasses.replace(
        _request(tmp_path), model="alphafold3", input_format="native"
    )
    changed = dataclasses.replace(omitted, options=options)

    assert backend.cache_profile(changed) != backend.cache_profile(omitted)
    assert resolve_cache_dir(changed, backend) != resolve_cache_dir(omitted, backend)


def test_boltz2_released_defaults_share_the_omitted_cache_namespace(
    tmp_path: Path,
) -> None:
    backend = Boltz2Backend()
    omitted = _request(tmp_path)
    native = dataclasses.replace(
        omitted,
        options={
            "num_steps": 200,
            "num_recycles": 3,
            "num_samples": 1,
            "cp_atom_windows": True,
            "cp_devices": 1,
            "cp_layout": "1d",
            "affinity_num_steps": 200,
            "affinity_num_samples": 5,
            "compute_dtype": "bfloat16",
            "attention_backend": "xla",
            "trunk_atom_attention_backend": "xla",
            "triangle_backend": "cueq",
            "glu_backend": "xla",
            "bucket": False,
        },
    )
    neutral = dataclasses.replace(
        omitted,
        num_steps=200,
        num_recycles=3,
        num_samples=1,
        options={
            "cp_atom_windows": True,
            "cp_devices": 1,
            "cp_layout": "auto",
            "affinity_num_steps": 200,
            "affinity_num_samples": 5,
            "dtype": "bfloat16",
            "attention_kernel": "auto",
            "triangle_kernel": "auto",
            "glu_backend": "xla",
            "bucket": False,
        },
    )

    assert backend.cache_profile(omitted) == {}
    assert backend.cache_profile(native) == backend.cache_profile(omitted)
    assert backend.cache_profile(neutral) == backend.cache_profile(omitted)
    assert resolve_cache_dir(native, backend) == resolve_cache_dir(omitted, backend)
    assert resolve_cache_dir(neutral, backend) == resolve_cache_dir(omitted, backend)


def test_boltz2_inherited_atom_attention_backend_has_one_cache_identity(
    tmp_path: Path,
) -> None:
    backend = Boltz2Backend()
    omitted = _request(tmp_path)
    explicit_none = dataclasses.replace(
        omitted, options={"trunk_atom_attention_backend": None}
    )
    global_tokamax = dataclasses.replace(
        omitted, options={"attention_backend": "tokamax"}
    )
    repeated_tokamax = dataclasses.replace(
        omitted,
        options={
            "attention_backend": "tokamax",
            "trunk_atom_attention_backend": "tokamax",
        },
    )

    assert backend.cache_profile(explicit_none) == backend.cache_profile(omitted)
    assert backend.cache_profile(repeated_tokamax) == backend.cache_profile(
        global_tokamax
    )
    assert resolve_cache_dir(explicit_none, backend) == resolve_cache_dir(
        omitted, backend
    )
    assert resolve_cache_dir(repeated_tokamax, backend) == resolve_cache_dir(
        global_tokamax, backend
    )


def test_boltz2_cache_profile_normalizes_only_proven_cp_layout_aliases(
    tmp_path: Path,
) -> None:
    backend = Boltz2Backend()
    request = _request(tmp_path)
    cp_omitted = dataclasses.replace(request, options={"cp_devices": 4})
    cp_auto = dataclasses.replace(
        request, options={"cp_devices": 4, "cp_layout": "auto"}
    )
    cp_rows = dataclasses.replace(
        request, options={"cp_devices": 4, "cp_layout": "1d"}
    )
    cp_grid = dataclasses.replace(
        request, options={"cp_devices": 4, "cp_layout": "2d"}
    )
    cp_xla = dataclasses.replace(
        request,
        options={
            "cp_devices": 4,
            "cp_layout": "1d",
            "triangle_backend": "xla",
        },
    )

    assert backend.cache_profile(cp_omitted) == {"cp_devices": 4}
    assert backend.cache_profile(cp_auto) == backend.cache_profile(cp_omitted)
    assert backend.cache_profile(cp_rows) == backend.cache_profile(cp_omitted)
    assert resolve_cache_dir(cp_auto, backend) == resolve_cache_dir(cp_rows, backend)
    assert resolve_cache_dir(cp_grid, backend) != resolve_cache_dir(cp_rows, backend)
    # CP currently resolves the released cueq default to XLA internally. Keep
    # an explicitly requested XLA route separate until that conditional alias
    # has its own whole-control-flow proof.
    assert resolve_cache_dir(cp_xla, backend) != resolve_cache_dir(cp_rows, backend)


@pytest.mark.parametrize(
    ("request_fields", "options"),
    [
        ({"num_steps": 201}, {}),
        ({"num_recycles": 4}, {}),
        ({"num_samples": 2}, {}),
        ({}, {"cp_atom_windows": False}),
        ({}, {"affinity_num_steps": 201}),
        ({}, {"affinity_num_samples": 6}),
        ({}, {"compute_dtype": "float32"}),
        ({}, {"attention_backend": "tokamax"}),
        ({}, {"trunk_atom_attention_backend": "triton"}),
        ({}, {"triangle_backend": "xla"}),
        ({}, {"glu_backend": "tokamax"}),
        ({}, {"bucket": True}),
    ],
)
def test_boltz2_nondefault_compile_options_keep_distinct_namespaces(
    tmp_path: Path,
    request_fields: dict[str, object],
    options: dict[str, object],
) -> None:
    backend = Boltz2Backend()
    omitted = _request(tmp_path)
    changed = dataclasses.replace(omitted, options=options, **request_fields)

    assert resolve_cache_dir(changed, backend) != resolve_cache_dir(omitted, backend)


def test_opendde_released_defaults_share_the_omitted_cache_namespace(
    tmp_path: Path,
) -> None:
    backend = OpenDDEBackend()
    omitted = dataclasses.replace(_request(tmp_path), model="opendde")
    native = dataclasses.replace(
        omitted,
        options={
            "num_samples": 5,
            "num_steps": 200,
            "num_recycles": 10,
            "max_msa_depth": 16384,
            "n_queries": 32,
            "n_keys": 128,
            "diffusion_attention_backend": "xla_jit",
            "trunk_single_attention_backend": "xla_jit",
            "structural_single_attention_backend": "xla_jit",
            "trunk_dtype": "bf16",
            "chunk_policy": "auto",
            "cp_devices": 1,
            "cp_layout": "1d",
        },
    )
    neutral = dataclasses.replace(
        omitted,
        num_samples=5,
        num_steps=200,
        num_recycles=10,
        max_msa_depth=16384,
        options={
            "n_queries": 32,
            "n_keys": 128,
            "diffusion_attention_backend": "xla_jit",
            "structural_single_attention_backend": "xla_jit",
            "dtype": "bfloat16",
            "attention_kernel": "auto",
            "chunk_policy": "auto",
            "cp_devices": 1,
            "cp_layout": "auto",
        },
    )

    assert backend.cache_profile(omitted) == {"return_confidence_details": False}
    assert backend.cache_profile(native) == backend.cache_profile(omitted)
    assert backend.cache_profile(neutral) == backend.cache_profile(omitted)
    assert resolve_cache_dir(native, backend) == resolve_cache_dir(omitted, backend)
    assert resolve_cache_dir(neutral, backend) == resolve_cache_dir(omitted, backend)


def test_opendde_cache_profile_normalizes_only_proven_cp_layout_aliases(
    tmp_path: Path,
) -> None:
    backend = OpenDDEBackend()
    request = dataclasses.replace(_request(tmp_path), model="opendde")
    cp_omitted = dataclasses.replace(request, options={"cp_devices": 4})
    cp_auto = dataclasses.replace(
        request, options={"cp_devices": 4, "cp_layout": "auto"}
    )
    cp_rows = dataclasses.replace(request, options={"cp_devices": 4, "cp_layout": "1d"})
    cp_grid = dataclasses.replace(request, options={"cp_devices": 4, "cp_layout": "2d"})

    expected = {"cp_devices": 4, "return_confidence_details": False}
    assert backend.cache_profile(cp_omitted) == expected
    assert backend.cache_profile(cp_auto) == expected
    assert backend.cache_profile(cp_rows) == expected
    assert resolve_cache_dir(cp_auto, backend) == resolve_cache_dir(cp_rows, backend)
    assert resolve_cache_dir(cp_grid, backend) != resolve_cache_dir(cp_rows, backend)


@pytest.mark.parametrize(
    ("request_fields", "options"),
    [
        ({"num_samples": 6}, {}),
        ({"num_steps": 201}, {}),
        ({"num_recycles": 11}, {}),
        ({"max_msa_depth": 16383}, {}),
        ({}, {"n_queries": 31}),
        ({}, {"n_keys": 127}),
        ({}, {"diffusion_attention_backend": "xla"}),
        ({}, {"trunk_single_attention_backend": "xla"}),
        ({}, {"structural_single_attention_backend": "xla"}),
        ({}, {"trunk_dtype": "fp32"}),
        ({}, {"chunk_policy": "manual"}),
        ({}, {"chunk_policy": "off"}),
        ({}, {"triangle_mul_chunk_size": 128}),
        ({}, {"triangle_att_q_chunk_size": 128}),
        ({}, {"single_att_q_chunk_size": 128}),
        ({}, {"token_q_chunk_size": 128}),
        ({}, {"cp_devices": 2}),
        ({}, {"cp_layout": "2d"}),
        ({}, {"include_raw": True}),
    ],
)
def test_opendde_nondefault_compile_options_keep_distinct_namespaces(
    tmp_path: Path,
    request_fields: dict[str, object],
    options: dict[str, object],
) -> None:
    backend = OpenDDEBackend()
    omitted = dataclasses.replace(_request(tmp_path), model="opendde")
    changed = dataclasses.replace(omitted, options=options, **request_fields)

    assert resolve_cache_dir(changed, backend) != resolve_cache_dir(omitted, backend)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_samples", np.int64(5)),
        ("num_steps", np.int64(200)),
        ("cp_devices", np.int64(1)),
    ],
)
def test_opendde_default_lookalikes_with_another_type_stay_distinct(
    tmp_path: Path, name: str, value: object
) -> None:
    backend = OpenDDEBackend()
    omitted = dataclasses.replace(
        _request(tmp_path), model="opendde", input_format="foldjax"
    )
    lookalike = dataclasses.replace(omitted, options={name: value})

    backend.validate_request(lookalike)
    assert backend.cache_profile(lookalike) != backend.cache_profile(omitted)
    assert resolve_cache_dir(lookalike, backend) != resolve_cache_dir(omitted, backend)


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
    assert "all_arrays" not in serial

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
    explicit_default_arrays = dataclasses.replace(
        request, options={"all_arrays": False}
    )
    all_arrays = dataclasses.replace(request, options={"all_arrays": True})
    trunk_all_arrays = dataclasses.replace(trunk, options={"all_arrays": True})
    assert resolve_cache_dir(represented, backend) != resolve_cache_dir(
        request, backend
    )
    assert resolve_cache_dir(trunk, backend) != resolve_cache_dir(represented, backend)
    assert resolve_cache_dir(padded, backend) != resolve_cache_dir(request, backend)
    assert resolve_cache_dir(explicit_default_arrays, backend) == resolve_cache_dir(
        request, backend
    )
    assert resolve_cache_dir(all_arrays, backend) != resolve_cache_dir(request, backend)
    assert resolve_cache_dir(trunk_all_arrays, backend) == resolve_cache_dir(
        trunk, backend
    )


def test_openfold3_released_sampling_aliases_share_cache_namespace(
    tmp_path: Path,
) -> None:
    backend = OpenFold3Backend()
    request = dataclasses.replace(_request(tmp_path), model="openfold3")
    neutral_defaults = dataclasses.replace(
        request,
        num_samples=5,
        num_steps=200,
        # Neutral recycles count repeats; OpenFold3 executes one extra cycle.
        num_recycles=3,
        max_msa_depth=1024,
    )
    native_defaults = dataclasses.replace(
        request,
        options={
            "num_samples": 5,
            "num_steps": 200,
            "num_recycles": 4,
            "max_msa_depth": 1024,
        },
    )
    capped_msa = dataclasses.replace(request, max_msa_depth=4096)

    aliases = (request, neutral_defaults, native_defaults, capped_msa)
    profiles = [backend.cache_profile(alias) for alias in aliases]
    namespaces = [resolve_cache_dir(alias, backend) for alias in aliases]

    assert profiles.count(profiles[0]) == len(profiles)
    assert namespaces.count(namespaces[0]) == len(namespaces)
    for name in ("num_samples", "num_steps", "num_recycles", "max_msa_depth"):
        assert name not in profiles[0]


@pytest.mark.parametrize(
    "changes",
    (
        {"num_samples": 6},
        {"num_steps": 199},
        {"num_recycles": 4},
        {"max_msa_depth": 512},
    ),
)
def test_openfold3_nondefault_sampling_keeps_distinct_cache_namespace(
    tmp_path: Path,
    changes: dict[str, int],
) -> None:
    backend = OpenFold3Backend()
    request = dataclasses.replace(_request(tmp_path), model="openfold3")
    changed = dataclasses.replace(request, **changes)

    assert resolve_cache_dir(changed, backend) != resolve_cache_dir(request, backend)


@pytest.mark.parametrize("name", ("pair_chunk_size", "diffusion_chunk_size"))
def test_openfold3_chunk_choices_keep_distinct_cache_namespace(
    tmp_path: Path,
    name: str,
) -> None:
    backend = OpenFold3Backend()
    request = dataclasses.replace(_request(tmp_path), model="openfold3")
    changed = dataclasses.replace(request, options={name: 1})

    assert resolve_cache_dir(changed, backend) != resolve_cache_dir(request, backend)


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
