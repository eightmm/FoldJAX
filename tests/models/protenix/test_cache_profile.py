from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.backends.protenix as backend_impl
from foldjax.api import resolve_cache_dir
from foldjax.backends.protenix import ProtenixBackend
from foldjax.cache import compilation_cache_scope
from foldjax.models._jit_pool import BoundedJitPool
from foldjax.models.protenix.cli import predict as predict_cli
from foldjax.models.protenix.models import model as model_impl
from foldjax.models.protenix.models import predict as predict_impl
from foldjax.models.protenix.runtime_policy import (
    KNOWN_MODEL_NAMES,
    MODEL_INFERENCE_DEFAULTS,
)
from foldjax.schema import PredictionRequest

from .test_model import _toy_features, _toy_params


class _DefaultsCapturedError(Exception):
    pass


def test_cache_defaults_track_native_parser_model_policy_and_cp_resolver(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_defaults(
        parser: argparse.ArgumentParser,
        _args: object = None,
        _namespace: object = None,
    ) -> None:
        captured.update(
            (action.dest, action.default)
            for action in parser._actions
            if action.dest != argparse.SUPPRESS
        )
        raise _DefaultsCapturedError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_defaults)
    with pytest.raises(_DefaultsCapturedError):
        predict_cli.main([])

    actual = {
        name: captured[name] for name in backend_impl._RELEASED_COMPILE_DEFAULTS
    }
    for name, expected in backend_impl._RELEASED_COMPILE_DEFAULTS.items():
        assert type(actual[name]) is type(expected)
    assert actual == backend_impl._RELEASED_COMPILE_DEFAULTS
    assert set(actual) <= set(ProtenixBackend.compile_options)

    # The backend imports the native policy authority rather than copying its
    # base/mini schedules. Every parser-recognized model therefore has exactly
    # one schedule source.
    assert backend_impl.MODEL_INFERENCE_DEFAULTS is MODEL_INFERENCE_DEFAULTS
    assert set(MODEL_INFERENCE_DEFAULTS) == set(KNOWN_MODEL_NAMES)

    resolved: list[tuple[int, str]] = []

    @contextmanager
    def fake_context_parallel(
        n_devices: int,
        *,
        layout: str = "1d",
        devices: object = None,
    ) -> Iterator[None]:
        del devices
        resolved.append((n_devices, layout))
        yield None

    class CapturePool:
        def __call__(self, *_args: object, **kwargs: object) -> dict[str, object]:
            resolved.append((int(kwargs["cp_shards"]), str(kwargs["cp_layout"])))
            return {}

    monkeypatch.setattr(model_impl, "context_parallel", fake_context_parallel)
    monkeypatch.setattr(model_impl, "replicate_tree", lambda value: value)
    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", CapturePool())
    features = {"asym_id": jnp.asarray([0], dtype=jnp.int32)}
    noise_schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    for layout in ("auto", "1d", "2d"):
        model_impl.protenix_infer_compiled(
            features,
            (),
            noise_schedule,
            cp_shards=4,
            cp_layout=layout,
        )

    assert resolved == [
        (4, "1d"),
        (4, "1d"),
        (4, "1d"),
        (4, "1d"),
        (4, "2d"),
        (4, "2d"),
    ]


def _request(
    tmp_path: Path,
    *,
    model_name: str = "protenix_base_default_v1.0.0",
    output: str = "out",
    options: dict[str, object] | None = None,
) -> PredictionRequest:
    input_path = tmp_path / "job.json"
    input_path.write_text(
        json.dumps([{"name": "tiny", "modelSeeds": [0], "sequences": []}]),
        encoding="utf-8",
    )
    weights = tmp_path / "protenix.jax"
    weights.write_bytes(b"native fixture")
    merged = {"model_name": model_name, **(options or {})}
    return PredictionRequest(
        model="protenix",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / output,
        cache_dir=tmp_path / "cache",
        options=merged,
    )


@pytest.mark.parametrize("model_name", tuple(MODEL_INFERENCE_DEFAULTS))
@pytest.mark.parametrize("empty_cli_args", ((), []), ids=("tuple", "list"))
def test_known_model_released_defaults_share_the_omitted_cache_namespace(
    tmp_path: Path,
    model_name: str,
    empty_cli_args: list[object] | tuple[()],
) -> None:
    backend = ProtenixBackend()
    schedule = MODEL_INFERENCE_DEFAULTS[model_name]
    omitted = _request(tmp_path, model_name=model_name)
    explicit = _request(
        tmp_path,
        model_name=model_name,
        output="explicit",
        options={
            **backend_impl._RELEASED_COMPILE_DEFAULTS,
            "num_steps": schedule["num_steps"],
            "num_recycles": schedule["num_recycles"],
            "cp_layout": "1d",
            "cli_args": empty_cli_args,
            "output_format": "protenix",
        },
    )

    expected = {"model_name": model_name, "return_confidence_details": False}
    assert backend.cache_profile(omitted) == expected
    assert backend.cache_profile(explicit) == expected
    assert resolve_cache_dir(explicit, backend) == resolve_cache_dir(omitted, backend)


def test_neutral_released_defaults_share_the_native_and_omitted_namespace(
    tmp_path: Path,
) -> None:
    backend = ProtenixBackend()
    omitted = _request(tmp_path)
    neutral = dataclasses.replace(
        omitted,
        output_dir=tmp_path / "neutral",
        num_samples=5,
        num_steps=200,
        num_recycles=10,
        max_msa_depth=16384,
        options={
            "model_name": "protenix_base_default_v1.0.0",
            "dtype": "bfloat16",
            "attention_kernel": "auto",
            "diffusion_attention_backend": "xla_jit",
            "chunk_policy": "auto",
            "cp_devices": 1,
            "cp_layout": "auto",
            "cli_args": (),
        },
    )

    assert backend.cache_profile(neutral) == backend.cache_profile(omitted)
    assert resolve_cache_dir(neutral, backend) == resolve_cache_dir(omitted, backend)


@pytest.mark.parametrize("model_name", ("auto", "unknown"))
def test_unresolved_model_schedules_remain_distinct(
    tmp_path: Path, model_name: str
) -> None:
    backend = ProtenixBackend()
    omitted = _request(tmp_path, model_name=model_name)
    explicit = _request(
        tmp_path,
        model_name=model_name,
        output="explicit",
        options={"num_steps": 200, "num_recycles": 10},
    )

    assert backend.cache_profile(explicit) != backend.cache_profile(omitted)
    assert resolve_cache_dir(explicit, backend) != resolve_cache_dir(omitted, backend)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_samples", np.int64(5)),
        ("num_steps", np.int64(200)),
        ("num_recycles", np.int64(10)),
        ("max_msa_depth", np.int64(16384)),
        ("cp_devices", np.int64(1)),
    ],
)
def test_default_lookalikes_with_another_type_stay_distinct(
    tmp_path: Path, name: str, value: object
) -> None:
    backend = ProtenixBackend()
    omitted = _request(tmp_path)
    lookalike = _request(tmp_path, output="lookalike", options={name: value})

    backend.validate_request(lookalike)
    assert backend.cache_profile(lookalike) != backend.cache_profile(omitted)
    assert resolve_cache_dir(lookalike, backend) != resolve_cache_dir(omitted, backend)


@pytest.mark.parametrize(
    "changes",
    (
        {"num_samples": 6},
        {"num_steps": 201},
        {"num_recycles": 11},
        {"max_msa_depth": 16383},
        {"trunk_dtype": "fp32"},
        {"diffusion_attention_backend": "xla"},
        {"trunk_single_attention_backend": "xla"},
        {"trunk_triangle_attention_backend": "cueq_jit"},
        {"confidence_triangle_attention_backend": "cueq_jit"},
        {"chunk_policy": "manual"},
        {"chunk_policy": "off"},
        {"triangle_mul_chunk_size": 512},
        {"triangle_att_q_chunk_size": 512},
        {"single_att_q_chunk_size": 512},
        {"token_q_chunk_size": 512},
        {"opm_chunk_size": 512},
        {"diffusion_chunk_size": 5},
        {"cli_args": ("--gamma0", "0.8")},
        {"cp_devices": 2},
        {"cp_layout": "2d"},
        {"output_format": "npz"},
    ),
)
def test_nondefault_and_conditional_routes_keep_distinct_namespaces(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    backend = ProtenixBackend()
    omitted = _request(tmp_path)
    changed = _request(tmp_path, output="changed", options=changes)

    assert backend.cache_profile(changed) != backend.cache_profile(omitted)
    assert resolve_cache_dir(changed, backend) != resolve_cache_dir(omitted, backend)


def test_cache_profile_normalizes_only_proven_cp_layout_aliases(
    tmp_path: Path,
) -> None:
    backend = ProtenixBackend()
    serial = _request(tmp_path)
    serial_auto = _request(
        tmp_path, output="serial-auto", options={"cp_devices": 1, "cp_layout": "auto"}
    )
    serial_rows = _request(
        tmp_path, output="serial-rows", options={"cp_devices": 1, "cp_layout": "1d"}
    )
    cp_auto = _request(
        tmp_path, output="cp-auto", options={"cp_devices": 4, "cp_layout": "auto"}
    )
    cp_rows = _request(
        tmp_path, output="cp-rows", options={"cp_devices": 4, "cp_layout": "1d"}
    )
    cp_grid = _request(
        tmp_path, output="cp-grid", options={"cp_devices": 4, "cp_layout": "2d"}
    )

    assert backend.cache_profile(serial_auto) == backend.cache_profile(serial_rows)
    assert backend.cache_profile(serial_auto) == backend.cache_profile(serial)
    assert backend.cache_profile(cp_auto) == backend.cache_profile(cp_rows)
    assert backend.cache_profile(cp_auto)["cp_devices"] == 4
    assert resolve_cache_dir(cp_auto, backend) == resolve_cache_dir(cp_rows, backend)
    assert resolve_cache_dir(cp_auto, backend) != resolve_cache_dir(serial, backend)
    assert resolve_cache_dir(cp_grid, backend) != resolve_cache_dir(cp_rows, backend)


def test_native_pool_identity_keeps_every_nonalias_static_route_distinct() -> None:
    args = ({}, (), jnp.zeros((6,), dtype=jnp.float32))
    common = {
        "params_treedef": (),
        "params_flags": (),
        "cp_shards": 1,
        "cp_layout": "1d",
        "num_samples": 5,
        "num_recycles": 4,
        "n_queries": 2,
        "n_keys": 4,
        "input_atom_heads": 1,
        "atom_encoder_heads": 1,
        "token_heads": 1,
        "atom_decoder_heads": 1,
        "diffusion_attention_backend": "xla_jit",
        "trunk_single_attention_backend": "xla_jit",
        "trunk_triangle_attention_backend": "xla_jit",
        "confidence_triangle_attention_backend": "xla_jit",
        "trunk_dtype": jnp.bfloat16,
        "gamma0": 0.0,
        "step_scale_eta": 1.0,
        "run_confidence": True,
        "run_confidence_scores": True,
        "return_trunk": False,
        "stop_after_trunk": False,
        "capture_names": (),
        "return_confidence_logits": False,
        "return_confidence_details": False,
        "preserve_prefix_rng": False,
    }

    def identity(**changes: object) -> object:
        return model_impl._compiled_protenix_infer._identity(
            args, {**common, **changes}
        )

    identities = [
        identity(),
        identity(trunk_triangle_attention_backend=None),
        identity(confidence_triangle_attention_backend=None),
        identity(diffusion_attention_backend="xla"),
        identity(trunk_single_attention_backend="xla"),
        identity(return_confidence_logits=True),
        identity(return_confidence_details=True),
        identity(return_trunk=True),
        identity(capture_names=("pair",)),
        identity(stop_after_trunk=True),
        identity(preserve_prefix_rng=True),
        identity(trunk_dtype=None),
        identity(diffusion_chunk_size=5),
        identity(triangle_mul_chunk_size=512),
        identity(triangle_att_q_chunk_size=512),
        identity(single_att_q_chunk_size=512),
        identity(token_q_chunk_size=512),
        identity(opm_chunk_size=512),
        identity(cp_shards=4),
        identity(cp_layout="2d"),
        identity(num_samples=6),
        identity(gamma0=0.8),
        identity(step_scale_eta=1.5),
        model_impl._compiled_protenix_infer._identity(
            ({}, (), jnp.zeros((201,), dtype=jnp.float32)), common
        ),
    ]

    assert len(set(identities)) == len(identities)


def test_released_default_aliases_reuse_one_real_bounded_native_owner(
    tmp_path: Path, monkeypatch
) -> None:
    common_cli_args = (
        "--prewarm-only",
        "--input-atom-heads",
        "1",
        "--atom-encoder-heads",
        "1",
        "--token-heads",
        "1",
        "--atom-decoder-heads",
        "1",
        "--n-queries",
        "2",
        "--n-keys",
        "4",
        "--sigma-data",
        "4.0",
    )
    fixed_routes = {
        "model_name": "protenix_mini_default_v0.5.0",
        "trunk_triangle_attention_backend": "xla_jit",
        "confidence_triangle_attention_backend": "xla_jit",
        "cli_args": common_cli_args,
    }
    omitted = _request(
        tmp_path,
        model_name="protenix_mini_default_v0.5.0",
        options={
            key: value
            for key, value in fixed_routes.items()
            if key != "model_name"
        },
    )
    explicit = _request(
        tmp_path,
        model_name="protenix_mini_default_v0.5.0",
        output="explicit",
        options={
            **fixed_routes,
            **backend_impl._RELEASED_COMPILE_DEFAULTS,
            "num_steps": 5,
            "num_recycles": 4,
            "cp_layout": "1d",
            "output_format": "protenix",
        },
    )
    backend = ProtenixBackend()
    omitted_scope = resolve_cache_dir(omitted, backend)
    explicit_scope = resolve_cache_dir(explicit, backend)
    assert explicit_scope == omitted_scope

    counts = {"features": 0, "loads": 0, "traces": 0}
    feature_depths: list[int] = []
    outputs: list[np.ndarray] = []
    owner_identities: list[object] = []

    class RecordingPool(BoundedJitPool):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            owner_identities.append(self._identity(args, kwargs))
            return super().__call__(*args, **kwargs)

    real_graph = model_impl._protenix_infer_graph

    def counted_graph(*args: Any, **kwargs: Any) -> dict[str, jax.Array]:
        counts["traces"] += 1
        return real_graph(*args, **kwargs)

    runner = RecordingPool(
        counted_graph,
        static_argnames=(
            *model_impl.GRAPH_STATIC_ARGNAMES,
            "params_treedef",
            "params_flags",
        ),
    )
    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", runner)

    from foldjax.models.protenix.data import featurize_json

    def tiny_featurize(*_args: object, **kwargs: object) -> dict[str, Any]:
        counts["features"] += 1
        feature_depths.append(int(kwargs["max_msa_depth"]))
        return dict(_toy_features())

    monkeypatch.setattr(featurize_json, "featurize_protein_json", tiny_featurize)

    def load_prepared(_path: Path, trunk_dtype: str) -> object:
        counts["loads"] += 1
        assert trunk_dtype == "bf16"
        return _toy_params()

    monkeypatch.setattr(predict_cli, "_load_prepared_params", load_prepared)
    real_predict = predict_impl.protenix_predict_static

    def capture_predict(*args: Any, **kwargs: Any) -> dict[str, jax.Array]:
        output = real_predict(*args, **kwargs)
        outputs.append(np.asarray(output["coordinate"]))
        return output

    monkeypatch.setattr(predict_impl, "protenix_predict_static", capture_predict)
    monkeypatch.setattr(backend, "_ccd_memory_scope", nullcontext)

    namespaced = (
        dataclasses.replace(omitted, cache_dir=omitted_scope),
        dataclasses.replace(explicit, cache_dir=explicit_scope),
    )
    with backend.session(namespaced):
        for request in namespaced:
            with compilation_cache_scope(request.cache_dir):
                backend.predict(request)

    assert feature_depths == [16384, 16384]
    assert owner_identities[0] == owner_identities[1]
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert counts == {"features": 2, "loads": 1, "traces": 1}
    assert runner._entry_count() == 1
    assert runner._cache_size() == 1
