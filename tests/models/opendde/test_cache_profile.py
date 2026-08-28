from __future__ import annotations

import argparse
import dataclasses
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.backends.opendde as backend_impl
from foldjax.api import resolve_cache_dir
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.cache import compilation_cache_scope
from foldjax.models._jit_pool import BoundedJitPool
from foldjax.models.opendde.cli import predict as predict_cli
from foldjax.models.opendde.models import model as model_impl
from foldjax.models.protenix.chunking import resolve_chunk_config
from foldjax.schema import PredictionRequest


class _DefaultsCapturedError(Exception):
    pass


def test_cache_defaults_track_the_native_parser_and_cp_resolver(monkeypatch) -> None:
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

    actual = {name: captured[name] for name in backend_impl._RELEASED_COMPILE_DEFAULTS}
    for name, expected in backend_impl._RELEASED_COMPILE_DEFAULTS.items():
        assert type(actual[name]) is type(expected)
    assert actual == backend_impl._RELEASED_COMPILE_DEFAULTS
    assert set(actual) <= set(OpenDDEBackend.compile_options)
    assert model_impl._resolve_cp_layout("auto") == "1d"
    assert model_impl._resolve_cp_layout("1d") == "1d"
    assert model_impl._resolve_cp_layout("2d") == "2d"


def _request(tmp_path: Path, *, output: str = "out") -> PredictionRequest:
    input_path = tmp_path / "job.json"
    input_path.write_text(
        json.dumps([{"name": "tiny", "modelSeeds": [0], "sequences": []}]),
        encoding="utf-8",
    )
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    return PredictionRequest(
        model="opendde",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / output,
        cache_dir=tmp_path / "cache",
    )


def test_released_default_cache_aliases_reuse_one_bounded_native_owner(
    tmp_path: Path, monkeypatch
) -> None:
    omitted = _request(tmp_path)
    explicit = dataclasses.replace(
        omitted,
        output_dir=tmp_path / "out-explicit",
        options=dict(backend_impl._RELEASED_COMPILE_DEFAULTS, cp_layout="1d"),
    )
    backend = OpenDDEBackend()
    omitted_scope = resolve_cache_dir(omitted, backend)
    explicit_scope = resolve_cache_dir(explicit, backend)
    assert explicit_scope == omitted_scope

    counts = {"loads": 0, "traces": 0}
    resolved_keys: list[tuple[object, ...]] = []
    outputs: list[np.ndarray] = []

    def tiny_graph(value, *, compile_key):
        counts["traces"] += 1
        del compile_key
        return value + jnp.asarray(1.0, dtype=value.dtype)

    runner = BoundedJitPool(tiny_graph, static_argnames=("compile_key",))
    features = {
        "restype": np.zeros((2, 32), dtype=np.float32),
        "asym_id": np.zeros((2,), dtype=np.int32),
    }

    monkeypatch.setattr(predict_cli, "_load_jobs", lambda _path: [{"name": "tiny"}])
    monkeypatch.setattr(predict_cli, "_featurize", lambda *_args, **_kwargs: features)
    monkeypatch.setattr(predict_cli, "compact_msa_storage", lambda value: value)
    monkeypatch.setattr(predict_cli, "dedup_templates", lambda value: value)
    monkeypatch.setattr(
        predict_cli, "compact_ref_atom_category_storage", lambda value: value
    )

    def load_prepared(_path: Path, _dtype: str) -> object:
        counts["loads"] += 1
        return object()

    monkeypatch.setattr(predict_cli, "_load_prepared_params", load_prepared)

    def tiny_predict(
        value: dict[str, Any],
        _params: object,
        **kwargs: Any,
    ) -> dict[str, Any]:
        chunks = resolve_chunk_config(
            n_token=int(value["restype"].shape[-2]),
            num_samples=kwargs["num_samples"],
            policy=kwargs["chunk_policy"],
            **{
                name: chunk
                for name, chunk in kwargs["chunk_overrides"].items()
                if chunk is not None
            },
        )
        trunk_dtype = kwargs["trunk_dtype"]
        compile_key = (
            kwargs["num_samples"],
            kwargs["num_steps"],
            kwargs["num_recycles"],
            kwargs["n_queries"],
            kwargs["n_keys"],
            kwargs["diffusion_attention_backend"],
            kwargs["trunk_single_attention_backend"],
            kwargs["structural_single_attention_backend"],
            None if trunk_dtype is None else jnp.dtype(trunk_dtype).name,
            chunks.triangle_mul_chunk_size,
            chunks.triangle_att_q_chunk_size,
            chunks.single_att_q_chunk_size,
            chunks.token_q_chunk_size,
            chunks.diffusion_chunk_size,
            kwargs["cp_shards"],
            model_impl._resolve_cp_layout(kwargs["cp_layout"]),
            kwargs["run_confidence_scores"],
            kwargs["return_confidence_logits"],
            kwargs["return_confidence_details"],
            kwargs["capture_names"],
            kwargs["stop_after_trunk"],
            kwargs.get("preserve_prefix_rng", False),
        )
        resolved_keys.append(compile_key)
        coordinate = runner(
            jnp.asarray(1.0, dtype=jnp.float32), compile_key=compile_key
        )
        outputs.append(np.asarray(coordinate))
        return {"coordinate": coordinate.reshape(1, 1, 1)}

    monkeypatch.setattr(predict_cli, "_predict", tiny_predict)
    monkeypatch.setattr(
        predict_cli,
        "_score",
        lambda output, *_args, **_kwargs: output,
    )
    monkeypatch.setattr(
        predict_cli,
        "_write",
        lambda root, **kwargs: [
            root
            / kwargs["job_name"]
            / f"seed_{kwargs['seed']}"
            / "predictions"
            / "tiny_sample_0.cif"
        ],
    )
    from foldjax.models.opendde import postprocess

    monkeypatch.setattr(
        postprocess, "project_generated_output_features", lambda value: value
    )
    monkeypatch.setattr(backend, "_ccd_memory_scope", nullcontext)

    namespaced = (
        dataclasses.replace(omitted, cache_dir=omitted_scope),
        dataclasses.replace(explicit, cache_dir=explicit_scope),
    )
    with backend.session(namespaced):
        for request in namespaced:
            with compilation_cache_scope(request.cache_dir):
                backend.predict(request)

    assert resolved_keys[0] == resolved_keys[1]
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert counts == {"loads": 1, "traces": 1}
    assert runner._entry_count() == 1
    assert runner._cache_size() == 1


def test_native_pool_identity_keeps_every_non_alias_static_route_distinct() -> None:
    args = ({}, (), jnp.zeros((2,), dtype=jnp.float32))
    common = {
        "params_treedef": (),
        "params_flags": (),
        "cp_shards": 1,
        "cp_layout": "1d",
        "num_samples": 5,
        "num_recycles": 10,
        "n_queries": 32,
        "n_keys": 128,
        "diffusion_attention_backend": "xla_jit",
        "trunk_single_attention_backend": "xla_jit",
        "structural_single_attention_backend": "xla_jit",
        "trunk_dtype": jnp.bfloat16,
        "run_confidence_scores": True,
        "return_confidence_logits": False,
        "return_confidence_details": False,
        "return_representations": False,
        "capture_names": (),
        "stop_after_trunk": False,
        "preserve_prefix_rng": False,
    }

    def identity(**changes: object) -> object:
        return model_impl._compiled_opendde_infer._identity(args, {**common, **changes})

    identities = [
        identity(),
        identity(trunk_triangle_attention_backend=None),
        identity(structural_triangle_attention_backend=None),
        identity(confidence_triangle_attention_backend=None),
        identity(diffusion_attention_backend="xla"),
        identity(trunk_single_attention_backend="xla"),
        identity(structural_single_attention_backend="xla"),
        identity(return_confidence_logits=True),
        identity(return_confidence_details=True),
        identity(return_confidence_logits=True, return_confidence_details=True),
        identity(return_representations=True),
        identity(capture_names=("pair",)),
        identity(stop_after_trunk=True),
        identity(preserve_prefix_rng=True),
        identity(trunk_dtype=None),
        identity(diffusion_chunk_size=1),
        identity(triangle_mul_chunk_size=128),
        identity(triangle_att_q_chunk_size=128),
        identity(single_att_q_chunk_size=128),
        identity(token_q_chunk_size=128),
        identity(cp_shards=4),
        identity(cp_layout="2d"),
        identity(num_samples=6),
        model_impl._compiled_opendde_infer._identity(
            ({}, (), jnp.zeros((3,), dtype=jnp.float32)), common
        ),
    ]

    assert len(set(identities)) == len(identities)
