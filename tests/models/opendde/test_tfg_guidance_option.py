"""Upstream OpenDDE's `--use_tfg_guidance`, on Protenix's TFG port.

`runner/batch_inference.py:848-853,487` enables
`sample_diffusion.guidance` on the default mapping in
`opendde/config/model_base.py:66-128`, which is Protenix's; OpenDDE's
`opendde/tfg` is Protenix's `protenix/tfg` with Fold-CP plumbing added. The
guided step runs where `opendde/model/generator.py:234` calls it: after the
rigid augmentation and churn noise, in place of the Euler update.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.models.opendde import runner as predict_runner
from foldjax.models.opendde.cli import predict as predict_cli
from foldjax.models.opendde.models.sampling import sample_diffusion
from foldjax.models.protenix.tfg.config import (
    UPSTREAM_GUIDANCE_CONFIG,
    upstream_guidance_config,
)
from foldjax.schema import PaddingConfig, PredictionRequest
from tests.models.opendde.toy_params import inference_params

_UPSTREAM_MODEL_BASE = next(
    (
        parent / "OpenDDE" / "opendde" / "config" / "model_base.py"
        for parent in Path(__file__).resolve().parents
        if (parent / "OpenDDE" / "opendde" / "config" / "model_base.py").is_file()
    ),
    None,
)


def test_upstream_opendde_ships_protenixs_mapping() -> None:
    if _UPSTREAM_MODEL_BASE is None:
        pytest.skip("upstream OpenDDE checkout not present")
    tree = ast.parse(_UPSTREAM_MODEL_BASE.read_text())
    guidance = next(
        value
        for mapping in ast.walk(tree)
        if isinstance(mapping, ast.Dict)
        for key, value in zip(mapping.keys, mapping.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "guidance"
    )
    assert ast.literal_eval(guidance) == UPSTREAM_GUIDANCE_CONFIG


def _sample(guidance_config=None, guidance_features=None):
    n_atom = 2
    return sample_diffusion(
        lambda x, t_hat: 0.5 * x,
        jnp.asarray([4.0, 2.0, 1.0, 0.5]),
        num_samples=1,
        n_atom=n_atom,
        key=jax.random.PRNGKey(0),
        guidance_config=guidance_config,
        guidance_features=guidance_features,
    )


def _one_term(weight: float) -> dict[str, Any]:
    return {
        "enable": True,
        "mu": 0.5,
        "steps": {"tfg_inner": 3, "projection_outer": 0},
        "terms": {
            "InterchainBondPotential": {
                "interval": 1,
                "weight": weight,
                "buffer": 0.1,
            }
        },
    }


_BOND = {"interchain_bond_index": np.asarray([[0], [1]], dtype=np.int64)}


def test_an_inert_guidance_term_reproduces_the_unguided_sampler() -> None:
    """Same draws, same update: the guided branch sits where the Euler step was."""

    unguided = _sample()
    inert = _sample(_one_term(0.0), _BOND)
    np.testing.assert_allclose(np.asarray(inert), np.asarray(unguided), rtol=1e-6)


def test_an_active_term_moves_the_structure() -> None:
    unguided = np.asarray(_sample())
    guided = np.asarray(_sample(_one_term(1.0), _BOND))
    assert not np.allclose(guided, unguided)
    distance = lambda x: np.linalg.norm(x[0, 0] - x[0, 1])  # noqa: E731
    assert distance(guided) < distance(unguided), "a bond potential pulls together"


def test_a_disabled_mapping_is_no_guidance() -> None:
    np.testing.assert_array_equal(
        np.asarray(_sample(UPSTREAM_GUIDANCE_CONFIG, None)), np.asarray(_sample())
    )


def test_the_sampler_refuses_what_guidance_cannot_run() -> None:
    with pytest.raises(ValueError, match="guidance_features are required"):
        _sample(_one_term(1.0), None)
    with pytest.raises(ValueError, match="use_scan=False"):
        sample_diffusion(
            lambda x, t: x,
            jnp.asarray([2.0, 1.0]),
            num_samples=1,
            n_atom=2,
            key=jax.random.PRNGKey(0),
            use_scan=True,
            guidance_config=_one_term(1.0),
            guidance_features=_BOND,
        )


def _run_cli(tmp_path: Path, monkeypatch, *argv: str) -> list[dict[str, Any]]:
    job = {"name": "tiny", "modelSeeds": [101], "sequences": []}
    input_path = tmp_path / "tiny.json"
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    features = {"restype": np.zeros((2, 32), dtype=np.float32)}
    predicted: list[dict[str, Any]] = []
    prepared: list[dict] = []

    monkeypatch.setattr(predict_runner, "_load_jobs", lambda _path: [job])
    monkeypatch.setattr(predict_runner, "_featurize", lambda *_a, **_k: features)
    monkeypatch.setattr(
        predict_runner,
        "_load_prepared_params",
        lambda _path, _dtype: inference_params(),
    )

    def prepare(raw):
        prepared.append(raw)
        return {**raw, "geometry_unsupported": False}

    monkeypatch.setattr(
        "foldjax.models.protenix.data.geometry.prepare_tfg_features", prepare
    )

    def capture_predict(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        predicted.append(kwargs)
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    monkeypatch.setattr(predict_runner, "_predict", capture_predict)
    monkeypatch.setattr(predict_runner, "_score", lambda output, *a, **k: dict(output))
    monkeypatch.setattr(
        predict_runner,
        "_write",
        lambda root, **kwargs: [root / kwargs["job_name"] / "predictions" / "t.cif"],
    )
    predict_cli.main(
        [
            "--input-json", str(input_path),
            "--weights", str(weights),
            "--out", str(tmp_path / "out"),
            *argv,
        ]
    )  # fmt: skip
    if prepared:
        assert prepared == [features], "geometry comes off the dense featurization"
    return predicted


def test_the_cli_switch_runs_upstreams_mapping_on_the_eager_path(
    tmp_path: Path, monkeypatch
) -> None:
    (call,) = _run_cli(tmp_path, monkeypatch, "--use-tfg-guidance", "true")
    assert call["guidance_config"] == upstream_guidance_config()
    assert call["guidance_features"]["geometry_unsupported"] is False
    assert call["graph_jit"] is False, "guidance cannot run in the compiled graph"


def test_without_the_switch_the_call_is_unchanged(tmp_path: Path, monkeypatch) -> None:
    (call,) = _run_cli(tmp_path, monkeypatch)
    assert "guidance_config" not in call
    assert call["graph_jit"] is True


def _request(tmp_path: Path, **options) -> PredictionRequest:
    job = tmp_path / "job.json"
    job.write_text("[]")
    weights = tmp_path / "opendde.jax"
    weights.touch()
    return PredictionRequest(
        model="opendde",
        input=job,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        seed=101,
        options=options,
    )


def test_the_adapter_renders_the_switch(tmp_path: Path) -> None:
    invocation = OpenDDEBackend()._native_invocation(
        _request(tmp_path, use_tfg_guidance=True)
    )
    assert "--use-tfg-guidance" in invocation.argv
    assert invocation.config_fields["use_tfg_guidance"] is True
    assert (
        OpenDDEBackend()
        ._native_invocation(_request(tmp_path))
        .config_fields["use_tfg_guidance"]
        is False
    )


@pytest.mark.parametrize(
    ("options", "padding", "message"),
    [
        ({"use_tfg_guidance": "yes"}, None, "use_tfg_guidance must be a boolean"),
        ({"use_tfg_guidance": True}, PaddingConfig(), "padding with TFG guidance"),
        (
            {"use_tfg_guidance": True, "deterministic": "on"},
            None,
            "deterministic reductions",
        ),
        ({"use_tfg_guidance": True, "cp_devices": 2}, None, "cannot partition"),
    ],
)
def test_plan_refuses_what_the_guided_sampler_cannot_run(
    tmp_path: Path, options: dict, padding, message: str
) -> None:
    request = dataclasses.replace(_request(tmp_path, **options), padding=padding)
    with pytest.raises(ValueError, match=message):
        OpenDDEBackend().validate_request(request)


def test_the_switch_forks_the_namespace_and_false_does_not(tmp_path: Path) -> None:
    backend = OpenDDEBackend()
    omitted = resolve_cache_dir(_request(tmp_path), backend)
    assert resolve_cache_dir(_request(tmp_path, use_tfg_guidance=False), backend) == (
        omitted
    )
    assert resolve_cache_dir(_request(tmp_path, use_tfg_guidance=True), backend) != (
        omitted
    )


def test_predict_hands_the_guidance_to_the_eager_entry_point(monkeypatch) -> None:
    from foldjax.models.opendde.data.featurize_json import featurize_opendde_json
    from foldjax.models.opendde.models import model as model_impl

    seen: dict = {}

    def static(features, params, schedule, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(model_impl, "opendde_infer_static", static)
    job = {
        "name": "tiny",
        "modelSeeds": [101],
        "sequences": [{"proteinChain": {"sequence": "ACDEFGHIK"}}],
    }
    features = featurize_opendde_json(job, n_queries=2, n_keys=4, seed=101)
    mapping = upstream_guidance_config()
    predict_runner._predict(
        features,
        None,
        seed=101,
        num_samples=1,
        num_steps=2,
        num_recycles=1,
        n_queries=2,
        n_keys=4,
        diffusion_attention_backend="xla_jit",
        trunk_single_attention_backend="xla_jit",
        structural_single_attention_backend="xla_jit",
        graph_jit=False,
        guidance_config=mapping,
        guidance_features={"marker": 1},
    )
    assert seen["guidance_config"] is mapping
    assert seen["guidance_features"] == {"marker": 1}
    assert "use_sampler_scan" not in seen, "the unrolled sampler is the default"


def test_the_opendde_featurization_satisfies_the_guidance_contract() -> None:
    """The geometry terms read the featurizer's own arrays, unmodified."""

    from foldjax.models.opendde.data.featurize_json import featurize_opendde_json
    from foldjax.models.protenix.data.geometry import (
        prepare_tfg_features,
        require_supported_geometry,
    )
    from foldjax.models.protenix.tfg.config import parse_tfg_config, validate_features

    job = {
        "name": "tiny",
        "modelSeeds": [101],
        "sequences": [{"proteinChain": {"sequence": "ACDEFGHIK", "count": 2}}],
    }
    features = featurize_opendde_json(job, n_queries=2, n_keys=4, seed=101)
    guided = prepare_tfg_features(features)
    require_supported_geometry(guided)
    validate_features(guided, parse_tfg_config(upstream_guidance_config()).terms)
    n_atom = int(np.asarray(features["atom_to_token_idx"]).shape[0])
    index = np.asarray(guided["pairwise_distance_index"])
    assert index.size and index.max() < n_atom
