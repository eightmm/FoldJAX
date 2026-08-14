from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from foldjax.backends.base import Backend
from foldjax.cli import _parser, _request
from foldjax.models._random import masked_prefix_draw
from foldjax.padding import PaddingPlan, resolve_axis
from foldjax.registry import capabilities
from foldjax.schema import (
    ModelCapabilities,
    PaddingConfig,
    PredictionRequest,
    PredictionResult,
)


def _input(tmp_path: Path) -> Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"entities": []}), encoding="utf-8")
    return path


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "256"])
def test_padding_targets_are_strict_positive_integers(value: object) -> None:
    with pytest.raises(ValueError, match=r"padding\.tokens"):
        PaddingConfig(tokens=value)  # type: ignore[arg-type]


def test_padding_config_normalizes_numpy_integers_and_summarizes() -> None:
    config = PaddingConfig(tokens=np.int64(512), msa=np.int32(128))

    assert config.tokens == 512 and type(config.tokens) is int
    assert config.msa == 128 and type(config.msa) is int
    assert config.explicit_axes == ("tokens", "msa")
    assert config.summary() == {
        "tokens": 512,
        "atoms": None,
        "msa": 128,
        "templates": None,
        "structural_tokens": None,
        "language_model_tokens": None,
        "overflow": "error",
    }


@pytest.mark.parametrize("value", ["grow", 1, None])
def test_padding_overflow_is_strict(value: object) -> None:
    with pytest.raises(ValueError, match="padding.overflow"):
        PaddingConfig(overflow=value)  # type: ignore[arg-type]


def test_request_accepts_simple_boolean_and_mapping_padding(tmp_path: Path) -> None:
    path = _input(tmp_path)

    automatic = PredictionRequest(model="boltz2", input=path, padding=True)
    pinned = PredictionRequest(
        model="boltz2",
        input=path,
        padding={"tokens": 512, "overflow": "exact"},
    )
    disabled = PredictionRequest(model="boltz2", input=path, padding=False)

    assert automatic.padding == PaddingConfig()
    assert pinned.padding == PaddingConfig(tokens=512, overflow="exact")
    assert disabled.padding is None


def test_request_rejects_unknown_padding_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported padding fields: 'token'"):
        PredictionRequest(
            model="boltz2",
            input=_input(tmp_path),
            padding={"token": 512},
        )


def test_resolve_axis_selects_standard_pinned_and_overflow_targets() -> None:
    automatic = PaddingConfig()
    assert resolve_axis(300, automatic, "tokens") == 512
    assert resolve_axis(300, PaddingConfig(tokens=768), "tokens") == 768
    assert (
        resolve_axis(5000, PaddingConfig(overflow="exact"), "tokens") == 5000
    )
    with pytest.raises(ValueError, match="largest standard bucket 4096"):
        resolve_axis(5000, automatic, "tokens")
    with pytest.raises(ValueError, match="smaller than the input size 600"):
        resolve_axis(
            300,
            PaddingConfig(tokens=512),
            "tokens",
            minimum=600,
        )


def test_padding_plan_reports_real_storage_and_target_shapes() -> None:
    plan = PaddingPlan(
        actual={"tokens": 300, "atoms": 2000},
        storage={"tokens": 320, "atoms": 2016},
        target={"tokens": 512, "atoms": 3072},
    )

    assert plan.changed is True
    assert plan.summary()["target"] == {"tokens": 512, "atoms": 3072}
    assert "tokens 300 (stored 320) -> 512" in plan.message("model")


def test_masked_random_draw_preserves_compact_matrix_stream() -> None:
    import jax
    import jax.numpy as jnp

    key = jax.random.key(23)
    exact = jax.random.normal(key, (2, 3, 3, 4))
    token_mask = jnp.asarray([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=bool)
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    padded = masked_prefix_draw(
        lambda draw_key, shape: jax.random.normal(draw_key, shape),
        key,
        pair_mask,
        trailing_shape=(4,),
    )

    np.testing.assert_array_equal(np.asarray(padded[:, :3, :3]), np.asarray(exact))
    assert not np.any(np.asarray(padded)[:, 3:])
    assert not np.any(np.asarray(padded)[:, :, 3:])


def test_result_summary_only_adds_concrete_shape_profile_when_present() -> None:
    plain = PredictionResult(model="boltz2")
    padded = PredictionResult(
        model="boltz2",
        shape_profile={
            "actual": {"tokens": 300},
            "target": {"tokens": 512},
            "changed": True,
        },
    )

    assert "shape_profile" not in plain.summary()
    assert padded.summary()["shape_profile"] == padded.shape_profile


def test_all_builtin_models_advertise_their_complete_padding_profile() -> None:
    expected = {
        "alphafold3": ("tokens",),
        "boltz2": ("tokens", "atoms", "msa"),
        "esmfold2": ("tokens", "atoms", "msa", "language_model_tokens"),
        "opendde": ("tokens", "atoms", "msa", "structural_tokens"),
        "openfold3": ("tokens", "atoms", "msa", "templates"),
        "protenix": (
            "tokens",
            "atoms",
            "msa",
            "templates",
            "language_model_tokens",
        ),
    }

    assert {model: capabilities(model).padding_axes for model in expected} == expected


def test_cli_padding_shortcut_and_exact_targets(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "predict",
            "--model",
            "boltz2",
            "--input",
            str(_input(tmp_path)),
            "--pad-tokens",
            "512",
            "--pad-atoms",
            "4096",
            "--pad-msa",
            "128",
            "--padding-overflow",
            "exact",
        ]
    )

    request = _request(args)

    assert request.padding == PaddingConfig(
        tokens=512,
        atoms=4096,
        msa=128,
        overflow="exact",
    )

    automatic_args = _parser().parse_args(
        [
            "cache",
            "warm",
            "--model",
            "boltz2",
            "--input",
            str(_input(tmp_path)),
            "--padding",
        ]
    )
    assert _request(automatic_args).padding == PaddingConfig()


def test_cli_rejects_overflow_policy_without_padding(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "plan",
            "--model",
            "boltz2",
            "--input",
            str(_input(tmp_path)),
            "--padding-overflow",
            "exact",
        ]
    )
    with pytest.raises(ValueError, match="requires --padding"):
        _request(args)


class _NoPaddingBackend(Backend):
    name = "third-party"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model=self.name, input_formats=("foldjax",))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        raise AssertionError("not called")


class _TokenPaddingBackend(_NoPaddingBackend):
    padding_axes = ("tokens",)


def test_backend_padding_support_is_fail_closed(tmp_path: Path) -> None:
    path = _input(tmp_path)
    request = PredictionRequest(
        model="third-party",
        input=path,
        input_format="foldjax",
        padding=True,
    )
    with pytest.raises(ValueError, match="does not support input padding"):
        _NoPaddingBackend().validate_request(request)

    request = PredictionRequest(
        model="third-party",
        input=path,
        input_format="foldjax",
        padding=PaddingConfig(atoms=512),
    )
    with pytest.raises(ValueError, match="explicit padding axes: atoms"):
        _TokenPaddingBackend().validate_request(request)


def test_host_padding_policy_does_not_fragment_cache_namespace(tmp_path: Path) -> None:
    path = _input(tmp_path)
    backend = _TokenPaddingBackend()
    exact = PredictionRequest(model="third-party", input=path)
    automatic = PredictionRequest(model="third-party", input=path, padding=True)
    pinned = PredictionRequest(
        model="third-party", input=path, padding=PaddingConfig(tokens=512)
    )

    assert backend.cache_profile(exact) == backend.cache_profile(automatic)
    assert backend.cache_profile(exact) == backend.cache_profile(pinned)
