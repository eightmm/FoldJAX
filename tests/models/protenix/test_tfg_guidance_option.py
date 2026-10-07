"""Upstream Protenix's `--use_tfg_guidance`, as a switch.

`runner/batch_inference.py:697-702` and `:410` set
`sample_diffusion.guidance.enable` on upstream's default mapping
(`configs/configs_base.py:185-247`). The switch must hand that mapping, and
only that one, to the sampler, run the unrolled sampler it needs, and be
refused wherever the guided path cannot run.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.protenix import ProtenixBackend
from foldjax.models.protenix.bridge.weights_io import save_native_weights
from foldjax.models.protenix.cli.predict import main
from foldjax.models.protenix.data.static_io import save_static_feature_npz
from foldjax.models.protenix.tfg.config import (
    UPSTREAM_GUIDANCE_CONFIG,
    parse_tfg_config,
    upstream_guidance_config,
)
from foldjax.schema import PaddingConfig, PredictionRequest

from .test_model import _toy_features, _toy_params

#: An upstream Protenix checkout beside this repository, if there is one.
_UPSTREAM_CONFIGS = next(
    (
        parent / "protenix" / "configs"
        for parent in Path(__file__).resolve().parents
        if (parent / "protenix" / "configs" / "configs_base.py").is_file()
    ),
    Path("protenix/configs"),
)


class _StopError(Exception):
    pass


def _toy_argv(tmp_path: Path) -> list[str]:
    weights = tmp_path / "toy_weights.pkl"
    features = tmp_path / "toy_features.npz"
    save_native_weights(weights, _toy_params(), compress=False)
    save_static_feature_npz(features, _toy_features())
    return [
        "--model-name", "unknown",
        "--weights", str(weights),
        "--features", str(features),
        "--out", str(tmp_path / "out.npz"),
        "--n-sample", "1", "--n-step", "1", "--n-cycle", "1",
        "--cpu-only",
    ]  # fmt: skip


def _guidance_reaching_the_sampler(tmp_path: Path, monkeypatch, *flags) -> dict:
    seen: dict = {}

    def predict(*args, **kwargs):
        seen.update(kwargs)
        raise _StopError

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", predict
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.geometry.prepare_tfg_features",
        lambda features, **_kwargs: dict(features),
    )
    with pytest.raises(_StopError):
        main([*_toy_argv(tmp_path), *flags])
    return seen


def test_the_mapping_is_upstreams_with_guidance_on() -> None:
    """Read off upstream's own config file, not restated from memory."""

    if not (_UPSTREAM_CONFIGS / "configs_base.py").is_file():
        pytest.skip("upstream Protenix checkout not present")
    tree = ast.parse((_UPSTREAM_CONFIGS / "configs_base.py").read_text())
    guidance = next(
        value
        for mapping in ast.walk(tree)
        if isinstance(mapping, ast.Dict)
        for key, value in zip(mapping.keys, mapping.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "guidance"
    )
    assert ast.literal_eval(guidance) == UPSTREAM_GUIDANCE_CONFIG
    enabled = upstream_guidance_config()
    assert enabled == {**UPSTREAM_GUIDANCE_CONFIG, "enable": True}
    assert parse_tfg_config(enabled).enable
    assert UPSTREAM_GUIDANCE_CONFIG["enable"] is False, "the copy is not shared"


def test_the_switch_hands_the_sampler_upstreams_mapping(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    seen = _guidance_reaching_the_sampler(tmp_path, monkeypatch, "--use-tfg-guidance")

    assert seen["guidance_config"] == upstream_guidance_config()
    assert seen["use_sampler_scan"] is False, "guidance needs the unrolled sampler"
    assert "TFG enabled" in capsys.readouterr().out


def test_without_the_switch_no_guidance_runs(tmp_path: Path, monkeypatch) -> None:
    seen = _guidance_reaching_the_sampler(tmp_path, monkeypatch)

    assert seen["guidance_config"] is None
    assert seen["guidance_features"] is None


def test_the_switch_and_a_guidance_file_are_refused_together(tmp_path: Path) -> None:
    config = tmp_path / "guidance.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="pass one of them"):
        main(
            [
                *_toy_argv(tmp_path),
                "--use-tfg-guidance",
                "--guidance-config",
                str(config),
            ]
        )


def _request(tmp_path: Path, **options) -> PredictionRequest:
    job = tmp_path / "job.json"
    job.write_text("[]")
    (tmp_path / "protenix.jax").touch()
    return PredictionRequest(
        model="protenix",
        input=job,
        input_format="native",
        weights=tmp_path / "protenix.jax",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        seed=101,
        options=options,
    )


def test_the_adapter_renders_the_switch(tmp_path: Path) -> None:
    invocation = ProtenixBackend()._native_invocation(
        _request(tmp_path, use_tfg_guidance=True)
    )
    assert "--use-tfg-guidance" in invocation.argv
    assert invocation.config_fields["use_tfg_guidance"] is True
    omitted = ProtenixBackend()._native_invocation(_request(tmp_path))
    assert "--use-tfg-guidance" not in omitted.argv
    assert omitted.config_fields["use_tfg_guidance"] is False


@pytest.mark.parametrize(
    ("options", "padding", "message"),
    [
        ({"use_tfg_guidance": "maybe"}, None, "use_tfg_guidance must be a boolean"),
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
        ProtenixBackend().validate_request(request)


@pytest.mark.parametrize("spelling", ["false", "off", "no", "0", 0, False])
def test_an_off_switch_is_not_guidance_under_padding(
    tmp_path: Path, spelling
) -> None:
    # Every switch spelling is one `bool` (`boolean_options`); the padding
    # check read the raw value, where the string "false" is truthy.
    request = dataclasses.replace(
        _request(tmp_path, use_tfg_guidance=spelling), padding=PaddingConfig()
    )
    ProtenixBackend().validate_request(request)


def test_the_switch_forks_the_namespace_and_false_does_not(tmp_path: Path) -> None:
    backend = ProtenixBackend()
    omitted = resolve_cache_dir(_request(tmp_path), backend)

    assert resolve_cache_dir(_request(tmp_path, use_tfg_guidance=False), backend) == (
        omitted
    )
    assert resolve_cache_dir(_request(tmp_path, use_tfg_guidance=True), backend) != (
        omitted
    )
