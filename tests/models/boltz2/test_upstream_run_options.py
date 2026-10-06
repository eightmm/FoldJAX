"""Upstream `boltz predict` options the port now takes: where each one lands.

`--step_scale`, `--subsample_msa`/`--num_subsampled_msa`, `--method` and
`--use_potentials` (`boltz/main.py:876-888,968-1031`). Each must reach the
call that consumes it, keep upstream's value when omitted, and fork the
compilation-cache namespace when it departs from that value.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.api as native_api
from foldjax.api import resolve_cache_dir
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models.boltz2.data.featurize import _cache_opts
from foldjax.schema import PaddingConfig, PredictionRequest
from tests.test_boltz2_session import _features


def _request(tmp_path: Path, **options) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("{}\n")
    weights = tmp_path / "boltz2_conf.npz"
    weights.write_bytes(b"weights")
    mols = tmp_path / "mols"
    mols.mkdir(exist_ok=True)
    return PredictionRequest(
        model="boltz2",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / "out",
        seed=0,
        cache_dir=tmp_path / "cache",
        options={"mols": mols, "write_fmt": None, **options},
    )


def _native_run(monkeypatch, tmp_path: Path, **options) -> dict[str, dict]:
    """Run the native `predict` with the model and featurizer replaced."""

    seen: dict[str, dict] = {}

    def featurize(**kwargs):
        seen["featurize"] = kwargs
        return _features(), "job", tmp_path

    def boltz2_predict(params, feats, key, **kwargs):
        seen["model"] = kwargs
        value = params["trunk"]["weight"][0]
        return {
            "sample_atom_coords": jnp.full((1, 3, 3), value),
            "plddt": jnp.ones((1, 2)),
            "iptm": jnp.asarray([value]),
        }

    monkeypatch.setattr(native_api, "featurize", featurize)
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params",
        lambda path: {"trunk": {"weight": jnp.asarray([1.0])}},
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", boltz2_predict
    )
    weights = tmp_path / "w.npz"
    weights.write_bytes(b"w")
    native_api.predict(
        seq=["AA"],
        weights=weights,
        mols=tmp_path,
        out_dir=tmp_path,
        seed=0,
        write_fmt=None,
        **options,
    )
    return seen


def test_omitted_options_run_upstream_values(tmp_path: Path, monkeypatch) -> None:
    seen = _native_run(monkeypatch, tmp_path)

    assert seen["model"]["step_scale"] == 1.5, "boltz/main.py:1227"
    assert seen["model"]["subsample_msa"] is False, "click is_flag"
    assert seen["model"]["num_subsampled_msa"] == 1024
    assert seen["model"]["steering_args"] is None
    assert seen["featurize"]["method"] is None


def test_each_option_reaches_the_call_that_consumes_it(
    tmp_path: Path, monkeypatch
) -> None:
    seen = _native_run(
        monkeypatch,
        tmp_path,
        step_scale=2,
        subsample_msa=True,
        num_subsampled_msa=256,
        method="SOLUTION NMR",
    )

    assert seen["model"]["step_scale"] == 2.0
    assert type(seen["model"]["step_scale"]) is float
    assert seen["model"]["subsample_msa"] is True
    assert seen["model"]["num_subsampled_msa"] == 256
    assert seen["featurize"]["method"] == "solution nmr"


def test_use_potentials_is_upstream_steering_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    seen = _native_run(monkeypatch, tmp_path, use_potentials=True)

    # `boltz/main.py:1309-1311`: two switches flipped, contact guidance kept.
    assert seen["model"]["steering_args"] == {
        "fk_steering": True,
        "num_particles": 3,
        "fk_lambda": 4.0,
        "fk_resampling_interval": 3,
        "physical_guidance_update": True,
        "contact_guidance_update": True,
        "num_gd_steps": 20,
    }


def test_method_reaches_the_featurizer_and_its_cache_key() -> None:
    released = _cache_opts(False, "u", "greedy", None, "released", 0)
    assert _cache_opts(False, "u", "greedy", None, "released", 0, None) == released
    assert _cache_opts(False, "u", "greedy", None, "released", 0, "md") != released


def test_featurize_yaml_hands_the_method_to_the_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    """`override_method` is what `featurizerv2.py` reads into `method_feature`."""

    from foldjax.models.boltz2.data import featurize as module

    seen: dict = {}

    class Dataset:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __getitem__(self, index):
            return {}

    monkeypatch.setattr(module, "check_inputs", lambda path: [path])
    monkeypatch.setattr(
        module,
        "process_inputs",
        lambda **kwargs: SimpleNamespace(records=[SimpleNamespace(id="job")]),
    )
    monkeypatch.setattr(module, "PredictionDataset", Dataset)
    yaml_path = tmp_path / "job.yaml"
    yaml_path.write_text("{}\n")

    module.featurize_yaml(yaml_path, tmp_path, tmp_path, method="md")

    assert seen["override_method"] == "md"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"step_scale": 0}, "step_scale must be a positive number"),
        ({"step_scale": True}, "step_scale must be a positive number"),
        ({"subsample_msa": "yes"}, "subsample_msa must be a boolean"),
        ({"num_subsampled_msa": 0}, "num_subsampled_msa must be a positive"),
        ({"method": "cryo"}, "method 'cryo' is not supported"),
        ({"use_potentials": 1}, "use_potentials must be a boolean"),
        (
            {"use_potentials": True, "steering_args": {"fk_steering": False}},
            "pass it or steering_args, not both",
        ),
        (
            {"use_potentials": True, "deterministic": "on"},
            "builds no executable",
        ),
        ({"use_potentials": True, "cp_devices": 2}, "cannot partition"),
        ({"subsample_msa": True, "cp_devices": 4}, "drop subsample_msa or"),
    ],
)
def test_plan_refuses_what_the_run_would(
    tmp_path: Path, options: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Boltz2Backend().validate_request(_request(tmp_path, **options))


@pytest.mark.parametrize("name", ["subsample_msa", "use_potentials"])
def test_padding_refuses_the_options_it_cannot_carry(tmp_path: Path, name: str) -> None:
    request = dataclasses.replace(
        _request(tmp_path, **{name: True}), padding=PaddingConfig()
    )
    with pytest.raises(ValueError, match=f"{name} .*drop it or --padding"):
        Boltz2Backend().validate_request(request)


def test_adapter_forwards_the_options_to_the_native_call(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict = {}

    def native_predict(**kwargs):
        seen.update(kwargs)
        return {"coords": np.zeros((1, 3, 3)), "plddt": np.ones((1, 2))}

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    Boltz2Backend().predict(
        _request(
            tmp_path,
            step_scale=1.2,
            subsample_msa=True,
            num_subsampled_msa=512,
            method="md",
            use_potentials=True,
        )
    )

    assert seen["step_scale"] == 1.2
    assert seen["subsample_msa"] is True
    assert seen["num_subsampled_msa"] == 512
    assert seen["method"] == "md"
    assert seen["use_potentials"] is True


def test_released_spellings_share_a_namespace_and_departures_fork(
    tmp_path: Path,
) -> None:
    backend = Boltz2Backend()
    base = _request(tmp_path)
    omitted = resolve_cache_dir(base, backend)

    def scope(**options):
        return resolve_cache_dir(
            dataclasses.replace(base, options={**base.options, **options}),
            backend,
        )

    assert (
        scope(
            step_scale=1.5,
            subsample_msa=False,
            num_subsampled_msa=1024,
            use_potentials=False,
        )
        == omitted
    )
    assert scope(step_scale=2) == scope(step_scale=2.0)
    assert scope(method="MD") == scope(method="md")
    for departure in (
        {"step_scale": 2.0},
        {"subsample_msa": True},
        {"num_subsampled_msa": 512},
        {"method": "md"},
        {"use_potentials": True},
    ):
        assert scope(**departure) != omitted, departure
