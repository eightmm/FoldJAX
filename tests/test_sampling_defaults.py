"""`capabilities().sampling_defaults` is what each backend's run path takes.

The reported value is compared, per model, with the value the run itself would
be handed for the same request -- the adapter's translation plus the native
default the runner falls back to -- rather than with a copy of the number.
Before this, AlphaFold 3 reported 10 recycles where its adapter runs 3,
OpenFold3 reported its native 4 trunk passes as 4 neutral recycles, and
Protenix, ESMFold2 and Boltz-2 reported nulls for values they do run at.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from foldjax import paths
from foldjax.registry import capabilities, get_backend
from foldjax.schema import PredictionRequest


def _request(tmp_path: Path, model: str, weights: Path | None = None, **fields):
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    if weights is None:
        weights = tmp_path / f"{model}.jax"
        weights.touch()
    return PredictionRequest(
        model=model,
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=1,
        **fields,
    )


def _values(model: str, request: PredictionRequest) -> dict[str, int | None]:
    resolved = get_backend(model).sampling_resolution(request)
    return {knob: value for knob, (value, _source) in resolved.items()}


def test_alphafold3_reports_the_recycles_its_adapter_runs(tmp_path: Path) -> None:
    from foldjax.backends import alphafold3

    backend = get_backend("alphafold3")
    request = _request(tmp_path, "alphafold3")
    options = backend.apply_sampling(request)
    # `predict` pops each with the table's value as its fallback; the two
    # nested ones reach the vendored config, which a drift test pins.
    ran = {
        knob: options.get(native, alphafold3._RELEASED_COMPILE_DEFAULTS[native])
        for knob, native in backend.sampling_options.items()
    }
    assert ran["num_recycles"] == 3
    assert _values("alphafold3", request) == ran
    assert capabilities("alphafold3").sampling_defaults == ran

    external = _request(tmp_path, "alphafold3", options={"source": str(tmp_path)})
    values = _values("alphafold3", external)
    assert values["num_steps"] is None and values["max_msa_depth"] is None


def test_boltz2_reports_the_api_defaults_and_the_featurizer_depth(
    tmp_path: Path,
) -> None:
    from foldjax.models.boltz2 import api
    from foldjax.models.boltz2.data import const

    signature = inspect.signature(api.predict).parameters
    ran = {
        "num_samples": signature["num_samples"].default,
        "num_steps": signature["num_steps"].default,
        "num_recycles": signature["num_recycles"].default,
    }
    # `max_msa_depth=None` reaches `PredictionDataset`, which caps at this.
    assert signature["max_msa_depth"].default is None
    ran["max_msa_depth"] = const.max_msa_seqs
    request = _request(tmp_path, "boltz2")
    assert _values("boltz2", request) == ran
    assert capabilities("boltz2").sampling_defaults == ran


def test_openfold3_reports_neutral_recycles_not_trunk_passes(tmp_path: Path) -> None:
    from foldjax.models.openfold3 import inference

    signature = inspect.signature(inference.released_config).parameters
    passes = signature["num_recycles"].default
    ran = {
        "num_samples": signature["num_samples"].default,
        "num_steps": signature["num_steps"].default,
        # Upstream runs `num_recycles + 1` passes and stores the total.
        "num_recycles": passes - 1,
        "max_msa_depth": signature["msa_depth"].default,
    }
    assert ran["num_recycles"] == 3
    assert _values("openfold3", _request(tmp_path, "openfold3")) == ran
    assert capabilities("openfold3").sampling_defaults == ran

    # A requested depth above the released one runs at the released one.
    asked = _request(tmp_path, "openfold3", num_recycles=5, max_msa_depth=4096)
    backend = get_backend("openfold3")
    assert backend.apply_sampling(asked)["num_recycles"] == 6
    values = _values("openfold3", asked)
    assert values["num_recycles"] == 5
    assert values["max_msa_depth"] == ran["max_msa_depth"]


@pytest.mark.parametrize(
    ("weights_name", "expected"),
    [
        ("protenix_base_default_v1.0.0.jax", {"num_steps": 200, "num_recycles": 10}),
        ("protenix_mini_esm_v0.5.0.jax", {"num_steps": 5, "num_recycles": 4}),
        # The runner refuses a name it cannot read; nothing is claimed.
        ("renamed.jax", {"num_steps": None, "num_recycles": None}),
    ],
)
def test_protenix_reports_the_model_variants_schedule(
    tmp_path: Path, weights_name: str, expected: dict
) -> None:
    from foldjax.models.protenix.runtime_policy import (
        infer_model_name_from_path,
        model_inference_defaults,
    )

    weights = tmp_path / weights_name
    weights.touch()
    request = _request(tmp_path, "protenix", weights=weights)
    fields = get_backend("protenix")._native_invocation(request).config_fields
    # The parser leaves both unset and the runner reads them off the name.
    assert fields["num_steps"] is None and fields["num_recycles"] is None
    name = infer_model_name_from_path(weights)
    if name is not None:
        schedule = model_inference_defaults(name)
        assert expected == {k: schedule[k] for k in ("num_steps", "num_recycles")}
    values = _values("protenix", request)
    assert values == {
        "num_samples": fields["num_samples"],
        "max_msa_depth": fields["max_msa_depth"],
        **expected,
    }


def test_protenix_capabilities_are_the_released_profiles(tmp_path: Path) -> None:
    weights = tmp_path / "protenix_base_default_v1.0.0.jax"
    weights.touch()
    released = _values("protenix", _request(tmp_path, "protenix", weights=weights))
    assert capabilities("protenix").sampling_defaults == released


def test_opendde_reports_what_its_native_config_is_handed(tmp_path: Path) -> None:
    request = _request(tmp_path, "opendde")
    fields = get_backend("opendde")._native_invocation(request).config_fields
    ran = {
        knob: fields[knob]
        for knob in ("num_samples", "num_steps", "num_recycles", "max_msa_depth")
    }
    assert _values("opendde", request) == ran
    assert capabilities("opendde").sampling_defaults == ran


def test_esmfold2_reads_samples_and_steps_off_the_checkpoint(tmp_path: Path) -> None:
    from foldjax.backends import esmfold2

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {"num_diffusion_samples": 7, "structure_head": {"inference_num_steps": 9}}
        )
    )
    request = _request(tmp_path, "esmfold2", weights=checkpoint)
    options = get_backend("esmfold2").apply_sampling(request)
    resolved = get_backend("esmfold2").sampling_resolution(request)
    assert resolved["num_recycles"] == (options["num_recycles"], "default")
    assert resolved["num_samples"] == (7, "checkpoint")
    assert resolved["num_steps"] == (9, "checkpoint")
    assert resolved["max_msa_depth"][0] == esmfold2.DEFAULTS["max_msa_depth"]

    bare = tmp_path / "bare"
    bare.mkdir()
    unread = _values("esmfold2", _request(tmp_path, "esmfold2", weights=bare))
    assert unread["num_samples"] is None and unread["num_steps"] is None

    assert capabilities("esmfold2").sampling_defaults == {
        **esmfold2._RELEASED_CHECKPOINT_SAMPLING,
        "num_recycles": esmfold2.DEFAULTS["num_recycles"],
        "max_msa_depth": esmfold2.DEFAULTS["max_msa_depth"],
    }


def test_the_released_esmfold2_numbers_are_the_managed_checkpoints() -> None:
    from foldjax.backends import esmfold2

    config = paths.weights_dir() / "esmfold2" / "config.json"
    if not config.is_file():
        pytest.skip("the managed ESMFold2 checkpoint is not in this weight store")
    assert esmfold2._checkpoint_sampling(config.parent) == dict(
        esmfold2._RELEASED_CHECKPOINT_SAMPLING
    )
