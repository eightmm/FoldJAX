"""`return_expected_errors` on ESMFold2 and OpenFold3: a switch, default on.

Both ports return the expected PAE/PDE matrices by default, and the common
writers turn them into `confidence_full.npz` arrays and the per-sample
`predicted_aligned_error.json`. The option lets a caller drop them. Three
properties are pinned here:

* an omitted option and an explicit `true` reach the port in the call form a
  released run always used -- no keyword at all -- and select the cache
  namespace every earlier run wrote, so the default program is unchanged;
* `false` reaches the port, joins the compile identity, and is refused when
  it is not a switch;
* the availability table names the option as the way to remove them.

What the compiled graphs return with the setting off is pinned at the model
layer: `tests/models/esmfold2/test_confidence_logits_flag.py` and
`tests/models/openfold3/test_full_confidence_outputs.py`. What the writers do
with a prediction that lacks them -- an `unavailable` reason in
`confidence_full.npz` and no `predicted_aligned_error.json` -- is pinned in
`tests/test_confidence_arrays.py` and `tests/test_viewer_exports.py`.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends import openfold3 as openfold3_backend
from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.models.esmfold2 import inference as esmfold2_inference
from foldjax.models.esmfold2.models import model as esmfold2_model
from foldjax.schema import PredictionRequest
from tests.test_esmfold2_backend import _stub_prediction

OPTION = "return_expected_errors"


def _request(tmp_path: Path, model: str, **fields: Any) -> PredictionRequest:
    job = tmp_path / "job.json"
    job.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model=model,
        input=job,
        input_format="foldjax",
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        **fields,
    )


@pytest.mark.parametrize(
    "backend", [ESMFold2Backend, OpenFold3Backend], ids=["esmfold2", "openfold3"]
)
def test_the_option_is_a_declared_compile_relevant_switch(backend) -> None:
    assert OPTION in backend.native_options
    assert OPTION in backend.boolean_options
    assert OPTION in backend.compile_options


@pytest.mark.parametrize(
    "backend", [ESMFold2Backend, OpenFold3Backend], ids=["esmfold2", "openfold3"]
)
def test_on_shares_the_namespace_an_omitted_option_selects(
    tmp_path: Path, backend
) -> None:
    """The released program keeps its namespace; only `false` forks one."""
    model = backend.name
    omitted = _request(tmp_path, model)
    profile = backend().cache_profile(omitted)
    assert OPTION not in profile
    namespace = resolve_cache_dir(omitted, backend())
    for spelling in (True, "true", "on", 1):
        spelled = dataclasses.replace(omitted, options={OPTION: spelling})
        assert backend().cache_profile(spelled) == profile, spelling
        assert resolve_cache_dir(spelled, backend()) == namespace, spelling
    for spelling in (False, "false", "off", 0):
        off = dataclasses.replace(omitted, options={OPTION: spelling})
        assert backend().cache_profile(off)[OPTION] is False, spelling
        assert resolve_cache_dir(off, backend()) != namespace, spelling


def test_esmfold2_omitted_namespace_is_the_one_every_earlier_run_wrote(
    tmp_path: Path,
) -> None:
    """The profile an omitted ESMFold2 request had before the option existed."""
    assert ESMFold2Backend().cache_profile(_request(tmp_path, "esmfold2")) == {
        "num_recycles": 3
    }


def test_openfold3_trunk_only_graph_ignores_the_option(tmp_path: Path) -> None:
    """No confidence head runs, so `false` is the same program: one namespace."""
    trunk = _request(
        tmp_path, "openfold3", stop_after="trunk", representations=("pair",)
    )
    off = dataclasses.replace(trunk, options={OPTION: False})
    backend = OpenFold3Backend()
    assert backend.cache_profile(off) == backend.cache_profile(trunk)
    assert resolve_cache_dir(off, backend) == resolve_cache_dir(trunk, backend)


@pytest.mark.parametrize(
    "backend", [ESMFold2Backend, OpenFold3Backend], ids=["esmfold2", "openfold3"]
)
def test_a_value_that_is_not_a_switch_is_refused_at_plan_time(
    tmp_path: Path, backend
) -> None:
    request = _request(tmp_path, backend.name, options={OPTION: "maybe"})
    with pytest.raises(ValueError, match=f"{OPTION} must be a boolean"):
        backend().validate_request(request)


# --- ESMFold2: adapter -> `inference.predict` -> `ModelSettings` ----------------


@pytest.mark.parametrize("options", [{}, {OPTION: True}, {OPTION: "true"}])
def test_esmfold2_on_reaches_the_port_as_no_keyword(
    tmp_path, monkeypatch, options
) -> None:
    """Omission and `true` are the call form the released program came from."""
    seen, result = _stub_prediction(tmp_path, monkeypatch, options)
    assert OPTION not in seen
    assert OPTION not in result.raw["overrides"]


@pytest.mark.parametrize("spelling", [False, "false", "off"])
def test_esmfold2_off_reaches_the_port(tmp_path, monkeypatch, spelling) -> None:
    seen, result = _stub_prediction(tmp_path, monkeypatch, {OPTION: spelling})
    assert seen[OPTION] is False
    assert result.raw["overrides"][OPTION] is False


def test_esmfold2_port_applies_the_override_to_its_settings() -> None:
    parameter = inspect.signature(esmfold2_inference.predict).parameters[OPTION]
    assert parameter.default is None
    settings = esmfold2_model.ModelSettings()
    assert settings.return_expected_errors is True
    # `None` is "leave the settings alone": the unasked call is the same
    # static settings object value, so the same compiled program.
    assert esmfold2_model.with_overrides(settings) == settings
    assert (
        esmfold2_model.with_overrides(settings, return_expected_errors=None)
        == settings
    )
    off = esmfold2_model.with_overrides(settings, return_expected_errors=False)
    assert off.return_expected_errors is False
    assert off != settings


# --- OpenFold3: adapter -> `released_config` -----------------------------------


class _ConfigReachedError(Exception):
    """Raised by the `released_config` double once the adapter has called it."""


def _released_config_kwargs(
    tmp_path: Path, monkeypatch, **fields: Any
) -> dict[str, Any]:
    """Drive `OpenFold3Backend.predict` up to `released_config`; return its kwargs."""
    seen: dict[str, Any] = {}
    features = {
        "token_mask": np.ones((1, 4), dtype=np.float32),
        "atom_mask": np.ones((1, 4), dtype=np.float32),
    }

    def released_config(**kwargs: Any):
        seen.update(kwargs)
        raise _ConfigReachedError

    modules = {
        "foldjax.models.openfold3.data": SimpleNamespace(
            pocket_sampling_config=lambda batch: None,
            has_atomized_tokens=lambda batch: False,
        ),
        "foldjax.models.openfold3.inference": SimpleNamespace(
            released_config=released_config,
        ),
        "foldjax.models.openfold3.output": SimpleNamespace(),
        "foldjax.models.openfold3.bridge.chemistry": SimpleNamespace(),
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(),
        "jax": SimpleNamespace(),
    }
    monkeypatch.setattr(openfold3_backend, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        openfold3_backend,
        "_features_chemistry_and_metadata",
        lambda *args, **kwargs: (features, None, {}),
    )
    with pytest.raises(_ConfigReachedError):
        OpenFold3Backend().predict(_request(tmp_path, "openfold3", **fields))
    return seen


@pytest.mark.parametrize("options", [{}, {OPTION: True}, {OPTION: "on"}])
def test_openfold3_on_leaves_released_config_its_own_default(
    tmp_path, monkeypatch, options
) -> None:
    from foldjax.models.openfold3 import inference

    seen = _released_config_kwargs(tmp_path, monkeypatch, options=options)
    assert OPTION not in seen
    # The config is the compiled program's static argument, and an unasked
    # call builds the one every released run did.
    assert inference.released_config(n_token=4, n_atom=4).return_expected_errors


@pytest.mark.parametrize("spelling", [False, "false", "off"])
def test_openfold3_off_reaches_released_config(
    tmp_path, monkeypatch, spelling
) -> None:
    seen = _released_config_kwargs(tmp_path, monkeypatch, options={OPTION: spelling})
    assert seen[OPTION] is False


def test_openfold3_trunk_only_run_keeps_the_released_call_form(
    tmp_path, monkeypatch
) -> None:
    seen = _released_config_kwargs(
        tmp_path,
        monkeypatch,
        stop_after="trunk",
        representations=("pair",),
        options={OPTION: False},
    )
    assert OPTION not in seen


# --- Availability ---------------------------------------------------------------


def test_the_availability_table_names_the_option_that_removes_them() -> None:
    from foldjax import confidence_arrays

    for model in ("esmfold2", "openfold3"):
        entry = confidence_arrays.AVAILABILITY[model]
        removed = entry["opt_out"][f"{OPTION}=false"]
        assert set(removed) == {"pae", "pde"}
        assert set(removed) <= set(entry["default"])
