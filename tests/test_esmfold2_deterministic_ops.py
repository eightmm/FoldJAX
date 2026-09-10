"""The ESMFold2 adapter's half of the deterministic knob.

The port's own owners are covered in
``tests/models/esmfold2/test_deterministic_ops.py``. What is left here is the
adapter: that the neutral knob is declared in the shape this port is driven
in, that it reaches the port's signature rather than stopping in the options
dict, that it selects its own compile namespace, and that the one piece of
state this session carries between predictions -- the compact ESMC embedding
-- cannot be handed to a run whose policy it was not built under.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.esmfold2 import _FIXED_COMPILE_DEFAULTS, ESMFold2Backend
from foldjax.execution import DETERMINISTIC_API_OPTION
from foldjax.schema import PredictionRequest


def _request(tmp_path: Path, **options: Any) -> PredictionRequest:
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "name": "t",
                "entities": [{"type": "protein", "id": ["A"], "sequence": "ACDEF"}],
            }
        )
    )
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model="esmfold2",
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        input_format="foldjax",
        options=options,
    )


def test_the_adapter_declares_the_shared_bool_shaped_entry() -> None:
    """This port is called through a Python signature, so `off` is `False`.

    Object identity rather than equality: a literal copy here would be a
    second chance to spell the vocabulary differently.
    """
    table = ESMFold2Backend.execution_options

    assert table["deterministic"] is DETERMINISTIC_API_OPTION["deterministic"]
    assert "deterministic" in ESMFold2Backend.compile_options
    assert _FIXED_COMPILE_DEFAULTS["deterministic"] is False


def test_the_option_reaches_the_ports_own_signature(tmp_path, monkeypatch) -> None:
    """Declared and translated is not enough; it has to leave the adapter."""
    seen: dict[str, Any] = {}

    def capture(*_args: Any, **kwargs: Any):
        seen.update(kwargs)
        return ({}, {})

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *_args, **_kwargs: SimpleNamespace(has_language_model=True),
            seed_key=lambda seed: seed,
            predict_job=capture,
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *_args, **_kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 91.0}],
            }
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )

    ESMFold2Backend().predict(_request(tmp_path, deterministic="on"))

    assert seen["deterministic"] is True


def test_an_unasked_run_passes_no_policy_at_all(tmp_path, monkeypatch) -> None:
    """Off is the model's own default; spelling it would be a value nobody asked
    for in the recorded overrides, exactly as `structure_sample_sequential` is."""
    seen: dict[str, Any] = {}

    def capture(*_args: Any, **kwargs: Any):
        seen.update(kwargs)
        return ({}, {})

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *_args, **_kwargs: SimpleNamespace(has_language_model=True),
            seed_key=lambda seed: seed,
            predict_job=capture,
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *_args, **_kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 91.0}],
            }
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )

    ESMFold2Backend().predict(_request(tmp_path))

    assert "deterministic" not in seen


def test_off_shares_the_omitted_namespace_and_on_gets_its_own(tmp_path) -> None:
    """A repeatable run is a different executable, so it is a different cache."""
    backend = ESMFold2Backend()
    omitted = _request(tmp_path)
    explicit_off = dataclasses.replace(omitted, options={"deterministic": "off"})
    asked = dataclasses.replace(omitted, options={"deterministic": "on"})

    assert backend.cache_profile(explicit_off) == backend.cache_profile(omitted)
    assert backend.cache_profile(asked) == {"num_recycles": 9, "deterministic": True}
    assert resolve_cache_dir(asked, backend) != resolve_cache_dir(omitted, backend)


class _FakeInference:
    """Just enough of the port for the adapter's derived-state cache."""

    LANGUAGE_MODEL_FEATURES = ("input_ids",)

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def language_model_embedding(
        self,
        _features: Any,
        _model: Any,
        *,
        packed_length: int | None,
        deterministic: bool = False,
    ) -> object:
        self.calls.append(deterministic)
        return object()


def _retaining_backend(model: Any) -> ESMFold2Backend:
    backend = ESMFold2Backend()
    backend._loaded_model = model
    backend._session_active = True
    return backend


def test_a_retained_embedding_never_serves_a_run_with_another_policy() -> None:
    """The compact ESMC value is the one thing that survives between seeds.

    It is built by executables the policy selects, so serving the one built
    without the option to a run that asked for it would make `deterministic=on`
    a claim about a value produced without it -- the exact failure the second
    owner exists to prevent.
    """
    model = SimpleNamespace(has_language_model=True)
    backend = _retaining_backend(model)
    fake = _FakeInference()
    features = {"input_ids": np.asarray([[1, 2, 3]], np.int32)}

    plain = backend._language_model_embedding(
        fake, features, model, packed_length=None, deterministic=False
    )
    again = backend._language_model_embedding(
        fake, features, model, packed_length=None, deterministic=False
    )
    repeatable = backend._language_model_embedding(
        fake, features, model, packed_length=None, deterministic=True
    )

    assert again is plain
    assert repeatable is not plain
    assert fake.calls == [False, True]


def test_the_policy_travels_with_the_states_helper() -> None:
    """The legacy split path recomputes the stack; it must recompute it under
    the same policy the structure graph is about to be built with."""
    model = SimpleNamespace(has_language_model=True)
    seen: list[bool] = []

    fake = SimpleNamespace(
        language_model_states=lambda *_args, **kwargs: seen.append(
            kwargs["deterministic"]
        )
    )
    ESMFold2Backend()._language_model_states(
        fake, {}, model, packed_length=None, deterministic=True
    )

    assert seen == [True]


def test_a_value_outside_the_vocabulary_is_refused(tmp_path) -> None:
    """A misspelled policy must not look like an ordinary run."""
    with pytest.raises(ValueError, match="deterministic must be one of"):
        ESMFold2Backend().apply_sampling(_request(tmp_path, deterministic="true"))
