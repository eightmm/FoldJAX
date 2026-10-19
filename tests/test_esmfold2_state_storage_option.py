"""Buffered LM storage reaches every managed route without changing graph ABI."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.models.esmfold2 import inference
from foldjax.schema import PredictionRequest
from tests.test_esmfold2_backend import (
    _fake_session_modules,
    _fake_session_weights,
    _fake_staged_session_modules,
    _job,
)


def _request(tmp_path):
    return PredictionRequest(
        model="esmfold2",
        input=_job(
            tmp_path / "input",
            [{"type": "protein", "id": ["A"], "sequence": "ACD"}],
        ),
        weights=_fake_session_weights(tmp_path),
        output_dir=tmp_path / "out",
        seed=7,
    )


@pytest.mark.parametrize("value", [None, False, 3, "stream", ["buffer"]])
def test_invalid_storage_is_rejected_without_loading_weights(value):
    with pytest.raises(ValueError, match="lm_state_storage"):
        ESMFold2Backend().validate_native_options({"lm_state_storage": value})


def test_storage_preserves_default_namespace_options(tmp_path):
    backend = ESMFold2Backend()
    request = _request(tmp_path)
    expected = backend.cache_profile(request)
    for value in ("stack", "buffer"):
        changed = dataclasses.replace(request, options={"lm_state_storage": value})
        assert backend.apply_sampling(changed)["lm_state_storage"] == value
        assert backend.cache_profile(changed) == expected


@pytest.mark.parametrize("compact", [False, True])
def test_session_dispatches_requested_storage_and_keeps_embedding_ownership(
    tmp_path, monkeypatch, compact
):
    calls = {"load": [], "lm": [], "predict": []}
    _, modules = _fake_session_modules(tmp_path, calls)
    native = modules["foldjax.models.esmfold2.inference"]
    native.LANGUAGE_MODEL_STATE_STORAGE_API = True
    native.COMPACT_LANGUAGE_MODEL_API = compact
    function = "language_model_embedding" if compact else "language_model_states"
    original = getattr(native, function)
    choices = []

    def storage(features, loaded, *, state_storage="stack", **kwargs):
        choices.append(state_storage)
        return original(features, loaded, **kwargs)

    setattr(native, function, storage)
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    request = _request(tmp_path)
    changed = dataclasses.replace(request, options={"lm_state_storage": "buffer"})
    backend = ESMFold2Backend()
    with backend.session((request, changed, changed)):
        results = [backend.predict(item) for item in (request, changed, changed)]
    assert choices == (
        ["stack", "buffer"] if compact else ["stack", "buffer", "buffer"]
    )
    assert len(calls["load"]) == 1
    assert "lm_state_storage" not in results[0].raw["overrides"]
    assert results[1].raw["overrides"]["lm_state_storage"] == "buffer"
    np.testing.assert_array_equal(calls["predict"][0][1], calls["predict"][1][1])


def test_staged_session_restages_after_a_storage_change(tmp_path, monkeypatch):
    events, choices = [], []
    modules = _fake_staged_session_modules(tmp_path, events)
    native = modules["foldjax.models.esmfold2.inference"]
    native.LANGUAGE_MODEL_STATE_STORAGE_API = True
    original = native.language_model_embedding

    def embedding(*args, state_storage="stack", **kwargs):
        choices.append(state_storage)
        return original(*args, **kwargs)

    native.language_model_embedding = embedding
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    request = _request(tmp_path)
    backend = ESMFold2Backend()
    with backend.session((request,)):
        backend.predict(request)
        backend.predict(
            dataclasses.replace(request, options={"lm_state_storage": "buffer"})
        )
    assert choices == ["stack", "buffer"]
    assert [name for name, _ in events].count("stage") == 2


def test_unsupported_wrapper_fails_before_weight_load(tmp_path, monkeypatch):
    calls = {"load": [], "lm": [], "predict": []}
    _, modules = _fake_session_modules(tmp_path, calls)
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    with pytest.raises(NotImplementedError, match="buffered language-model"):
        ESMFold2Backend().predict(
            dataclasses.replace(
                _request(tmp_path), options={"lm_state_storage": "buffer"}
            )
        )
    assert not calls["load"] and not calls["predict"]


@pytest.mark.parametrize("storage", ["stack", "buffer"])
def test_native_states_and_compact_projection_forward_storage(monkeypatch, storage):
    from types import SimpleNamespace

    calls = []

    def states(*args, **kwargs):
        calls.append(kwargs)
        return np.ones((1, 2, 3, 4), np.float32)

    monkeypatch.setattr(inference.esmc_model, "lm_hidden_states", states)
    monkeypatch.setattr(
        inference, "_language_model_embedding_from_states", lambda x, *a, **kw: x
    )
    features = {
        name: np.ones((1, 2), np.int32) for name in inference.LANGUAGE_MODEL_FEATURES
    }
    model = SimpleNamespace(esmc_parameters={}, esmc_settings=object())
    inference.language_model_states(features, model, state_storage=storage)
    inference.language_model_embedding(features, model, state_storage=storage)
    for call in calls:
        assert call.get("state_storage", "stack") == storage
        assert ("state_storage" in call) == (storage != "stack")
