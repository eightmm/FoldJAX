"""The managed compact-LM route must refuse before running ESMC."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from foldjax import memory_policy
from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.models.esmfold2 import inference
from foldjax.schema import PaddingConfig, PredictionRequest
from tests.test_esmfold2_backend import (
    _fake_session_weights,
    _fake_staged_session_modules,
    _job,
)


@pytest.fixture
def managed(tmp_path, monkeypatch):
    memory_policy.clear_record()
    memory_policy.reset_warnings()
    events = []
    modules = _fake_staged_session_modules(tmp_path, events)
    native = modules["foldjax.models.esmfold2.inference"]
    native.LANGUAGE_MODEL_STATE_STORAGE_API = True
    original_stage = native.load_language_model_stage

    def stage(*args, **kwargs):
        loaded = original_stage(*args, **kwargs)
        loaded.settings.num_samples = 32
        loaded.settings.diffusion = SimpleNamespace(
            glu_backend="xla", rigid_align_backend="svd"
        )
        return loaded

    native.load_language_model_stage = stage
    native.language_model_length = lambda features: features["input_ids"].shape[-1] + 2
    native.normalize_msa_features = lambda _key, features, **kwargs: features
    native.pad_features = lambda features, *, n_token, **kwargs: {
        key: np.pad(value, ((0, 0), (0, n_token - value.shape[-1])))
        for key, value in features.items()
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    monkeypatch.setattr(
        "foldjax.backends.esmfold2._padding_plan",
        lambda *args, **kwargs: SimpleNamespace(
            target={"tokens": 1792, "atoms": 16384, "msa": 16}
        ),
    )
    monkeypatch.setattr(
        memory_policy,
        "device_memory_budget",
        lambda **kwargs: memory_policy.resolve_budget(
            pool_bytes=int(40.3 * 2**30),
            card_bytes=int(47.4 * 2**30),
            override_gib=None,
        ),
    )
    backend = ESMFold2Backend()
    weights = _fake_session_weights(tmp_path)

    def request(options=None):
        return PredictionRequest(
            model="esmfold2",
            input=_job(
                tmp_path / "input",
                [{"type": "protein", "id": ["A"], "sequence": "A" * 1726}],
            ),
            weights=weights,
            output_dir=tmp_path / "out",
            padding=PaddingConfig(),
            options=options or {},
        )

    yield backend, request, events
    memory_policy.clear_record()
    memory_policy.reset_warnings()


def test_managed_1726_padded_to_1792_refuses_before_embedding_or_structure_weights(
    managed, monkeypatch
):
    backend, make_request, events = managed
    request = make_request()
    monkeypatch.setattr(
        backend,
        "_language_model_embedding",
        lambda *args, **kwargs: pytest.fail("refused request ran the LM"),
    )
    monkeypatch.setattr(
        backend,
        "_materialize_structure_model",
        lambda *args, **kwargs: pytest.fail("refused request loaded structure weights"),
    )
    with backend.session((request,)):
        with pytest.raises(MemoryError, match="observed ESMFold2 capacity refusal"):
            backend.predict(request)
    assert [name for name, _value in events] == ["stage"]
    record = memory_policy.recorded()
    assert record["n_token"] == 1792
    assert record["calibration_id"] == memory_policy.ESMFOLD2_CAPACITY_POLICY_ID


@pytest.mark.parametrize(
    "options",
    [
        {"memory_check": "warn"},
        {"lm_state_storage": "buffer"},
        {"rigid_align_backend": "jacobi"},
        {"glu_backend": "tokamax"},
        {"cp_devices": 2},
        {"num_samples": 1},
    ],
)
def test_managed_warn_and_uncalibrated_alternatives_reach_the_lm(
    managed, monkeypatch, options
):
    class _ReachedLMError(Exception):
        pass

    backend, make_request, _events = managed
    request = make_request(options)

    def reached_lm(*args, **kwargs):
        raise _ReachedLMError

    monkeypatch.setattr(backend, "_language_model_embedding", reached_lm)
    with backend.session((request,)):
        with pytest.raises(_ReachedLMError):
            backend.predict(request)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"cp_shards": 2},
        {"stop_after_inputs": True},
        {"lm_state_storage": "buffer"},
        {"glu_backend": "tokamax"},
        {"rigid_align_backend": "jacobi"},
        {"num_samples": 1},
        {"cp_shards": 2, "lm_state_storage": "buffer", "rigid_align_backend": "jacobi"},
    ],
)
def test_managed_guard_matches_native_off_profile_criteria(monkeypatch, options):
    class _CapturedError(Exception):
        pass

    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        raise _CapturedError

    monkeypatch.setattr(memory_policy, "admit", capture)
    model = inference.LoadedModel(
        parameters={}, settings=inference.structure_model.ModelSettings()
    )
    with pytest.raises(_CapturedError):
        inference.predict(
            None,
            {"token_attention_mask": np.ones((1, 1792), dtype=bool)},
            model,
            memory_budget=memory_policy.resolve_budget(
                pool_bytes=int(40.3 * 2**30),
                card_bytes=int(47.4 * 2**30),
                override_gib=None,
            ),
            **options,
        )
    assert seen["off_profile"] == memory_policy.esmfold2_off_profile_reason(
        num_samples=options.get("num_samples", model.settings.num_samples),
        cp_shards=options.get("cp_shards", 1),
        stop_after_inputs=options.get("stop_after_inputs", False),
        lm_state_storage=options.get("lm_state_storage", "stack"),
        glu_backend=options.get("glu_backend", model.settings.diffusion.glu_backend),
        rigid_align_backend=options.get(
            "rigid_align_backend", model.settings.diffusion.rigid_align_backend
        ),
    )
