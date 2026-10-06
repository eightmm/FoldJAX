"""Upstream ESMFold2's inference-time augmentations, as options.

`forward(lm_mask_pct=None, msa_column_mask_rate=0.1,
msa_subsample_at_inference=True)` (`modeling_esmfold2.py:877-884`). Each must
reach what consumes it -- ESMC's input ids, the structure graph's settings --
keep upstream's value when omitted, and fork the compilation-cache namespace
when it departs from it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.esmfold2 import _FIXED_COMPILE_DEFAULTS, ESMFold2Backend
from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.data.features import build_features, pad_features
from foldjax.models.esmfold2.models import esmc
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.schema import PredictionRequest


def _request(tmp_path: Path, **options) -> PredictionRequest:
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
        input_format="foldjax",
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=options,
    )


def _adapter_overrides(tmp_path: Path, monkeypatch, **options) -> dict:
    seen: dict = {}

    def predict_job(key, chains, alignments, model, **overrides):
        seen.update(overrides)
        return object(), object()

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *args, **kwargs: SimpleNamespace(has_language_model=True),
            seed_key=lambda seed: seed,
            predict_job=predict_job,
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *args, **kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 91.0}],
            }
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    ESMFold2Backend().predict(_request(tmp_path, **options))
    return seen


def test_the_fixed_defaults_are_upstreams() -> None:
    settings = structure_model.ModelSettings()
    assert _FIXED_COMPILE_DEFAULTS["msa_column_mask_rate"] == (
        settings.msa_column_mask_rate
    )
    assert _FIXED_COMPILE_DEFAULTS["lm_mask_pct"] == inference.LM_MASK_PCT == 0.0
    assert _FIXED_COMPILE_DEFAULTS["full_depth_msa"] is False
    assert settings.max_msa_depth is not None, "the subsample is on by default"


def test_omitted_options_reach_the_port_in_its_usual_call_form(
    tmp_path: Path, monkeypatch
) -> None:
    overrides = _adapter_overrides(tmp_path, monkeypatch)
    for name in ("lm_mask_pct", "msa_column_mask_rate", "full_depth_msa"):
        assert name not in overrides


def test_the_adapter_forwards_each_option(tmp_path: Path, monkeypatch) -> None:
    overrides = _adapter_overrides(
        tmp_path,
        monkeypatch,
        lm_mask_pct=0.15,
        msa_column_mask_rate=0,
        full_depth_msa=True,
    )
    assert overrides["lm_mask_pct"] == 0.15
    assert overrides["msa_column_mask_rate"] == 0.0
    assert overrides["full_depth_msa"] is True


def test_the_overrides_reach_the_settings() -> None:
    base = structure_model.ModelSettings()
    full = structure_model.with_overrides(
        base, full_depth_msa=True, msa_column_mask_rate=0.0
    )
    assert full.max_msa_depth is None, "None is the model's no-subsample path"
    assert full.msa_column_mask_rate == 0.0
    assert structure_model.with_overrides(base) == base
    with pytest.raises(ValueError, match="pass one of the two"):
        structure_model.with_overrides(base, full_depth_msa=True, max_msa_depth=64)


def test_the_inference_keywords_reach_the_compiled_identity(monkeypatch) -> None:
    """Accepted and dropped would answer one program out of another's entry."""

    features = pad_features(
        build_features([("AG", "A", 0, 0)]), n_token=8, n_atom=64, n_msa=4
    )
    settings = structure_model.ModelSettings()
    model = SimpleNamespace(
        settings=settings,
        parameters={"token_bonds.weight": jnp.ones((settings.d_pair, 1), jnp.bfloat16)},
        esmc_parameters=None,
        esmc_settings=None,
    )
    seen: list = []

    def fake_compiled_predict(*identity, **_):
        seen.append(identity[0])
        return lambda *args, **kwargs: {}

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)
    inference.predict(
        jax.random.key(0),
        features,
        model,
        msa_column_mask_rate=0.25,
        full_depth_msa=True,
    )

    assert seen[0].msa_column_mask_rate == 0.25
    assert seen[0].max_msa_depth is None


def test_predict_masks_the_lm_input_off_its_own_key(monkeypatch) -> None:
    features = pad_features(
        build_features([("AG", "A", 0, 0)]), n_token=8, n_atom=64, n_msa=4
    )
    settings = structure_model.ModelSettings()
    model = SimpleNamespace(
        settings=settings,
        parameters={"token_bonds.weight": jnp.ones((settings.d_pair, 1), jnp.bfloat16)},
        esmc_parameters={},
        esmc_settings=object(),
    )
    seen: dict = {}

    def language_model_states(features, model, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(inference, "language_model_states", language_model_states)
    monkeypatch.setattr(
        inference, "compiled_predict", lambda *a, **k: lambda *a, **k: {}
    )
    key = jax.random.key(3)
    inference.predict(key, features, model, lm_mask_pct=0.2)

    assert seen["lm_mask_pct"] == 0.2
    np.testing.assert_array_equal(
        jax.random.key_data(seen["mask_key"]),
        jax.random.key_data(inference.lm_mask_key(key)),
    )


def _encoded_ids(monkeypatch, **kwargs) -> np.ndarray:
    seen: dict = {}

    def encode(lm_input_ids, sequence_id, params, prefix, **_):
        seen["ids"] = np.asarray(lm_input_ids)
        return jnp.zeros((1, *lm_input_ids.shape, 1))

    monkeypatch.setattr(esmc, "encode", encode)
    n = 40
    esmc.lm_hidden_states(
        np.full((1, n), 5),
        np.zeros((1, n), dtype=np.int64),
        np.arange(n)[None],
        np.zeros((1, n), dtype=np.int64),
        np.ones((1, n), dtype=bool),
        {},
        settings=None,
        **kwargs,
    )
    return seen["ids"]


def test_lm_masking_replaces_residues_and_never_special_tokens(monkeypatch) -> None:
    plain = _encoded_ids(monkeypatch)
    assert not np.any(plain == esmc.MASK_TOKEN_ID)

    key = jax.random.key(0)
    masked = _encoded_ids(monkeypatch, lm_mask_pct=0.5, mask_key=key)
    special = np.isin(plain, (esmc.BOS_TOKEN_ID, esmc.EOS_TOKEN_ID, esmc.PAD_TOKEN_ID))
    assert np.array_equal(masked[special], plain[special])
    replaced = masked != plain
    assert replaced.any() and not replaced.all()
    assert np.all(masked[replaced] == esmc.MASK_TOKEN_ID)

    # A padded LM axis draws the same mask over the real prefix.
    padded = _encoded_ids(monkeypatch, lm_mask_pct=0.5, mask_key=key, packed_length=64)
    assert np.array_equal(padded[:, : masked.shape[1]], masked)
    assert np.all(padded[:, masked.shape[1] :] == esmc.PAD_TOKEN_ID)


def test_a_masked_embedding_is_not_served_to_another_seed(tmp_path: Path) -> None:
    """The compact embedding is retained per input; masking makes it per seed."""

    backend = ESMFold2Backend()
    model = SimpleNamespace(has_language_model=True)
    backend._loaded_model = model
    backend._session_active = True
    calls: list = []

    def language_model_embedding(features, model, **kwargs):
        calls.append(kwargs)
        return len(calls)

    fake = SimpleNamespace(
        LANGUAGE_MODEL_FEATURES=("input_ids",),
        language_model_embedding=language_model_embedding,
    )
    features = {"input_ids": np.arange(4)}

    def embedding(seed):
        return backend._language_model_embedding(
            fake,
            features,
            model,
            packed_length=None,
            masking={
                "lm_mask_pct": 0.1,
                "mask_key": inference.lm_mask_key(jax.random.key(seed)),
            },
        )

    assert embedding(0) == 1
    assert embedding(0) == 1, "the same seed's draw is reusable"
    assert embedding(1) == 2, "another seed's draw is not"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"lm_mask_pct": 1.0}, r"lm_mask_pct must lie in \[0, 1\)"),
        ({"lm_mask_pct": True}, "lm_mask_pct must be a number"),
        ({"msa_column_mask_rate": 1.5}, r"msa_column_mask_rate must lie in \[0, 1\]"),
        ({"full_depth_msa": "yes"}, "full_depth_msa must be a boolean"),
        (
            {"full_depth_msa": True, "max_msa_depth": 256},
            "pass one of the two",
        ),
        (
            {"lm_mask_pct": 0.1, "no_language_model": True},
            "runs none",
        ),
    ],
)
def test_plan_refuses_what_the_run_cannot_honour(
    tmp_path: Path, options: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ESMFold2Backend().validate_request(_request(tmp_path, **options))


def test_released_spellings_share_a_namespace_and_departures_fork(
    tmp_path: Path,
) -> None:
    backend = ESMFold2Backend()
    base = _request(tmp_path)
    omitted = resolve_cache_dir(base, backend)

    def scope(**options):
        return resolve_cache_dir(dataclasses.replace(base, options=options), backend)

    assert scope(lm_mask_pct=0, msa_column_mask_rate=0.1, full_depth_msa=False) == (
        omitted
    )
    assert scope(msa_column_mask_rate=0) == scope(msa_column_mask_rate=0.0)
    for departure in (
        {"lm_mask_pct": 0.15},
        {"msa_column_mask_rate": 0.0},
        {"full_depth_msa": True},
    ):
        assert scope(**departure) != omitted, departure
