from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.api as api
from foldjax.backends.boltz2 import _padding_shape_profile
from foldjax.models.boltz2.data.ownership import (
    COMPACT_TOKEN_TO_REP_ATOM,
    TOKEN_TO_REP_ATOM_INDEX,
)
from foldjax.schema import PaddingConfig


def _fake_features(*, affinity: bool) -> dict[str, np.ndarray]:
    return {
        "atom_pad_mask": np.ones((1, 3), dtype=np.float32),
        "token_pad_mask": np.ones((1, 2), dtype=np.float32),
        "mol_type": np.asarray([[0, 3]], dtype=np.int32),
        "token_to_rep_atom": np.asarray(
            [[[1, 0, 0], [0, 0, 1]]], dtype=np.int64
        ),
        "affinity_token_mask": np.asarray(
            [[0.0, 1.0 if affinity else 0.0]], dtype=np.float32
        ),
    }


def test_featurize_forwards_all_msa_server_controls(tmp_path, monkeypatch) -> None:
    seen = {}

    def fake_featurize_yaml(*args, **kwargs):
        seen.update(kwargs)
        structures = tmp_path / "structures"
        structures.mkdir()
        (structures / "job.npz").touch()
        return _fake_features(affinity=False), None, structures

    monkeypatch.setattr(api, "featurize_yaml", fake_featurize_yaml)

    api.featurize(
        input=tmp_path / "job.yaml",
        mols=tmp_path,
        out_dir=tmp_path / "out",
        use_msa_server=True,
        msa_server_url="https://msa.example.test",
        msa_pairing_strategy="complete",
        msa_server_username="user",
        msa_server_password="password",
        msa_api_key_header=None,
        msa_api_key_value=None,
    )

    assert seen["use_msa_server"] is True
    assert seen["msa_server_url"] == "https://msa.example.test"
    assert seen["msa_pairing_strategy"] == "complete"
    assert seen["msa_server_username"] == "user"
    assert seen["msa_server_password"] == "password"


def test_featurize_reads_msa_api_key_from_environment(tmp_path, monkeypatch) -> None:
    seen = {}

    def fake_featurize_yaml(*args, **kwargs):
        seen.update(kwargs)
        structures = tmp_path / "structures"
        structures.mkdir()
        (structures / "job.npz").touch()
        return _fake_features(affinity=False), None, structures

    monkeypatch.setattr(api, "featurize_yaml", fake_featurize_yaml)
    monkeypatch.setenv("MSA_API_KEY_VALUE", "env-key")

    api.featurize(
        input=tmp_path / "job.yaml",
        mols=tmp_path,
        out_dir=tmp_path / "out",
        use_msa_server=True,
    )

    assert seen["msa_server_username"] is None
    assert seen["msa_server_password"] is None
    assert seen["msa_api_key_value"] == "env-key"


def test_api_passes_native_affinity_head_to_model(tmp_path, monkeypatch) -> None:
    confidence_weights = tmp_path / "boltz2_conf"
    affinity_weights = tmp_path / "boltz2_aff"
    affinity_weights.with_suffix(".npz").touch()
    loaded = []
    seen = []

    primary_features = _fake_features(affinity=True)
    primary_features["disto_target"] = np.full(
        (1, 2, 2, 1, 64), np.nan, dtype=np.float32
    )
    affinity_features = _fake_features(affinity=True)
    affinity_features["disto_target"] = np.full(
        (1, 2, 2, 1, 64), np.inf, dtype=np.float32
    )
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (primary_features, "job", tmp_path),
    )

    def fake_load_params(path: Path):
        loaded.append(Path(path))
        if Path(path) == affinity_weights:
            return {
                "trunk": {"stage": jnp.asarray([2.0])},
                "affinity": {"modules": [{"head": jnp.asarray([1.0])}]},
            }
        return {"trunk": {}}

    def fake_predict(params, feats, key, **kwargs):
        controls = {
            key: kwargs[key]
            for key in (
                "steering_args",
                "multiplicity",
                "diffusion_chunk_size",
                "attention_backend",
                "triangle_backend",
                "glu_backend",
            )
        }
        controls["noise_tape"] = "init_noise" in kwargs or "step_noises" in kwargs
        second_stage = "affinity" in params
        seen.append((second_stage, controls, frozenset(feats)))
        samples = kwargs["multiplicity"]
        if second_stage:
            return {
                "sample_atom_coords": jnp.zeros((samples, 3, 3)),
                "plddt": jnp.ones((samples, 3)),
                "affinity_pred_value": jnp.asarray([2.0]),
            }
        return {
            "sample_atom_coords": jnp.zeros((samples, 3, 3)),
            "plddt": jnp.ones((samples, 3)),
            "iptm": jnp.asarray([0.1, 0.9]),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", fake_load_params
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )
    monkeypatch.setattr(
        api,
        "_prepare_affinity_features",
        lambda **kwargs: affinity_features,
        raising=False,
    )

    out = api.predict(
        seq=["ACD"],
        weights=confidence_weights,
        affinity_weights=affinity_weights,
        mols=tmp_path,
        out_dir=tmp_path,
        steering_args={"fk_steering": False},
        num_samples=2,
        affinity_num_samples=6,
        attention_backend="xla",
        triangle_backend="xla",
        glu_backend="xla",
    )

    assert loaded == [confidence_weights, affinity_weights]
    assert seen[0][:2] == (
        False,
        {
            "steering_args": {"fk_steering": False},
            "multiplicity": 2,
            "diffusion_chunk_size": None,
            "attention_backend": "xla",
            "triangle_backend": "xla",
            "glu_backend": "xla",
            "noise_tape": False,
        },
    )
    assert seen[1][0] is True
    assert seen[1][1]["multiplicity"] == 6
    assert seen[1][1]["diffusion_chunk_size"] == 5
    assert seen[1][1]["noise_tape"] is False
    assert "disto_target" not in seen[0][2]
    assert "disto_target" not in seen[1][2]
    assert "token_to_rep_atom" not in seen[0][2]
    assert "token_to_rep_atom" not in seen[1][2]
    assert COMPACT_TOKEN_TO_REP_ATOM in seen[0][2]
    assert COMPACT_TOKEN_TO_REP_ATOM in seen[1][2]
    assert TOKEN_TO_REP_ATOM_INDEX in seen[0][2]
    assert TOKEN_TO_REP_ATOM_INDEX in seen[1][2]
    np.testing.assert_array_equal(out["raw"]["affinity_pred_value"], [2.0])
    assert out["coords"].shape == (2, 3, 3)
    assert out["plddt"].shape == (2, 3)


def test_api_rejects_affinity_job_without_native_affinity_weights(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (_fake_features(affinity=True), "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )

    with pytest.raises(FileNotFoundError, match="affinity weights"):
        api.predict(
            seq=["ACD"],
            weights=tmp_path / "boltz2_conf",
            mols=tmp_path,
            out_dir=tmp_path,
        )


def test_bucket_api_crops_public_and_raw_outputs(tmp_path, monkeypatch) -> None:
    feats = _fake_features(affinity=False)
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (feats, "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )

    def fake_predict(params, model_feats, key, **kwargs):
        atoms = model_feats["atom_pad_mask"].shape[-1]
        tokens = model_feats["token_pad_mask"].shape[-1]
        samples = kwargs["multiplicity"]
        return {
            "sample_atom_coords": jnp.zeros((samples, atoms, 3)),
            "plddt": jnp.ones((samples, tokens)),
            "plddt_logits": jnp.ones((samples, tokens, 50)),
            "pae": jnp.ones((samples, tokens, tokens)),
            "pdistogram": jnp.ones((1, tokens, tokens, 1, 64)),
            "iptm": jnp.ones((samples,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )

    out = api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        bucket=True,
        num_samples=2,
    )

    assert out["coords"].shape == (2, 3, 3)
    assert out["plddt"].shape == (2, 2)
    assert out["raw"]["sample_atom_coords"].shape == (2, 3, 3)
    assert out["raw"]["plddt_logits"].shape == (2, 2, 50)
    assert out["raw"]["pae"].shape == (2, 2, 2)
    assert out["raw"]["pdistogram"].shape == (1, 2, 2, 1, 64)
    assert out["padding"]["primary"]["target"] == {
        "tokens": 256,
        "atoms": 256,
        "msa": 1,
    }


@pytest.mark.parametrize("bucket", [False, True])
def test_nonsteering_exact_and_bucket_routes_filter_dead_features(
    tmp_path, monkeypatch, bucket
) -> None:
    feats = _fake_features(affinity=False)
    feats["disto_target"] = np.full(
        (1, 2, 2, 1, 64), np.nan, dtype=np.float32
    )
    feats["writer_only"] = np.full((1, 9), np.inf, dtype=np.float32)
    monkeypatch.setattr(api, "featurize", lambda **kwargs: (feats, "job", tmp_path))
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )
    seen = []

    def fake_predict(params, model_feats, key, **kwargs):
        seen.append(frozenset(model_feats))
        atoms = model_feats["atom_pad_mask"].shape[-1]
        tokens = model_feats["token_pad_mask"].shape[-1]
        return {
            "sample_atom_coords": jnp.zeros((1, atoms, 3)),
            "plddt": jnp.ones((1, tokens)),
            "iptm": jnp.ones((1,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )

    api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        bucket=bucket,
        num_steps=1,
    )

    assert len(seen) == 1
    assert "disto_target" not in seen[0]
    assert "writer_only" not in seen[0]


def test_api_routes_compact_msa_storage_to_the_model(tmp_path, monkeypatch) -> None:
    feats = _fake_features(affinity=False)
    shape = (1, 3, 2)
    feats.update(
        msa=(np.arange(np.prod(shape), dtype=np.int64) % 32).reshape(shape),
        has_deletion=(np.arange(np.prod(shape)).reshape(shape) % 2).astype(bool),
        deletion_value=np.linspace(
            0.0, 1.0, np.prod(shape), dtype=np.float32
        ).reshape(shape),
        msa_paired=np.ones(shape, dtype=np.float32),
        msa_mask=np.ones(shape, dtype=np.int64),
    )
    monkeypatch.setattr(api, "featurize", lambda **kwargs: (feats, "job", tmp_path))
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )
    captured = {}

    def fake_predict(_params, model_features, _key, **_kwargs):
        captured.update(
            {
                name: model_features[name].dtype
                for name in (
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "msa_paired",
                    "msa_mask",
                )
            }
        )
        return {
            "sample_atom_coords": jnp.zeros((1, 3, 3)),
            "plddt": jnp.ones((1, 2)),
            "iptm": jnp.ones((1,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )

    api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        num_steps=1,
    )

    assert captured == {
        "msa": np.dtype(np.uint8),
        "has_deletion": np.dtype(np.bool_),
        "deletion_value": np.dtype(np.float32),
        "msa_paired": np.dtype(np.bool_),
        "msa_mask": np.dtype(np.bool_),
    }


def test_api_keeps_public_features_dense_while_model_input_is_compact(
    tmp_path, monkeypatch
) -> None:
    public_features = _fake_features(affinity=False)
    captured = {}
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (public_features, "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )

    def fake_predict(_params, model_features, _key, **_kwargs):
        captured["keys"] = frozenset(model_features)
        captured["marker_dtype"] = model_features[COMPACT_TOKEN_TO_REP_ATOM].dtype
        captured["marker_shape"] = model_features[COMPACT_TOKEN_TO_REP_ATOM].shape
        captured["dtype"] = model_features[TOKEN_TO_REP_ATOM_INDEX].dtype
        captured["shape"] = model_features[TOKEN_TO_REP_ATOM_INDEX].shape
        return {
            "sample_atom_coords": jnp.zeros((1, 3, 3)),
            "plddt": model_features[TOKEN_TO_REP_ATOM_INDEX].astype(jnp.float32),
            "iptm": jnp.ones((1,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )

    api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        num_steps=1,
    )

    assert "token_to_rep_atom" in public_features
    assert COMPACT_TOKEN_TO_REP_ATOM not in public_features
    assert TOKEN_TO_REP_ATOM_INDEX not in public_features
    assert "token_to_rep_atom" not in captured["keys"]
    assert COMPACT_TOKEN_TO_REP_ATOM in captured["keys"]
    assert TOKEN_TO_REP_ATOM_INDEX in captured["keys"]
    assert captured["marker_dtype"] == jnp.uint8
    assert captured["marker_shape"] == ()
    assert captured["dtype"] == jnp.int32
    assert captured["shape"] == (1, 2)


@pytest.mark.parametrize(
    "private",
    [
        {COMPACT_TOKEN_TO_REP_ATOM: np.asarray(1, dtype=np.uint8)},
        {TOKEN_TO_REP_ATOM_INDEX: np.asarray([[0, 2]], dtype=np.int32)},
    ],
)
def test_managed_api_rejects_incomplete_private_representation(
    tmp_path, monkeypatch, private
) -> None:
    features = _fake_features(affinity=False)
    del features["token_to_rep_atom"]
    features.update(private)
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (features, "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )

    with pytest.raises(ValueError, match="requires both"):
        api.predict(
            seq=["AC"],
            weights=tmp_path / "boltz2_conf",
            mols=tmp_path,
            out_dir=tmp_path,
            num_steps=1,
        )


def test_active_steering_retains_variable_and_host_features(
    tmp_path, monkeypatch
) -> None:
    feats = _fake_features(affinity=False)
    feats["disto_target"] = np.zeros((1, 2, 2, 1, 64), dtype=np.float32)
    feats["steering_only"] = np.arange(7, dtype=np.int32)
    monkeypatch.setattr(api, "featurize", lambda **kwargs: (feats, "job", tmp_path))
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )
    seen = []

    def fake_predict(params, model_feats, key, **kwargs):
        seen.append(frozenset(model_feats))
        return {
            "sample_atom_coords": jnp.zeros((1, 3, 3)),
            "plddt": jnp.ones((1, 2)),
            "iptm": jnp.ones((1,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )

    api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        steering_args={"fk_steering": True},
        num_steps=1,
    )

    assert seen == [frozenset(feats)]


def test_fk_bucket_tape_expands_the_particle_axis(tmp_path, monkeypatch) -> None:
    feats = _fake_features(affinity=False)
    monkeypatch.setattr(api, "featurize", lambda **kwargs: (feats, "job", tmp_path))
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )
    monkeypatch.setattr(api, "_prefix_rng_is_supported", lambda: False)
    tape_calls = []

    def fake_tape(key, *, multiplicity, storage_atoms, target_atoms, steps):
        del key
        tape_calls.append((multiplicity, storage_atoms, target_atoms, steps))
        return (
            jnp.zeros((multiplicity, target_atoms, 3), dtype=jnp.float32),
            jnp.zeros((steps, multiplicity, target_atoms, 3), dtype=jnp.float32),
        )

    monkeypatch.setattr(api, "_prefix_stable_noise_tape", fake_tape)
    seen = []

    def fake_predict(params, model_feats, key, **kwargs):
        del params, key
        seen.append((kwargs["diffusion_chunk_size"], kwargs["init_noise"].shape))
        return {
            "sample_atom_coords": jnp.zeros(
                (7, model_feats["atom_pad_mask"].shape[-1], 3)
            ),
            "plddt": jnp.ones((7, model_feats["token_pad_mask"].shape[-1])),
            "iptm": jnp.ones((7,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )
    api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        steering_args={"fk_steering": True, "num_particles": 3},
        num_samples=7,
        num_steps=2,
        bucket=True,
    )

    assert tape_calls == [(21, 3, 256, 2)]
    assert seen == [(5, (21, 256, 3))]


def test_neutral_padding_crops_back_to_the_default_public_shapes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (_fake_features(affinity=False), "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )

    def fake_predict(params, model_feats, key, **kwargs):
        atoms = model_feats["atom_pad_mask"].shape[-1]
        tokens = model_feats["token_pad_mask"].shape[-1]
        return {
            "sample_atom_coords": jnp.zeros((1, atoms, 3)),
            "plddt": jnp.ones((1, tokens)),
            "pae": jnp.ones((1, tokens, tokens)),
            "iptm": jnp.ones((1,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )

    out = api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        padding=PaddingConfig(tokens=8, atoms=32, msa=1),
    )

    assert out["coords"].shape == (3, 3)
    assert out["plddt"].shape == (2,)
    assert out["raw"]["sample_atom_coords"].shape == (1, 3, 3)
    assert out["raw"]["pae"].shape == (1, 2, 2)
    assert out["padding"]["primary"] == {
        "actual": {"tokens": 2, "atoms": 3, "msa": 1},
        "storage": {"tokens": 2, "atoms": 3, "msa": 1},
        "target": {"tokens": 8, "atoms": 32, "msa": 1},
        "changed": True,
        "static": {
            "use_template": False,
            "recompute_nonpolymer_frames": True,
        },
    }


def test_neutral_padding_noise_tape_preserves_the_real_unpadded_prefix() -> None:
    import jax

    key = jax.random.PRNGKey(17)
    init, noises = api._prefix_stable_noise_tape(
        key,
        multiplicity=2,
        storage_atoms=5,
        target_atoms=8,
        steps=4,
    )

    run_key, init_key = jax.random.split(key)
    expected_init = jax.random.normal(init_key, (2, 5, 3), dtype=jnp.float32)
    expected_steps = []
    for _ in range(4):
        run_key, noise_key = jax.random.split(run_key)
        expected_steps.append(
            jax.random.normal(noise_key, (2, 5, 3), dtype=jnp.float32)
        )

    np.testing.assert_array_equal(np.asarray(init[:, :5]), np.asarray(expected_init))
    np.testing.assert_array_equal(
        np.asarray(noises[:, :, :5]), np.asarray(jnp.stack(expected_steps))
    )
    assert not np.any(np.asarray(init[:, 5:]))
    assert not np.any(np.asarray(noises[:, :, 5:]))


@pytest.mark.parametrize("prefix_supported", [False, True])
def test_padding_noise_uses_each_stage_storage_stride(
    tmp_path, monkeypatch, prefix_supported: bool
) -> None:
    confidence_weights = tmp_path / "boltz2_conf"
    affinity_weights = tmp_path / "boltz2_aff"
    affinity_weights.with_suffix(".npz").touch()
    primary = _fake_features(affinity=True)
    primary["atom_pad_mask"] = np.asarray([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    affinity = _fake_features(affinity=True)
    affinity["atom_pad_mask"] = np.asarray(
        [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    monkeypatch.setattr(api, "featurize", lambda **kwargs: (primary, "job", tmp_path))

    def fake_load_params(path):
        if Path(path) == affinity_weights:
            return {"trunk": {}, "affinity": {}}
        return {"trunk": {}}

    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", fake_load_params
    )
    monkeypatch.setattr(api, "_prepare_affinity_features", lambda **kwargs: affinity)
    monkeypatch.setattr(
        api, "_prefix_rng_is_supported", lambda: prefix_supported
    )
    tape_calls = []
    prefix_calls = []

    def fake_tape(key, *, multiplicity, storage_atoms, target_atoms, steps):
        tape_calls.append((multiplicity, storage_atoms, target_atoms, steps))
        return (
            jnp.zeros((multiplicity, target_atoms, 3), dtype=jnp.float32),
            jnp.zeros((steps, multiplicity, target_atoms, 3), dtype=jnp.float32),
        )

    monkeypatch.setattr(api, "_prefix_stable_noise_tape", fake_tape)

    def fake_prefix_size(*, storage_atoms, target_atoms):
        prefix_calls.append((storage_atoms, target_atoms))
        return np.asarray(storage_atoms, dtype=np.int32)

    monkeypatch.setattr(api, "_storage_prefix_size", fake_prefix_size)

    routed_noise = []

    def fake_predict(params, feats, key, **kwargs):
        routed_noise.append(
            (
                "noise_storage_atoms" in kwargs,
                "init_noise" in kwargs,
                "step_noises" in kwargs,
            )
        )
        samples = kwargs["multiplicity"]
        atoms = feats["atom_pad_mask"].shape[-1]
        tokens = feats["token_pad_mask"].shape[-1]
        output = {
            "sample_atom_coords": jnp.zeros((samples, atoms, 3)),
            "plddt": jnp.ones((samples, tokens)),
        }
        if "affinity" in params:
            output["affinity_pred_value"] = jnp.asarray([1.0])
        else:
            output["iptm"] = jnp.arange(samples, dtype=jnp.float32)
        return output

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )

    out = api.predict(
        seq=["AC"],
        weights=confidence_weights,
        affinity_weights=affinity_weights,
        mols=tmp_path,
        out_dir=tmp_path,
        num_samples=2,
        affinity_num_samples=3,
        num_steps=2,
        affinity_num_steps=2,
        padding=PaddingConfig(tokens=8, atoms=32, msa=1),
    )

    if prefix_supported:
        assert tape_calls == []
        assert prefix_calls == [(5, 32), (7, 32)]
        assert routed_noise == [(True, False, False), (True, False, False)]
    else:
        assert tape_calls == [(2, 5, 32, 2), (3, 7, 32, 2)]
        assert prefix_calls == []
        assert routed_noise == [(False, True, True), (False, True, True)]
    assert out["coords"].shape == (2, 3, 3)
    assert out["padding"]["primary"]["static"] == {
        "use_template": False,
        "recompute_nonpolymer_frames": True,
    }
    assert out["padding"]["affinity"]["static"] == {
        "use_template": False,
        "recompute_nonpolymer_frames": True,
    }


def test_explicit_padding_overflow_fails_before_primary_inference(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (_fake_features(affinity=False), "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )
    predicted = []
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict",
        lambda *args, **kwargs: predicted.append(True),
    )

    with pytest.raises(ValueError, match="tokens.*smaller"):
        api.predict(
            seq=["AC"],
            weights=tmp_path / "boltz2_conf",
            mols=tmp_path,
            out_dir=tmp_path,
            num_steps=2,
            padding=PaddingConfig(tokens=1, atoms=32, msa=1),
        )

    assert predicted == []


def test_affinity_feature_recovery_threads_the_primary_msa_cap(
    tmp_path, monkeypatch
) -> None:
    processed_structures = tmp_path / "processed" / "structures"
    processed_structures.mkdir(parents=True)
    regenerated_structures = tmp_path / "regenerated" / "processed" / "structures"
    regenerated_structures.mkdir(parents=True)
    manifest = SimpleNamespace(records=(SimpleNamespace(id="job"),))
    seen = {}

    def fake_featurize_yaml(*args, **kwargs):
        seen["regeneration_cap"] = kwargs["max_msa_depth"]
        return {}, manifest, regenerated_structures

    def fake_affinity(**kwargs):
        seen["affinity_cap"] = kwargs["max_msa_depth"]
        return {}

    monkeypatch.setattr(
        "foldjax.models.boltz2.data.featurize.featurize_yaml",
        fake_featurize_yaml,
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.data.featurize.featurize_affinity_from_prediction",
        fake_affinity,
    )

    api._prepare_affinity_features(
        input_path=tmp_path / "job.yaml",
        struct_dir=processed_structures,
        record_id="job",
        mol_dir=tmp_path,
        predicted_coords=np.zeros((1, 3, 3)),
        atom_pad_mask=np.ones((1, 3)),
        out_dir=tmp_path / "affinity",
        max_msa_depth=17,
    )

    assert seen == {"regeneration_cap": 17, "affinity_cap": 17}


def test_affinity_dataset_receives_the_requested_msa_cap(tmp_path, monkeypatch) -> None:
    import foldjax.models.boltz2.data.featurize as module

    atom_dtype = np.dtype([("coords", np.float32, (3,))])

    @dataclass(frozen=True)
    class FakeStructure:
        atoms: np.ndarray
        coords: np.ndarray

        def dump(self, path):
            Path(path).touch()

    atoms = np.zeros(2, dtype=atom_dtype)
    structure = FakeStructure(atoms=atoms, coords=atoms.copy())
    monkeypatch.setattr(module.StructureV2, "load", lambda path: structure)
    seen = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __getitem__(self, index):
            return {}

    monkeypatch.setattr(module, "PredictionDataset", FakeDataset)
    manifest = SimpleNamespace(records=(SimpleNamespace(id="job"),))
    processed = tmp_path / "processed"
    (processed / "structures").mkdir(parents=True)

    result = module.featurize_affinity_from_prediction(
        manifest=manifest,
        processed_dir=processed,
        mol_dir=tmp_path,
        predicted_coords=np.zeros((1, 2, 3), dtype=np.float32),
        atom_pad_mask=np.ones((1, 2), dtype=np.float32),
        out_dir=tmp_path / "affinity",
        max_msa_depth=19,
    )

    assert result == {}
    assert seen["max_msa_seqs"] == 19


def test_native_api_rejects_neutral_and_legacy_padding_together(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        api.predict(
            seq=["AC"],
            weights=tmp_path / "boltz2_conf",
            mols=tmp_path,
            bucket=True,
            padding=PaddingConfig(),
        )


def test_backend_shape_profile_reports_both_compiled_affinity_stages() -> None:
    primary = {
        "target": {"tokens": 256, "atoms": 512, "msa": 64},
        "static": {"use_template": False, "recompute_nonpolymer_frames": False},
    }
    affinity = {
        "target": {"tokens": 256, "atoms": 512, "msa": 64},
        "static": {"use_template": True, "recompute_nonpolymer_frames": True},
    }

    assert primary["target"] == affinity["target"]
    assert primary != affinity
    assert _padding_shape_profile({"primary": primary, "affinity": affinity}) == {
        "primary": primary,
        "affinity": affinity,
    }
    assert _padding_shape_profile({"primary": primary}) == primary


def test_api_rejects_nonpositive_diffusion_samples(tmp_path) -> None:
    with pytest.raises(ValueError, match="num_samples"):
        api.predict(
            seq=["ACD"],
            weights=tmp_path / "boltz2_conf",
            mols=tmp_path,
            num_samples=0,
        )


def test_stop_after_trunk_crops_and_saves_representations(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (_fake_features(affinity=True), "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params",
        lambda path: {"trunk": {}},
    )

    def fake_predict(params, model_feats, key, **kwargs):
        assert kwargs["stop_after_trunk"] is True
        assert "token_to_rep_atom" not in model_feats
        assert COMPACT_TOKEN_TO_REP_ATOM not in model_feats
        assert TOKEN_TO_REP_ATOM_INDEX not in model_feats
        tokens = model_feats["token_pad_mask"].shape[-1]
        return {
            "single": jnp.ones((1, tokens, 4), dtype=jnp.float32),
            "pair": jnp.ones((1, tokens, tokens, 2), dtype=jnp.float32),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict",
        fake_predict,
    )
    destination = tmp_path / "representations"
    result = api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        stop_after="trunk",
        representations=("single", "pair"),
        representations_dir=destination,
        padding=PaddingConfig(tokens=8, atoms=32, msa=1),
        write_fmt=None,
    )

    assert "coords" not in result
    assert "plddt" not in result
    assert result["raw"]["single"].shape == (1, 2, 4)
    assert result["raw"]["pair"].shape == (1, 2, 2, 2)
    assert result["representations"] == destination / "representations.npz"
    assert result["representations"].is_file()
    with np.load(result["representations"]) as archive:
        # The archive carries the axes its spec names -- `("token", "channel")`
        # and `("token", "token", "channel")` -- so the batch of one the raw
        # result still holds is not written. Keeping it made the manifest say
        # four numbers in `shape` beside three names in `axes`.
        assert archive["single"].shape == (2, 4)
        assert archive["pair"].shape == (2, 2, 2)
    assert result["padding"]["primary"]["target"] == {
        "tokens": 8,
        "atoms": 32,
        "msa": 1,
    }
