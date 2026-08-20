from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.api as api
from foldjax.backends.boltz2 import _padding_shape_profile
from foldjax.schema import PaddingConfig


def _fake_features(*, affinity: bool) -> dict[str, np.ndarray]:
    return {
        "atom_pad_mask": np.ones((1, 3), dtype=np.float32),
        "token_pad_mask": np.ones((1, 2), dtype=np.float32),
        "mol_type": np.asarray([[0, 3]], dtype=np.int32),
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

    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (_fake_features(affinity=True), "job", tmp_path),
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
                "attention_backend",
                "triangle_backend",
                "glu_backend",
            )
        }
        controls["noise_tape"] = "init_noise" in kwargs or "step_noises" in kwargs
        second_stage = "affinity" in params
        seen.append((second_stage, controls))
        if second_stage:
            return {
                "sample_atom_coords": jnp.zeros((5, 3, 3)),
                "plddt": jnp.ones((5, 3)),
                "affinity_pred_value": jnp.asarray([2.0]),
            }
        return {
            "sample_atom_coords": jnp.zeros((2, 3, 3)),
            "plddt": jnp.ones((2, 3)),
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
        lambda **kwargs: _fake_features(affinity=True),
        raising=False,
    )

    out = api.predict(
        seq=["ACD"],
        weights=confidence_weights,
        affinity_weights=affinity_weights,
        mols=tmp_path,
        out_dir=tmp_path,
        steering_args={"fk_steering": False},
        diffusion_samples=2,
        attention_backend="xla",
        triangle_backend="xla",
        glu_backend="xla",
    )

    assert loaded == [confidence_weights, affinity_weights]
    assert seen[0] == (
        False,
        {
            "steering_args": {"fk_steering": False},
            "multiplicity": 2,
            "attention_backend": "xla",
            "triangle_backend": "xla",
            "glu_backend": "xla",
            "noise_tape": False,
        },
    )
    assert seen[1][0] is True
    assert seen[1][1]["multiplicity"] == 5
    assert seen[1][1]["noise_tape"] is False
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
        diffusion_samples=2,
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


def test_padding_noise_tapes_use_each_stage_storage_stride(
    tmp_path, monkeypatch
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
    tape_calls = []

    def fake_tape(key, *, multiplicity, storage_atoms, target_atoms, steps):
        tape_calls.append((multiplicity, storage_atoms, target_atoms, steps))
        return (
            jnp.zeros((multiplicity, target_atoms, 3), dtype=jnp.float32),
            jnp.zeros((steps, multiplicity, target_atoms, 3), dtype=jnp.float32),
        )

    monkeypatch.setattr(api, "_prefix_stable_noise_tape", fake_tape)

    def fake_predict(params, feats, key, **kwargs):
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
        diffusion_samples=2,
        affinity_diffusion_samples=3,
        steps=2,
        affinity_steps=2,
        padding=PaddingConfig(tokens=8, atoms=32, msa=1),
    )

    assert tape_calls == [(2, 5, 32, 2), (3, 7, 32, 2)]
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
            steps=2,
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
    with pytest.raises(ValueError, match="diffusion_samples"):
        api.predict(
            seq=["ACD"],
            weights=tmp_path / "boltz2_conf",
            mols=tmp_path,
            diffusion_samples=0,
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
        assert archive["single"].shape == (1, 2, 4)
        assert archive["pair"].shape == (1, 2, 2, 2)
    assert result["padding"]["primary"]["target"] == {
        "tokens": 8,
        "atoms": 32,
        "msa": 1,
    }
