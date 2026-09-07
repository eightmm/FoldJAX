import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.boltz_foldjax_capture import (
    StageObserver,
    load_inputs,
    make_prediction,
    native_settings,
    prediction_options,
)


def _meta():
    return {
        "num_samples": 5,
        "num_steps": 200,
        "num_recycles": 3,
        "seed": 101,
        "precision": "bf16-mixed",
        "kernels": True,
        "subsample_msa": False,
        "step_scale": 1.5,
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.003,
        "sigma_data": 16.0,
    }


def _effective():
    return {
        "float32_matmul_precision": "highest",
        "cuda_autocast_enabled": True,
        "cuda_autocast_dtype": "torch.bfloat16",
        "use_templates": True,
        "steering_args": {
            "fk_steering": False,
            "physical_guidance_update": False,
            "contact_guidance_update": True,
        },
    }


def test_options_keep_native_schedule_and_full_sequential_confidence():
    full = prediction_options(_meta(), _effective(), trunk_only=False, bfactor=True)
    assert full["confidence_sequentially"]
    assert full["run_confidence"] and full["return_confidence_logits"]
    assert full["run_distogram"] and full["run_bfactor"]
    assert full["use_scan"] and full["trunk_use_scan"] and full["score_use_scan"]
    assert full["compute_dtype"] == "bfloat16"
    assert full["triangle_backend"] == "cueq"
    assert full["matmul_precision"] == "highest"
    assert full["use_template"]
    trunk = prediction_options(_meta(), _effective(), trunk_only=True)
    assert trunk["stop_after_trunk"] and not trunk["run_confidence"]
    assert not trunk["run_distogram"]


@pytest.mark.parametrize("defect", [None, "incomplete", "fp32", "steps", "steering"])
def test_native_settings_fail_closed(tmp_path, defect):
    meta, effective = _meta(), _effective()
    if defect == "fp32":
        meta["precision"] = "32"
    elif defect == "steps":
        meta["num_steps"] = 20
    elif defect == "steering":
        effective["steering_args"]["fk_steering"] = True
    for name, value in (
        ("tape.json", meta),
        ("effective-model-settings.json", effective),
        ("capture-complete.json", {"passed": defect != "incomplete"}),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    if defect:
        with pytest.raises(ValueError):
            native_settings(tmp_path)
    else:
        assert native_settings(tmp_path) == (meta, effective)


def test_runtime_callbacks_keep_scan_and_save_only_first_last_cycles(tmp_path):
    embedder = SimpleNamespace(
        atom_encoder_forward=lambda params, feats: (feats["x"], feats["x"], feats["x"]),
        atom_attention_encoder_forward=lambda params, q, c: (q + 2, q, c),
    )

    def input_embedder(params, feats):
        q, c, _ = embedder.atom_encoder_forward(params, feats)
        return embedder.atom_attention_encoder_forward(params, q, c)[0]

    trunk = SimpleNamespace(
        input_embedder_forward=input_embedder,
        relative_position_forward=lambda params, feats: jnp.ones_like(feats["x"]),
        msa_module_forward=lambda params, z, *a: z * 0.25,
        pairformer_module_forward=lambda params, s, z: (s + 1, z + 1),
    )
    observer = StageObserver(tmp_path, 3)

    @jax.jit
    def run(value):
        feats = {"x": value}
        s = trunk.input_embedder_forward(None, feats)
        z = trunk.relative_position_forward(None, feats)

        def cycle(carry, unused):
            s, z = carry
            z = z + trunk.msa_module_forward(None, z, s, feats)
            return trunk.pairformer_module_forward(None, s, z), None

        return jax.lax.scan(cycle, (s, z), None, length=4)[0]

    original = trunk.msa_module_forward
    with observer.install(trunk, embedder):
        output = run(jnp.ones(2, jnp.bfloat16))
        jax.block_until_ready(output)
        jax.effects_barrier()
    observer.validate()
    assert trunk.msa_module_forward is original
    assert observer.counts["msa_module"] == observer.counts["pairformer_module"] == 4
    np.testing.assert_array_equal(output[0], [7, 7])
    assert (tmp_path / "trunk-boundaries/cycle-00/msa_module.npz").exists()
    assert (tmp_path / "trunk-boundaries/cycle-03/pairformer_module.npz").exists()
    assert not (tmp_path / "trunk-boundaries/cycle-01").exists()
    metadata = json.loads(
        (tmp_path / "trunk-boundaries/input_embedder.tree.json").read_text()
    )
    assert metadata[""]["dtype"] == "bfloat16"


def test_replay_schedule_is_dynamic_jit_input_and_native_callable_is_restored():
    def original_schedule(*args, **kwargs):
        raise AssertionError("not native tape")

    trunk = SimpleNamespace(_sample_schedule=original_schedule)
    traces = []

    def predict(params, feats, key, **kwargs):
        traces.append(True)
        assert kwargs["use_scan"] and kwargs["confidence_sequentially"]
        sigmas = trunk._sample_schedule(2)
        return {
            "sample_atom_coords": kwargs["init_noise"] + sigmas[0],
            "pae_logits": kwargs["step_noises"][0],
            "pdistogram": kwargs["aug_transforms"][0],
        }

    tape = {
        "sigmas": jnp.array([4.0, 2.0, 0.0]),
        "init_noise": jnp.zeros((5, 2, 3)),
        "step_noises": jnp.ones((2, 5, 2, 3)),
        "rotations": jnp.ones((2, 5, 3, 3)),
        "translations": jnp.zeros((2, 5, 1, 3)),
    }
    run = jax.jit(
        make_prediction(
            predict, trunk, {"use_scan": True, "confidence_sequentially": True}
        )
    )
    first = run({}, {}, jax.random.PRNGKey(101), tape)
    second = run(
        {}, {}, jax.random.PRNGKey(101), {**tape, "sigmas": tape["sigmas"] + 5}
    )
    np.testing.assert_array_equal(first["sample_atom_coords"], np.full((5, 2, 3), 4))
    np.testing.assert_array_equal(second["sample_atom_coords"], np.full((5, 2, 3), 9))
    assert len(traces) == 1
    assert trunk._sample_schedule is original_schedule
    assert set(first) == {"sample_atom_coords", "pae_logits", "pdistogram"}


def test_loader_rejects_integer_overflow_before_legacy_narrowing(tmp_path):
    np.savez(tmp_path / "features.npz", index=np.array([2**40], np.int64))
    with pytest.raises(ValueError, match="losslessly narrowed"):
        load_inputs(SimpleNamespace(), tmp_path, _meta())


def test_stage_rejects_missing_or_duplicate_execution(tmp_path):
    observer = StageObserver(tmp_path, 3)
    with pytest.raises(RuntimeError, match="incomplete stage"):
        observer.validate()
    observer.callback("input_embedder", np.ones(2))
    with pytest.raises(RuntimeError, match="unexpected repeated"):
        observer.callback("input_embedder", np.ones(2))
