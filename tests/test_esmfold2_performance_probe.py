from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench import esmfold2_performance_probe as probe
from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.models import diffusion, model


@pytest.mark.parametrize("experiment", probe.EXPERIMENTS)
def test_experiments_change_only_their_own_variable(experiment):
    baseline = probe.arm_options(experiment, "baseline")
    assert probe.arm_options(experiment, "baseline_repeat") == baseline
    candidate = probe.arm_options(experiment, "candidate")
    changed = {key for key in baseline if baseline[key] != candidate[key]}
    expected = (
        "rigid_align_backend" if experiment == "alignment" else "compile_esmc_blocks"
    )
    assert changed == {expected}
    assert baseline == {"rigid_align_backend": "svd", "compile_esmc_blocks": False}


def test_released_schedule_changes_only_sample_count_and_explicit_alignment():
    native = model.ModelSettings(
        num_recycles=7,
        num_samples=32,
        diffusion=diffusion.DiffusionSettings(num_steps=14, noise_scale=1.003),
    )
    effective = probe._settings(native, smoke=False, alignment="jacobi")
    assert effective == replace(
        native,
        num_samples=5,
        diffusion=replace(native.diffusion, rigid_align_backend="jacobi"),
    )
    smoke = probe._settings(native, smoke=True, alignment="svd")
    assert smoke.num_recycles == 0 and smoke.diffusion.num_steps == 2


def test_native_diffusion_tape_keeps_the_actual_draw_stream(monkeypatch):
    settings = model.ModelSettings(
        num_samples=5,
        diffusion=diffusion.DiffusionSettings(
            num_steps=3, c_token=2, noise_scale=1.003
        ),
    )
    atom_mask = np.asarray([[1, 1, 1, 1, 1, 0]], np.float32)
    key = inference.seed_key(101)
    tape = probe.diffusion_tape(key, atom_mask, settings)
    sample_key = jax.random.split(key, 5)[-1]
    native_key, initial_key = jax.random.split(sample_key)
    expected_initial = jax.random.normal(initial_key, (5, 6, 3), jnp.float32)
    np.testing.assert_array_equal(tape["diffusion_initial_normal"], expected_initial)
    _, augmentation, churn = jax.random.split(native_key, 3)
    rotation, translation = jax.random.split(augmentation)
    for name, expected in (
        ("diffusion_rotation_quaternions", jax.random.normal(rotation, (5, 4))),
        ("diffusion_translations", jax.random.normal(translation, (5, 1, 3))),
        ("diffusion_churn_normals", jax.random.normal(churn, (5, 6, 3))),
    ):
        np.testing.assert_array_equal(tape[name][0], expected)
    cache = SimpleNamespace(
        atom_mask=jnp.asarray(np.repeat(atom_mask, 5, axis=0)), n_tokens=2
    )

    def denoise(x, sigma, *args, **kwargs):
        return x / (1 + sigma[:, None, None]), jnp.broadcast_to(x.mean(), (5, 2, 2))

    monkeypatch.setattr(diffusion, "diffusion_module", denoise)
    ordinary = diffusion.sample(
        sample_key,
        jnp.zeros((1, 2, 2)),
        cache,
        {},
        settings=settings.diffusion,
        num_samples=5,
    )
    replay = diffusion.sample(
        sample_key,
        jnp.zeros((1, 2, 2)),
        cache,
        {},
        settings=settings.diffusion,
        num_samples=5,
        **tape,
    )
    for left, right in zip(ordinary, replay, strict=True):
        np.testing.assert_allclose(left, right, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("existing", [None, "0.85"])
def test_allocator_matches_predict_and_preserves_the_callers_choice(
    monkeypatch, existing
):
    from foldjax import oom

    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    if existing is not None:
        monkeypatch.setenv(oom.FRACTION_ENV, existing)
    updates = []
    monkeypatch.setattr(jax, "local_devices", lambda: [SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(jax.config, "update", lambda *args: updates.append(args))
    probe._benchmark_runtime()
    selected = oom.configured_fraction_env()
    assert probe.os.environ[selected] == (existing or "0.9")
    assert updates == [("jax_enable_compilation_cache", False)]


def test_binding_rejects_changed_weights_and_sources(tmp_path):
    weights, source = tmp_path / "weights", tmp_path / "source"
    weights.mkdir()
    source.mkdir()
    checkpoint, code = weights / "weight", source / "code.py"
    checkpoint.write_bytes(b"checkpoint")
    code.write_text("original")
    binding = {
        "weights": {"weight": {"local_change_anchor": probe._local_anchor(checkpoint)}},
        "source": {"code.py": probe._sha256(code)},
        "probe_sha256": probe._sha256(probe.Path(probe.__file__)),
    }
    probe._verify_binding(binding, weights, source)
    code.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        probe._verify_binding(binding, weights, source)
    code.write_text("original")
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        probe._verify_binding(binding, weights, source)


def test_coordinator_prepares_tape_in_its_own_process_before_every_arm(
    tmp_path, monkeypatch
):
    weights = tmp_path / "weights"
    (weights / "esmc").mkdir(parents=True)
    for filename in (
        "config.json",
        "model.safetensors",
        "ccd.pkl",
        "esmc/config.json",
        "esmc/model.safetensors",
    ):
        (weights / filename).write_bytes(b"fixture")
    jobs = [tmp_path / "small.json", tmp_path / "large.json"]
    for index, path in enumerate(jobs):
        path.write_text(probe.json.dumps({"tokens": index + 2}))

    def features(document, **kwargs):
        return {
            "token_attention_mask": np.ones((1, document["tokens"]), np.float32),
            "atom_attention_mask": np.ones((1, 8), np.float32),
        }

    monkeypatch.setattr(inference, "build_common_job_features", features)
    calls = []
    monkeypatch.setattr(
        probe.subprocess, "run", lambda command, **kw: calls.append((command, kw))
    )
    monkeypatch.setattr(
        probe, "_comparison", lambda *args: {"full_model_admission": None}
    )
    args = SimpleNamespace(
        weights=weights,
        input=jobs,
        case=None,
        output=tmp_path / "evidence",
        experiment="alignment",
        seed=101,
        smoke=False,
    )
    probe._coordinator(args)
    assert len(calls) == 8
    for offset in (0, 4):
        assert "--prepare-tape" in calls[offset][0]
        assert "--worker" not in calls[offset][0]
        for command, kwargs in calls[offset + 1 : offset + 4]:
            assert "--worker" in command and "--prepare-tape" not in command
            assert kwargs["check"] is True
    result = probe.json.loads((args.output / "report.json").read_text())
    assert result["full_model_admission"] is None
