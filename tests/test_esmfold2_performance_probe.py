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
    expected = {
        "alignment": "rigid_align_backend",
        "esmc": "compile_esmc_blocks",
        "lm-storage": "lm_state_storage",
    }[experiment]
    assert changed == {expected}
    assert baseline == {
        "rigid_align_backend": "svd",
        "compile_esmc_blocks": False,
        **({"lm_state_storage": "stack"} if experiment == "lm-storage" else {}),
    }


@pytest.mark.parametrize("arm", probe.ARMS)
def test_storage_experiment_alone_names_the_native_storage_keyword(arm):
    for experiment in probe.DEFAULT_EXPERIMENTS:
        options = probe.arm_options(experiment, arm)
        assert set(options) == {"rigid_align_backend", "compile_esmc_blocks"}
        assert probe._language_model_keywords(options) == {}
    options = probe.arm_options("lm-storage", arm)
    expected = "buffer" if arm == "candidate" else "stack"
    assert options == {
        "rigid_align_backend": "svd",
        "compile_esmc_blocks": False,
        "lm_state_storage": expected,
    }
    assert probe._language_model_keywords(options) == {"state_storage": expected}


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


@pytest.mark.parametrize("experiment", ["alignment", "lm-storage", "both"])
@pytest.mark.parametrize("controls", [False, True])
def test_coordinator_prepares_tape_in_its_own_process_before_every_arm(
    tmp_path, monkeypatch, experiment, controls
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
        experiment=experiment,
        seed=101,
        smoke=False,
        deterministic=controls,
        compile_esmc_blocks=controls and experiment != "both",
        compiler_fingerprints=controls,
    )
    probe._coordinator(args)
    experiments = probe.DEFAULT_EXPERIMENTS if experiment == "both" else (experiment,)
    per_case = 1 + 3 * len(experiments)
    assert len(calls) == 2 * per_case
    for offset in (0, per_case):
        assert "--prepare-tape" in calls[offset][0]
        assert "--worker" not in calls[offset][0]
        assert ("--deterministic" in calls[offset][0]) is controls
        assert "--compiler-fingerprints" not in calls[offset][0]
        workers = calls[offset + 1 : offset + per_case]
        assert [
            (
                command[command.index("--experiment") + 1],
                command[command.index("--arm") + 1],
            )
            for command, _kwargs in workers
        ] == [(selected, arm) for selected in experiments for arm in probe.ARMS]
        for command, kwargs in workers:
            assert "--worker" in command and "--prepare-tape" not in command
            assert kwargs["check"] is True
            for flag in (
                "--deterministic",
                "--compile-esmc-blocks",
                "--compiler-fingerprints",
            ):
                expected = (
                    controls and experiment != "both"
                    if flag == "--compile-esmc-blocks"
                    else controls
                )
                assert (flag in command) is expected
    result = probe.json.loads((args.output / "report.json").read_text())
    assert result["full_model_admission"] is None
    plan = probe.json.loads((args.output / "plan.json").read_text())
    assert plan["experiments"] == list(experiments)
    assert plan["samples"] == 5 and plan["smoke_only"] is False
    assert plan["control_policy"]["deterministic_requested"] is controls
    assert plan["control_policy"]["compile_esmc_blocks"] is (
        controls and experiment != "both"
    )
    assert plan["compiler_observer_enabled"] is controls
    if experiment == "both":
        assert list(experiments) == ["alignment", "esmc"]


def test_deterministic_policy_is_set_before_device_initialization(monkeypatch):
    import sys

    from foldjax import oom

    monkeypatch.setitem(
        sys.modules,
        "jax._src.xla_bridge",
        SimpleNamespace(backends_are_initialized=lambda: False),
    )
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    observed = []
    monkeypatch.setattr(oom, "set_predict_mem_fraction", lambda: None)

    def devices():
        observed.append(probe.os.environ["XLA_FLAGS"])
        return [SimpleNamespace(platform="gpu")]

    monkeypatch.setattr(jax, "local_devices", devices)
    monkeypatch.setattr(jax.config, "update", lambda *args: None)
    probe._benchmark_runtime(deterministic=True)
    assert observed == [
        "--xla_force_host_platform_device_count=1 "
        "--xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false"
    ]


@pytest.mark.parametrize(
    "existing",
    [
        "--xla_gpu_deterministic_ops=false",
        "--xla_gpu_enable_triton_gemm true",
        "--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false",
    ],
)
def test_deterministic_policy_rejects_explicit_conflicts(monkeypatch, existing):
    monkeypatch.setitem(
        probe.sys.modules,
        "jax._src.xla_bridge",
        SimpleNamespace(backends_are_initialized=lambda: False),
    )
    monkeypatch.setenv("XLA_FLAGS", existing)
    with pytest.raises(ValueError, match="conflicts"):
        probe._configure_deterministic_runtime(True)
    assert probe.os.environ["XLA_FLAGS"] == existing


def test_deterministic_policy_requires_fresh_process_and_default_is_untouched(
    monkeypatch,
):
    monkeypatch.setitem(
        probe.sys.modules,
        "jax._src.xla_bridge",
        SimpleNamespace(backends_are_initialized=lambda: True),
    )
    monkeypatch.setenv("XLA_FLAGS", "caller-flags")
    probe._configure_deterministic_runtime(False)
    assert probe.os.environ["XLA_FLAGS"] == "caller-flags"
    with pytest.raises(RuntimeError, match="fresh process"):
        probe._configure_deterministic_runtime(True)


def test_deterministic_policy_preserves_matching_flags_without_duplication(monkeypatch):
    monkeypatch.setitem(
        probe.sys.modules,
        "jax._src.xla_bridge",
        SimpleNamespace(backends_are_initialized=lambda: False),
    )
    original = "--xla_gpu_deterministic_ops 1 --xla_gpu_enable_triton_gemm=0"
    monkeypatch.setenv("XLA_FLAGS", original)
    probe._configure_deterministic_runtime(True)
    assert probe.os.environ["XLA_FLAGS"] == original


def test_fresh_cpu_process_accepts_the_deterministic_startup_flags():
    environment = {
        **probe.os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    }
    code = """
from bench.esmfold2_performance_probe import _configure_deterministic_runtime
_configure_deterministic_runtime(True)
import jax
import jax.numpy as jnp
assert int(jax.jit(lambda x: x + 1)(jnp.array(1))) == 2
import os
print(os.environ['XLA_FLAGS'])
"""
    completed = probe.subprocess.run(
        [probe.sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--xla_gpu_deterministic_ops=true" in completed.stdout
    assert "--xla_gpu_enable_triton_gemm=false" in completed.stdout


@pytest.mark.parametrize("compiled", [False, True])
def test_encoder_control_does_not_conflate_block_compilation_with_determinism(compiled):
    calls = []
    selected = probe._select_encode(
        lambda *args, **kw: calls.append(kw), compile_blocks=compiled
    )
    selected("ids", deterministic=False)
    assert calls == [{"deterministic": False, "compile_blocks": compiled}]


@pytest.mark.parametrize(
    "flags",
    [[], ["--deterministic", "--compiler-fingerprints", "--compile-esmc-blocks"]],
)
def test_control_flags_are_available_on_the_original_paired_probe(
    tmp_path, monkeypatch, flags
):
    captured = []
    monkeypatch.setattr(probe, "_coordinator", captured.append)
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "probe",
            "--input",
            "small.json",
            "large.json",
            "--weights",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--experiment",
            "lm-storage",
            *flags,
        ],
    )
    probe.main()
    (args,) = captured
    assert args.deterministic is bool(flags)
    assert args.compiler_fingerprints is bool(flags)
    assert args.compile_esmc_blocks is bool(flags)


def test_block_control_cannot_erase_the_existing_esmc_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "probe",
            "--input",
            "small.json",
            "large.json",
            "--output",
            str(tmp_path),
            "--experiment",
            "esmc",
            "--compile-esmc-blocks",
        ],
    )
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
