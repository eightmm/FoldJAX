"""CPU diagnostic contracts; these tests do not reproduce GPU variation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench import esmc_state_diagnostics as diagnostic
from foldjax.models.esmfold2 import inference
from tests.models.esmfold2.test_deterministic_ops import _toy_settings
from tests.models.esmfold2.test_lm_embedding_cache import _parameters
from tests.models.esmfold2.test_lm_state_storage import _checkpoint, _features


def _loaded():
    parameters = _parameters()
    parameters["language_model.base_z_combine"] = parameters[
        "language_model.base_z_combine"
    ][:4]
    return inference.LoadedModel(
        parameters=parameters,
        settings=replace(
            inference.structure_model.ModelSettings(), trunk_dtype="bfloat16"
        ),
        esmc_parameters=_checkpoint(jnp.bfloat16),
        esmc_settings=replace(_toy_settings(), autocast_bfloat16=True),
    )


@pytest.mark.parametrize("storage", ["stack", "buffer"])
@pytest.mark.parametrize("barrier", [False, True])
def test_capture_restores_actual_dtypes_and_replays_identical_values(
    tmp_path, storage, barrier
):
    loaded = _loaded()
    ids, sequences, expand = inference.esmc_model.pack_lm_inputs(
        *_features(), packed_length=24
    )
    tokens, records = diagnostic._capture(
        jnp.asarray(ids),
        jnp.asarray(sequences),
        expand,
        loaded.esmc_parameters,
        loaded.esmc_settings,
        tmp_path,
        storage=storage,
        deterministic=False,
    )
    assert [row["dtype"] for row in records] == ["bfloat16"] * 3 + ["float32"]
    assert [
        str(value.dtype) for value in diagnostic._captured_layers(tmp_path, records)
    ] == [
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "float32",
    ]
    expected = inference.language_model_states(
        dict(zip(inference.LANGUAGE_MODEL_FEATURES, _features(), strict=True)),
        loaded,
        packed_length=24,
    )
    assert diagnostic._differences(tokens, expected)["bitwise_equal"]
    replay = diagnostic._replay(
        tmp_path, records, expand, loaded, deterministic=False, barrier=barrier
    )
    for mode in ("stack", "donated", "non-donated"):
        assert replay[mode]["transport_vs_stack"]["bitwise_equal"]
        assert replay[mode]["projection_vs_stack"]["bitwise_equal"]
        assert "lowering_sha256" in replay[mode]["projection_program"]
    captured = tmp_path / records[0]["file"]
    captured.write_bytes(b"changed")
    with pytest.raises(ValueError, match="captured layer file changed"):
        list(diagnostic._captured_layers(tmp_path, records))


def test_layer_identity_distinguishes_dtype_shape_and_first_divergence():
    a = diagnostic._array_identity(np.ones((2, 3), np.float32))
    assert a != diagnostic._array_identity(np.ones((2, 3), np.float64))
    assert a != diagnostic._array_identity(np.ones((3, 2), np.float32))
    before = [{"identity": a}, {"identity": a}]
    assert diagnostic._first_divergence(before, before) is None
    after = [
        before[0],
        {"identity": diagnostic._array_identity(np.zeros((2, 3), np.float32))},
    ]
    assert diagnostic._first_divergence(before, after)["layer_index"] == 1


@pytest.mark.parametrize("mode,count", [("native", 10), ("capture", 2), ("both", 12)])
@pytest.mark.parametrize("fingerprints", [False, True])
def test_fresh_workers_keep_repeats_and_continue_after_failure(
    tmp_path, monkeypatch, mode, count, fingerprints
):
    args = SimpleNamespace(
        features=tmp_path / "features.npz",
        binding=tmp_path / "binding.json",
        weights=tmp_path / "weights",
        output=tmp_path / "result",
        mode=mode,
        repeats=5,
        packed_length=1026,
        compute_dtype="float32",
        esmc_dtype="bfloat16",
        deterministic=True,
        compile_esmc_blocks=True,
        replay_barrier=True,
        compiler_fingerprints=fingerprints,
    )
    args.binding.write_text("{}")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    diagnostic._coordinator(args)
    assert len(calls) == count
    for command, kwargs in calls:
        assert command[:3] == [
            diagnostic.sys.executable,
            "-m",
            "bench.esmc_state_diagnostics",
        ]
        assert "--worker" in command and "--deterministic" in command
        assert "--compile-esmc-blocks" in command
        assert "--replay-barrier" in command
        assert ("--compiler-fingerprints" in command) == fingerprints
        assert command[command.index("--packed-length") + 1] == "1026"
        assert command[command.index("--compute-dtype") + 1] == "float32"
        assert kwargs["check"] is False
    report = diagnostic.json.loads((args.output / "comparison.json").read_text())
    output_names = [command[command.index("--output") + 1] for command, _ in calls]
    log_names = [kwargs["stdout"].name for _, kwargs in calls]
    assert len(set(output_names)) == count
    assert len(set(log_names)) == count
    assert set(report["return_codes"]) == {
        diagnostic.Path(name).name for name in output_names
    }
    assert list(report["return_codes"].values()).count(1) == 1
    assert report["full_model_admission"] is None
    if mode != "capture":
        assert [command[command.index("--arm") + 1] for command, _ in calls[:10]] == [
            arm for _ in range(5) for arm in ("stack", "buffer")
        ]


def test_comparison_records_within_and_between_storage_process_variation(tmp_path):
    for replicate in range(5):
        for arm in ("stack", "buffer"):
            directory = tmp_path / f"{arm}-{replicate}"
            directory.mkdir()
            value = np.ones((1, 2, 3), np.float32)
            if arm == "stack" and replicate == 0:
                value[0, 0, 0] = 2
            np.save(directory / "embedding.npy", value)
            report = {
                **_matching_controls(),
                "arm": arm,
                "storage": arm,
                "replicate": replicate,
                "cold_warm": diagnostic._differences(value, value),
            }
            diagnostic._write_json(directory / "report.json", report)
    result = diagnostic._comparison(tmp_path)
    assert len(result["native_pairwise"]) == 45
    assert len([row for row in result["native_pairwise"] if row["same_replicate"]]) == 5
    for group, expected in (
        ("within-stack", 10),
        ("within-buffer", 10),
        ("between-storage", 25),
    ):
        assert result["native_distributions"][group]["pairs"] == expected
    assert result["native_distributions"]["within-stack"]["max"] == 1
    assert result["native_distributions"]["within-buffer"]["max"] == 0
    differences = [row for row in result["native_pairwise"] if row["max_abs_error"]]
    assert differences and all(row["different_values"] == 1 for row in differences)
    assert all(result["binding_matches"].values())


def _matching_controls():
    return {
        "binding": {"features_sha256": "features"},
        "packed_inputs": {"ids": {"sha256": "packed"}},
        "actual_esmc_parameters": {"sha256": "weights"},
        "actual_projection_parameters": {"sha256": "projection"},
        "esmc_settings": {"d_model": 8},
        "compute_dtype": "bfloat16",
        "deterministic": False,
        "control_policy": diagnostic._control_policy(SimpleNamespace()),
        "device_kind": "test-device",
        "jax": "test-jax",
        "jaxlib": "test-jaxlib",
        "matmul_precision": "None",
        "xla_flags": "",
        "allocator_environment": {"XLA_CLIENT_MEM_FRACTION": "0.9"},
        "execution_environment": {
            "CUDA_LAUNCH_BLOCKING": None,
            "CUBLAS_WORKSPACE_CONFIG": None,
            "NVIDIA_TF32_OVERRIDE": None,
        },
        "packed_length": 24,
        "lm_mask_pct": 0.0,
        "diagnostic_sha256": "diagnostic",
        "compiler_observer_enabled": False,
        "compiler_observer_source_sha256": None,
    }


@pytest.mark.parametrize(
    "changed",
    [
        "actual_esmc_parameters",
        "deterministic",
        "control_policy",
        "malformed-policy",
        "execution_environment",
        "missing",
        "malformed",
    ],
)
def test_mismatched_controls_preserve_raw_pairs_but_are_excluded(tmp_path, changed):
    reports = []
    for index, arm in enumerate(("stack", "buffer")):
        directory = tmp_path / f"{arm}-0"
        directory.mkdir()
        value = np.full((1, 2, 3), index, np.float32)
        np.save(directory / "embedding.npy", value)
        reports.append(
            {
                **_matching_controls(),
                "arm": arm,
                "storage": arm,
                "replicate": 0,
                "cold_warm": diagnostic._differences(value, value),
            }
        )
    if changed == "missing":
        del reports[1]["actual_esmc_parameters"]
    elif changed == "malformed":
        reports[1]["actual_esmc_parameters"] = "unknown"
    elif changed == "actual_esmc_parameters":
        reports[1][changed] = {"sha256": "different-weights"}
    elif changed == "deterministic":
        reports[1][changed] = True
    elif changed == "control_policy":
        reports[1][changed]["compile_esmc_blocks"] = True
    elif changed == "malformed-policy":
        reports[1]["control_policy"] = {}
    else:
        reports[1][changed]["CUDA_LAUNCH_BLOCKING"] = "1"
    for arm, report in zip(("stack", "buffer"), reports, strict=True):
        diagnostic._write_json(tmp_path / f"{arm}-0" / "report.json", report)
    result = diagnostic._comparison(tmp_path)
    (pair,) = result["native_pairwise"]
    assert pair["max_abs_error"] == 1 and pair["controls_match"] is not True
    if changed in ("missing", "malformed", "malformed-policy", "deterministic"):
        assert pair["controls_match"] is None
        expected = (
            "actual_esmc_parameters"
            if changed in ("missing", "malformed")
            else "control_policy"
        )
        assert expected in pair["unverified_control_fields"]
    else:
        assert changed in pair["mismatched_control_fields"]
    assert result["native_distributions"]["between-storage"]["pairs"] == 0
    assert result["native_distributions"]["between-storage"]["excluded_pairs"] == 1


@pytest.mark.parametrize("fingerprints", [False, True])
@pytest.mark.parametrize(
    "deterministic,compiled",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_native_worker_reports_cold_warm_without_inserting_a_layer_observer(
    tmp_path, monkeypatch, fingerprints, deterministic, compiled
):
    device = jax.devices("cpu")[0]
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    runtime_calls = []
    monkeypatch.setattr(
        diagnostic,
        "_benchmark_runtime",
        lambda **kw: (runtime_calls.append(kw) or jax, device),
    )
    monkeypatch.setattr(diagnostic, "_verify_binding", lambda *args: None)
    loaded, calls = _loaded(), []
    monkeypatch.setattr(inference, "load_language_model_stage", lambda *a, **kw: loaded)
    original = inference.language_model_embedding
    encode_calls = []
    original_encode = inference.esmc_model.encode

    def encode(*args, **kwargs):
        encode_calls.append(kwargs)
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(inference.esmc_model, "encode", encode)

    def embedding(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "language_model_embedding", embedding)
    monkeypatch.setattr(
        diagnostic, "_capture", lambda *a, **kw: pytest.fail("native capture")
    )
    features = tmp_path / "features.npz"
    np.savez(
        features,
        **dict(zip(inference.LANGUAGE_MODEL_FEATURES, _features(), strict=True)),
    )
    binding = tmp_path / "binding.json"
    diagnostic._write_json(binding, {"features_sha256": diagnostic._sha256(features)})
    args = SimpleNamespace(
        features=features,
        binding=binding,
        weights=tmp_path,
        output=tmp_path / "worker",
        arm="buffer",
        replicate=2,
        deterministic=deterministic,
        compile_esmc_blocks=compiled,
        packed_length=24,
        compute_dtype=None,
        esmc_dtype="bfloat16",
        replay_barrier=False,
        compiler_fingerprints=fingerprints,
    )
    diagnostic._worker(args)
    report = diagnostic.json.loads((args.output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["cold_warm"]["bitwise_equal"]
    assert report["lm_cold_warm_max_abs_error"] == 0
    assert report["lm_mask_pct"] == 0 and "layers" not in report
    assert ("compilation_observation" in report) == fingerprints
    assert report["compiler_observer_enabled"] == fingerprints
    assert report["execution_environment"] == {
        "CUDA_LAUNCH_BLOCKING": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    assert report["memory"]["scope"] == report["memory_scope"]
    assert "structure" not in report["memory"]["scope"]
    assert report["actual_esmc_parameters"]["leaves"]
    assert len(calls) == 2 and all(row["state_storage"] == "buffer" for row in calls)
    assert runtime_calls == [{"deterministic": deterministic}]
    assert all(not row["deterministic"] for row in calls)
    assert len(encode_calls) == 2
    assert all(row["compile_blocks"] is compiled for row in encode_calls)
    assert report["control_policy"]["deterministic_requested"] is deterministic
    assert report["control_policy"]["compile_esmc_blocks"] is compiled
    assert not report["control_policy"]["per_executable_deterministic"]


@pytest.mark.parametrize("failed", [False, True])
def test_cli_defaults_and_propagates_worker_failure_after_collecting_reports(
    tmp_path, monkeypatch, failed
):
    captured = []

    def coordinator(args):
        captured.append(args)
        return {"stack-0": 0, "buffer-0": 1 if failed else 0}

    monkeypatch.setattr(diagnostic, "_coordinator", coordinator)
    monkeypatch.setattr(
        diagnostic.sys,
        "argv",
        [
            "probe",
            "--features",
            "features.npz",
            "--binding",
            "binding.json",
            "--weights",
            str(tmp_path),
            "--output",
            str(tmp_path / "result"),
        ],
    )
    assert diagnostic.main() == int(failed)
    assert captured[0].repeats == 5
    assert captured[0].mode == "native"
    assert not captured[0].compiler_fingerprints and not captured[0].deterministic
    assert not captured[0].compile_esmc_blocks


def _binding_files(tmp_path):
    from bench import esmfold2_performance_probe

    weights, source, output = (tmp_path / name for name in ("weights", "source", "out"))
    for directory in (weights, source, output):
        directory.mkdir()
    weight, code = weights / "checkpoint", source / "code.py"
    weight.write_bytes(b"checkpoint content")
    code.write_text("source content")
    features = tmp_path / "features.npz"
    np.savez(features, input_ids=np.ones((1, 2), np.int32))
    binding = tmp_path / "binding.json"
    diagnostic._write_json(
        binding,
        {
            "features_sha256": diagnostic._sha256(features),
            "weights": {
                "checkpoint": {
                    "sha256": diagnostic._sha256(weight),
                    "local_change_anchor": [1, 2, 3, 4, 5],
                }
            },
            "source": {"code.py": diagnostic._sha256(code)},
            "probe_sha256": diagnostic._sha256(
                diagnostic.Path(esmfold2_performance_probe.__file__)
            ),
        },
    )
    return binding, weights, features, output, source


def test_reanchoring_verifies_content_and_preserves_original_binding(tmp_path):
    binding, weights, features, output, source = _binding_files(tmp_path)
    original = binding.read_bytes()
    copied = diagnostic._reanchor_binding(binding, weights, features, output, source)
    assert copied != binding and binding.read_bytes() == original
    before, after = (
        diagnostic.json.loads(path.read_text()) for path in (binding, copied)
    )
    assert after["weights"]["checkpoint"][
        "local_change_anchor"
    ] == diagnostic._local_anchor(weights / "checkpoint")
    after["weights"]["checkpoint"]["local_change_anchor"] = before["weights"][
        "checkpoint"
    ]["local_change_anchor"]
    assert after == before


@pytest.mark.parametrize(
    "invalid", ["missing-hash", "weight-content", "source", "features"]
)
@pytest.mark.parametrize("rebind_probe", [False, True])
def test_reanchoring_refuses_unverified_provenance(tmp_path, invalid, rebind_probe):
    binding, weights, features, output, source = _binding_files(tmp_path)
    if invalid == "missing-hash":
        document = diagnostic.json.loads(binding.read_text())
        del document["weights"]["checkpoint"]["sha256"]
        diagnostic._write_json(binding, document)
    elif invalid == "weight-content":
        (weights / "checkpoint").write_bytes(b"different content")
    elif invalid == "source":
        (source / "code.py").write_text("different source")
    else:
        features.write_bytes(b"different features")
    with pytest.raises(ValueError):
        diagnostic._reanchor_binding(
            binding, weights, features, output, source, rebind_probe=rebind_probe
        )
    assert not (output / "local-binding.json").exists()


def test_reanchoring_detects_change_during_content_hash(tmp_path, monkeypatch):
    binding, weights, features, output, source = _binding_files(tmp_path)
    original = diagnostic._sha256

    def changing(path):
        value = original(path)
        if path == weights / "checkpoint":
            path.write_bytes(b"changed while hashing")
        return value

    monkeypatch.setattr(diagnostic, "_sha256", changing)
    with pytest.raises(ValueError, match="changed while hashing"):
        diagnostic._reanchor_binding(binding, weights, features, output, source)
    assert not (output / "local-binding.json").exists()


@pytest.mark.parametrize(
    "reanchor_weights,rebind_probe", [(True, False), (False, True), (True, True)]
)
def test_reanchored_copy_is_forwarded_to_every_worker(
    tmp_path, monkeypatch, reanchor_weights, rebind_probe
):
    binding = tmp_path / "binding.json"
    binding.write_text("{}")
    output = tmp_path / "result"
    args = SimpleNamespace(
        features=tmp_path / "features.npz",
        binding=binding,
        weights=tmp_path,
        output=output,
        mode="native",
        repeats=1,
        packed_length=None,
        compute_dtype=None,
        esmc_dtype="bfloat16",
        deterministic=False,
        replay_barrier=False,
        compiler_fingerprints=False,
        reanchor_weights=reanchor_weights,
        rebind_probe=rebind_probe,
    )

    def reanchor(original, weights, features, destination, source_root, **kwargs):
        assert original == binding
        assert kwargs == {
            "reanchor_weights": reanchor_weights,
            "rebind_probe": rebind_probe,
        }
        result = destination / "local-binding.json"
        result.write_text('{"verified": true}')
        return result

    calls = []
    monkeypatch.setattr(diagnostic, "_reanchor_binding", reanchor)
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0),
    )
    diagnostic._coordinator(args)
    assert len(calls) == 2
    assert all(
        command[command.index("--binding") + 1] == str(output / "local-binding.json")
        for command in calls
    )
    assert binding.read_text() == "{}"
    plan = diagnostic.json.loads((output / "plan.json").read_text())
    assert plan["reanchored_weights"] is reanchor_weights
    assert plan["probe_rebound"] is rebind_probe
    assert plan["input_binding_sha256"] != plan["worker_binding_sha256"]


@pytest.mark.parametrize("reanchor", [False, True])
def test_archived_probe_rebinding_is_explicit_content_verified_and_immutable(
    tmp_path, reanchor
):
    binding, weights, features, output, source = _binding_files(tmp_path)
    document = diagnostic.json.loads(binding.read_text())
    document["probe_sha256"] = "0" * 64
    if not reanchor:
        document["weights"]["checkpoint"]["local_change_anchor"] = (
            diagnostic._local_anchor(weights / "checkpoint")
        )
    diagnostic._write_json(binding, document)
    original = binding.read_bytes()
    with pytest.raises(ValueError, match="probe source changed"):
        diagnostic._reanchor_binding(
            binding, weights, features, output, source, reanchor_weights=reanchor
        )
    assert not (output / "local-binding.json").exists()
    copied = diagnostic._reanchor_binding(
        binding,
        weights,
        features,
        output,
        source,
        reanchor_weights=reanchor,
        rebind_probe=True,
    )
    after = diagnostic.json.loads(copied.read_text())
    assert binding.read_bytes() == original
    assert after["probe_rebinding"]["previous_sha256"] == "0" * 64
    assert (
        after["probe_rebinding"]["current_sha256"] == after["probe_sha256"] != "0" * 64
    )
    diagnostic._verify_binding(after, weights, source)


def test_probe_rebinding_alone_cannot_reanchor_a_changed_checkpoint(tmp_path):
    binding, weights, features, output, source = _binding_files(tmp_path)
    with pytest.raises(ValueError, match="checkpoint changed"):
        diagnostic._reanchor_binding(
            binding,
            weights,
            features,
            output,
            source,
            reanchor_weights=False,
            rebind_probe=True,
        )
    assert not (output / "local-binding.json").exists()
