from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace

import numpy as np
import pytest

# ruff: noqa: E501
from bench import esmfold2_tape as tape


def test_capture_metadata_distinguishes_weight_dtype_from_autocast(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tape, "_sha256", lambda path: "fixture_hash")
    args = Namespace(
        command="capture", output_dir=tmp_path, input_features=tmp_path / "features.npz"
    )
    tape._write_metadata(args, {"dtype": "float32"}, {})
    precision = json.loads((tmp_path / "metadata.json").read_text())["precision"]
    assert precision["checkpoint_dtype"] == "float32"
    assert precision["trunk"] == "native_cuda_bfloat16_autocast"
    assert precision["conditioning"] == "native_mixed_with_bfloat16_z_transitions"


def _shapes() -> tape.TapeShapes:
    return tape.TapeShapes(1, 3, 4, 2, 5, 2, 2, None, 1024, 0.1)


def _events(shapes: tape.TapeShapes):
    pair = np.zeros((1, 3, 3, 2), np.float32)
    atom = np.zeros((5, 4, 3), np.float32)
    rotation = np.zeros((5, 4), np.float32)
    translation = np.zeros((5, 1, 3), np.float32)
    return (
        [pair],
        [np.ones_like(pair), np.ones_like(pair)],
        [],
        [],
        [atom, rotation, translation, atom, rotation, translation, atom],
    )


def test_classifies_complete_native_tape(tmp_path) -> None:
    shapes = _shapes()
    initial, dropout, rand, perm, normal = _events(shapes)
    got = tape.classify_random_events(
        shapes,
        initial_pair=initial,
        dropout=dropout,
        rand=rand,
        randperm=perm,
        normal=normal,
    )
    assert got["lm_dropout_masks"].shape == (2, 1, 3, 3, 2)
    assert got["diffusion_churn_normals"].shape == (2, 5, 4, 3)
    assert got["msa_row_choices"].shape == (0,)
    assert got["msa_row_choices"].dtype == np.int64
    assert got["msa_column_keep"].shape == (0,)
    assert got["msa_column_keep"].dtype == bool
    assert len(got) == 8
    path = tmp_path / "tape.npz"
    tape._save_npz(path, got)
    replay = tape._npz(path)
    assert set(replay) == set(got)
    assert replay["msa_column_keep"].shape == (0,)
    assert replay["msa_row_choices"].dtype == np.int64


def test_raw_outputs_bind_schema_and_reject_tampering(tmp_path):
    output = {
        name: np.ones((5,), np.float32)
        for name in ("plddt", "complex_plddt", "ptm", "iptm")
    }
    output["sample_atom_coords"] = np.ones((5, 4, 3), np.float32)
    schema = tape._save_outputs(tmp_path, "upstream", output, native=False)
    capture = {"output_schema": schema}
    tape._verify_reference_outputs(tmp_path, capture)
    assert "pae_logits" in schema["missing_optional"]
    schema["arrays"]["upstream_confidence.npz"]["fields"]["ptm"]["shape"] = [6]
    with pytest.raises(ValueError, match="schema"):
        tape._verify_reference_outputs(tmp_path, capture)


def test_tree_identity_binds_checkpoint_content_not_just_config(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"one")
    first = tape._tree_identity(tmp_path, (".json", ".safetensors"))
    (tmp_path / "model.safetensors").write_bytes(b"two")
    second = tape._tree_identity(tmp_path, (".json", ".safetensors"))
    assert first["config.json"] == second["config.json"]
    assert first["model.safetensors"] != second["model.safetensors"]


def test_msa_row_tape_retains_query_and_integer_indices():
    shapes = replace(_shapes(), msa_depth=5, max_msa_depth=3)
    initial, dropout, _, _, normal = _events(shapes)
    got = tape.classify_random_events(
        shapes,
        initial_pair=initial,
        dropout=dropout,
        rand=[np.ones((1, 3))],
        randperm=[np.array([3, 1, 0, 2]), np.array([2, 0, 3, 1])],
        normal=normal,
    )
    np.testing.assert_array_equal(got["msa_row_choices"], [[0, 2, 4], [0, 1, 3]])
    assert got["msa_row_choices"].dtype == np.int64


def test_rejects_unclassified_or_missing_random_consumer() -> None:
    shapes = _shapes()
    initial, dropout, rand, perm, normal = _events(shapes)
    with pytest.raises(ValueError, match="diffusion normal"):
        tape.classify_random_events(
            shapes,
            initial_pair=initial,
            dropout=dropout,
            rand=rand,
            randperm=perm,
            normal=normal[:-1],
        )


def test_replay_rejects_missing_core_contract_before_loading(monkeypatch):
    from argparse import Namespace

    from bench.esmfold2_tape import _replay
    from foldjax.models.esmfold2.models import model

    monkeypatch.setattr(model, "predict", lambda key: None)

    # No paths or checkpoint attributes: a path access would fail this test.
    with pytest.raises(RuntimeError, match="full tape replay is not implemented"):
        _replay(Namespace())


def test_all_eight_core_tape_routes_are_exposed():
    import inspect

    from foldjax.models.esmfold2.models.model import predict

    routes = {
        "initial_pair_state",
        "lm_dropout_masks",
        "msa_column_keep",
        "msa_row_choices",
        "diffusion_initial_normal",
        "diffusion_rotation_quaternions",
        "diffusion_translations",
        "diffusion_churn_normals",
    }
    assert not routes - set(inspect.signature(predict).parameters)


def test_actual_replay_settings_retain_raw_heads_without_compute_policy_change():
    from foldjax.models.esmfold2.models.model import ModelSettings

    original = ModelSettings(num_samples=1, num_recycles=7)
    assert not original.return_confidence_logits
    actual = tape._replay_settings(original)
    assert actual == replace(original, num_samples=5, return_confidence_logits=True)
    assert actual.trunk_dtype == "bfloat16"


def test_lm_observer_preserves_return_rng_and_restores_method():
    rng = np.random.default_rng(8)
    reference = np.random.default_rng(8)

    class Model:
        def _compute_lm_hidden_states(self):
            return rng.normal(size=8)

    model = Model()
    seen = []
    with tape._observe_native_lm(model, lambda value: seen.append(value)) as calls:
        result = model._compute_lm_hidden_states()
    assert seen[0] is result
    assert len(calls) == 1
    assert "_compute_lm_hidden_states" not in model.__dict__
    np.testing.assert_array_equal(result, reference.normal(size=8))
    np.testing.assert_array_equal(rng.normal(size=8), reference.normal(size=8))


@pytest.mark.parametrize("interchange", [False, True])
def test_replay_lm_arm_skips_independent_compute_only_when_explicit(
    tmp_path, interchange
):
    from types import SimpleNamespace

    import jax.numpy as jnp

    native = jnp.asarray(np.arange(24).reshape(1, 2, 3, 4) / 7, jnp.bfloat16)
    schema = tape._save_lm(tmp_path, "upstream", native, native=False)
    capture = dict(lm_schema=schema, config=dict(lm_num_layers=2, lm_d_model=4))
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return object()

    def compute(features, loaded):
        calls.append("compute")
        return native + 1

    api = SimpleNamespace(load=load, language_model_states=compute)
    args = Namespace(
        native_lm=interchange,
        weights=tmp_path,
        tape=tmp_path / "tape.npz",
        output_dir=tmp_path,
    )
    _, value, _ = tape._load_replay_lm(
        args, api, {"token_attention_mask": np.ones((1, 2), bool)}, capture
    )
    assert calls[0]["language_model"] is not interchange
    assert ("compute" in calls) is not interchange
    np.testing.assert_array_equal(value, native if interchange else native + 1)
    assert args.lm_schema["original_dtype"] == "bfloat16"
    assert args.lm_schema["storage_dtype"] == "float32"


def test_lm_archive_rejects_lossy_dtype_claim(tmp_path):
    values = np.array([[[[1.001]]]], np.float32)
    schema = tape._save_lm(tmp_path, "upstream", values, native=False)
    schema["original_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="lossless"):
        tape._read_lm(tmp_path, schema, values.shape)


def _policy_torch():
    from types import SimpleNamespace

    calls = []
    cuda = SimpleNamespace(
        matmul=SimpleNamespace(
            allow_tf32=False,
            allow_bf16_reduced_precision_reduction=True,
            allow_fp16_reduced_precision_reduction=True,
        ),
        flash_sdp_enabled=lambda: True,
        mem_efficient_sdp_enabled=lambda: True,
        math_sdp_enabled=lambda: True,
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False),
        backends=SimpleNamespace(
            cuda=cuda,
            cudnn=SimpleNamespace(
                allow_tf32=True, benchmark=False, deterministic=False
            ),
        ),
        use_deterministic_algorithms=lambda *a, **kw: calls.append((a, kw)),
        are_deterministic_algorithms_enabled=lambda: bool(calls),
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        get_float32_matmul_precision=lambda: "highest",
    )
    return torch, calls


def test_native_deterministic_control_is_explicit_and_plain_is_unmodified(monkeypatch):
    for name in (
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    torch, calls = _policy_torch()
    plain = tape._native_policy(torch)
    assert not calls and not plain["deterministic_requested"]
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        tape._native_policy(torch, deterministic=True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    control = tape._native_policy(torch, deterministic=True)
    assert calls == [((True,), {"warn_only": False})]
    assert control["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert control["deterministic_algorithms"]


@pytest.mark.parametrize("bad", ["tf32", "late_cuda"])
def test_native_deterministic_control_rejects_ambiguous_environment(monkeypatch, bad):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    torch, calls = _policy_torch()
    if bad == "tf32":
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    else:
        torch.cuda.is_initialized = lambda: True
    with pytest.raises(ValueError):
        tape._native_policy(torch, deterministic=True)
    assert not calls
