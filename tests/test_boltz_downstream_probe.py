import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.af3_closure_capture import sha
from bench.boltz_downstream_probe import (
    _CALLBACK,
    bound_arrays,
    load_conditioning,
    load_trunk,
    main,
    make_sampler,
    restore_conditioning,
    sampler_options,
)
from foldjax.models.boltz2.models.diffusion.atom import (
    get_indexing_matrix,
    single_to_keys,
)
from tests.test_boltz_foldjax_capture import _effective, _meta


def _bound(root, label, arrays, original=None):
    path = root / f"{label}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    tree = {
        name: {
            "shape": list(value.shape),
            "storage_dtype": str(value.dtype),
            "native_dtype": (original or {}).get(name, str(value.dtype)),
        }
        for name, value in arrays.items()
    }
    path.with_suffix(".tree.json").write_text(json.dumps(tree))
    complete_path = root / "capture-complete.json"
    complete = (
        json.loads(complete_path.read_text())
        if complete_path.exists()
        else {"passed": True, "artifacts": {}}
    )
    complete["artifacts"][label] = {
        "arrays_sha256": sha(path),
        "tree_sha256": sha(path.with_suffix(".tree.json")),
    }
    complete_path.write_text(json.dumps(complete))


def _conditioning(root):
    meta = {"n_atom": 32, "n_token": 2}
    matrix = np.asarray(get_indexing_matrix(1, 32, 128))
    arrays = {
        "q": np.ones((1, 32, 128), np.float32),
        "c": np.ones((1, 32, 128), np.float32),
        "atom_enc_bias": np.ones((1, 1, 32, 128, 12), np.float32),
        "atom_dec_bias": np.ones((1, 1, 32, 128, 12), np.float32),
        "token_trans_bias": np.ones((1, 2, 2, 384), np.float32),
        **{key: np.asarray(value) for key, value in _CALLBACK.items()},
        "to_keys.function_source_sha256": np.asarray("a" * 64),
        "to_keys.keywords.indexing_matrix": matrix,
    }
    original = {
        key: "torch.float32" if key in {"q", "c"} else "torch.bfloat16"
        for key in ("q", "c", "atom_enc_bias", "atom_dec_bias", "token_trans_bias")
    }
    _bound(root, "trunk-boundaries/diffusion_conditioning", arrays, original)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "upstream_python_source": {
                    "src/boltz/model/modules/encodersv2.py": "a" * 64
                }
            }
        )
    )
    return meta, matrix, arrays, original


def test_native_conditioning_restores_original_dtype_and_uses_jax_mapper(tmp_path):
    meta, matrix, _, _ = _conditioning(tmp_path)
    payload, tree = load_conditioning(tmp_path, meta, native_indexing=matrix)
    restored = restore_conditioning(payload)
    assert set(restored) == {
        "q",
        "c",
        "atom_enc_bias",
        "atom_dec_bias",
        "token_trans_bias",
        "indexing_matrix",
    }
    assert restored["token_trans_bias"].dtype == jnp.bfloat16
    assert (
        restored["atom_enc_bias"].dtype
        == restored["atom_dec_bias"].dtype
        == jnp.bfloat16
    )
    assert restored["q"].dtype == restored["c"].dtype == jnp.float32
    assert tree["token_trans_bias"]["native_dtype"] == "torch.bfloat16"
    for name in restored:
        np.testing.assert_array_equal(
            np.asarray(restored[name], np.float32), payload[name]
        )


@pytest.mark.parametrize(
    "defect",
    ["missing", "callback", "source", "dtype", "unrounded", "nan", "shape", "matrix"],
)
def test_conditioning_rejects_unproven_or_malformed_injection(tmp_path, defect):
    meta, matrix, arrays, original = _conditioning(tmp_path)
    if defect == "missing":
        arrays.pop("token_trans_bias")
    elif defect == "callback":
        arrays["to_keys.function"] = np.asarray("unknown.callback")
    elif defect == "source":
        arrays["to_keys.function_source_sha256"] = np.asarray("b" * 64)
    elif defect == "dtype":
        original["atom_enc_bias"] = "torch.float32"
    elif defect == "unrounded":
        arrays["atom_enc_bias"].flat[0] = 1.003
    elif defect == "nan":
        arrays["q"].flat[0] = np.nan
    elif defect == "shape":
        arrays["q"] = arrays["q"][:, :31]
    elif defect == "matrix":
        arrays["to_keys.keywords.indexing_matrix"] = np.zeros_like(matrix)
    _bound(tmp_path, "trunk-boundaries/diffusion_conditioning", arrays, original)
    with pytest.raises(ValueError):
        load_conditioning(tmp_path, meta, native_indexing=matrix)


def test_bound_archive_detects_changes_and_metadata_lies(tmp_path):
    _bound(tmp_path, "sample", {"value": np.ones(2, np.float32)})
    path = tmp_path / "sample.npz"
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        bound_arrays(tmp_path, "sample")
    _bound(tmp_path, "sample", {"value": np.ones(2, np.float32)})
    tree_path = path.with_suffix(".tree.json")
    tree = json.loads(tree_path.read_text())
    tree["value"]["shape"] = [3]
    tree_path.write_text(json.dumps(tree))
    complete = json.loads((tmp_path / "capture-complete.json").read_text())
    complete["artifacts"]["sample"]["tree_sha256"] = sha(tree_path)
    (tmp_path / "capture-complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="metadata mismatch"):
        bound_arrays(tmp_path, "sample")


def test_legacy_trunk_requires_all_bound_native_values(tmp_path):
    arrays = {
        "s": np.ones((1, 2, 384), np.float32),
        "s_inputs": np.ones((1, 2, 384), np.float32),
        "z": np.ones((1, 2, 2, 128), np.float32),
        "relative_position_encoding": np.ones((1, 2, 2, 128), np.float32),
    }
    _bound(tmp_path, "forward-output", {key: arrays[key] for key in ("s", "z")})
    _bound(tmp_path, "trunk-boundaries/input_embedder", {"": arrays["s_inputs"]})
    _bound(
        tmp_path, "trunk-boundaries/rel_pos", {"": arrays["relative_position_encoding"]}
    )
    np.savez(tmp_path / "trunk.npz", **arrays)
    actual, _ = load_trunk(tmp_path, {"n_token": 2})
    assert set(actual) == set(arrays)
    arrays["z"] = arrays["z"] + 1
    np.savez(tmp_path / "trunk.npz", **arrays)
    with pytest.raises(ValueError, match="differs from bound native"):
        load_trunk(tmp_path, {"n_token": 2})
    arrays.pop("relative_position_encoding")
    np.savez(tmp_path / "trunk.npz", **arrays)
    with pytest.raises(ValueError, match="missing or has unknown"):
        load_trunk(tmp_path, {"n_token": 2})


@pytest.mark.parametrize("inject", [False, True])
def test_runtime_inputs_remain_dynamic_and_patches_restore(inject):
    calls = []
    module = SimpleNamespace(
        _sample_schedule=lambda *a, **kw: pytest.fail("schedule was not replaced"),
        boltz2_trunk_forward=lambda *a, **kw: pytest.fail("trunk must not execute"),
        diffusion_conditioning_forward=lambda *a, **kw: {"q": kw["feats"]["q"]},
    )
    originals = vars(module).copy()

    def sample(params, feats, key, **kw):
        calls.append(True)
        condition = module.diffusion_conditioning_forward(feats=feats)
        result = kw["init_noise"] + kw["step_noises"][0] + module._sample_schedule(2)[0]
        result = result + kw["trunk"]["s"].sum() + condition["q"].sum()
        result = result + kw["aug_transforms"][0].sum() + kw["aug_transforms"][1].sum()
        if inject:
            assert condition["token_trans_bias"].dtype == jnp.bfloat16
            assert "token_trans_bias_params" not in condition
            result = result + condition["to_keys"](feats["window"]).sum()
        return {"sample_atom_coords": result}

    module.boltz2_sample_forward = sample
    tape = {
        "sigmas": jnp.asarray([3, 1, 0], jnp.float32),
        "init_noise": jnp.zeros((5, 2, 3)),
        "step_noises": jnp.zeros((2, 5, 2, 3)),
        "rotations": jnp.zeros((2, 5, 3, 3)),
        "translations": jnp.zeros((2, 5, 1, 3)),
    }
    features = {"q": jnp.ones(1), "window": jnp.ones((1, 32, 1))}
    conditioning = {
        key: jnp.ones(1, jnp.bfloat16 if key.endswith("bias") else jnp.float32)
        for key in ("q", "c", "atom_enc_bias", "atom_dec_bias", "token_trans_bias")
    }
    conditioning["indexing_matrix"] = get_indexing_matrix(1, 32, 128)
    run = jax.jit(
        make_sampler(
            module, {}, native_conditioning=inject, single_to_keys=single_to_keys
        )
    )
    first = run(
        {}, features, jax.random.PRNGKey(0), tape, {"s": jnp.ones(1)}, conditioning
    )
    second = run(
        {},
        features,
        jax.random.PRNGKey(0),
        {**tape, "sigmas": tape["sigmas"] + 2},
        {"s": jnp.ones(1) * 4},
        {**conditioning, "q": conditioning["q"] + 5},
    )
    np.testing.assert_array_equal(
        second["sample_atom_coords"] - first["sample_atom_coords"],
        np.full((5, 2, 3), 10 if inject else 5),
    )
    assert len(calls) == 1
    assert all(getattr(module, key) is value for key, value in originals.items())


def test_sampler_preserves_n5_native_policy_without_confidence_options():
    options = sampler_options(_meta(), _effective())
    assert (
        options["multiplicity"],
        options["num_sampling_steps"],
        options["recycling_steps"],
    ) == (5, 200, 3)
    assert options["use_scan"] and options["score_use_scan"]
    assert options["compute_dtype"] == "bfloat16"
    assert options["steering_args"]["contact_guidance_update"]
    assert not any("confidence" in key for key in options)


def test_materialized_control_changes_only_token_bias_execution_mode():
    lazy = sampler_options(_meta(), _effective())
    materialized = sampler_options(_meta(), _effective(), materialized_token_bias=True)
    assert lazy["lazy_token_trans_bias"] is True
    assert materialized == {**lazy, "lazy_token_trans_bias": False}


def test_existing_output_is_refused_before_loading_inputs(tmp_path):
    with pytest.raises(FileExistsError):
        main(
            [
                "--source-root",
                "missing",
                "--upstream-capture",
                "missing",
                "--weights",
                "missing",
                "--out-dir",
                str(tmp_path),
            ]
        )
