from collections import OrderedDict

import numpy as np
import pytest

from bench.openbind_tape_adapter import (
    UPSTREAM_COMMIT,
    capture_provenance,
    parse_forward_tape,
    required_triangle_kernel,
    validate_native_config,
)


def capture(mask=None):
    if mask is None:
        mask = np.ones((1, 3, 2), np.float32)
    draws = OrderedDict()

    def add(name, value):
        draws[f"{len(draws):06d}_{name}_torch_{value.dtype}"] = value

    for _ in range(2):
        add("randint", np.array([2], np.int64))
        add("randperm", np.array([2, 0, 1], np.int64))
    add("randn", np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3))
    for step in range(2):
        add("randn", np.full((1, 2, 4), step + 1, np.float32))
        add("randn", np.full((1, 2, 3), step + 3, np.float32))
        add("randn_like", np.full((1, 2, 3, 3), step + 5, np.float32))
    return draws, dict(
        msa_mask=mask, n_atom=3, samples=2, steps=2, cycles=2, msa_depth=2
    )


def test_preserves_draw_order_dtype_and_sample_axis():
    draws, settings = capture()
    tape = parse_forward_tape(draws, **settings)
    np.testing.assert_array_equal(tape.msa_indices, [[2, 0], [2, 0]])
    assert tape.noise.shape == (3, 2, 3, 3)
    np.testing.assert_array_equal(tape.noise[0], draws["000004_randn_torch_float32"][0])
    np.testing.assert_array_equal(tape.quaternions[:, 0, 0], [1, 2])
    np.testing.assert_array_equal(tape.translations[:, 0, 0], [3, 4])
    assert tape.noise.dtype == np.float32
    assert tape.augmentation().quaternions.shape == (2, 2, 4)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "reorder",
        "extra",
        "shape",
        "dtype",
        "nonfinite",
        "permutation",
        "depth",
        "zero_q",
    ],
)
def test_rejects_invalid_or_unconsumed_draws(kind):
    draws, settings = capture()
    keys = list(draws)
    if kind == "missing":
        del draws[keys[-1]]
    elif kind == "reorder":
        draws.move_to_end(keys[0])
    elif kind == "extra":
        draws["unexpected"] = np.zeros(1)
    elif kind == "shape":
        draws[keys[4]] = draws[keys[4]][0]
    elif kind == "dtype":
        draws[keys[4]] = draws[keys[4]].astype(np.float64)
    elif kind == "nonfinite":
        draws[keys[4]][0, 0, 0, 0] = np.nan
    elif kind == "permutation":
        draws[keys[1]][0] = 0
    elif kind == "depth":
        draws[keys[0]][0] = 1
    elif kind == "zero_q":
        draws[keys[5]][:] = 0
    with pytest.raises(ValueError):
        parse_forward_tape(draws, **settings)


def test_invalid_msa_rows_map_back_to_original_indices():
    mask = np.array([[[1, 0], [0, 0], [0, 0]]], np.float32)
    draws, settings = capture(mask)
    for key in list(draws)[:4:2]:
        permutation_key = list(draws)[list(draws).index(key) + 1]
        draws[permutation_key] = np.array([1, 0], np.int64)
    tape = parse_forward_tape(draws, **settings)
    np.testing.assert_array_equal(tape.msa_indices, [[0, 2], [0, 2]])


def test_prepare_features_uses_selected_rows_not_new_rng():
    draws, settings = capture()
    tape = parse_forward_tape(draws, **settings)
    features = {"msa_mask": settings["msa_mask"], "msa": np.arange(6).reshape(1, 3, 2)}
    prepared = tape.prepare_features(features)
    from foldjax.models.openfold3.data.featurize import _MSA_CYCLE_INDICES

    restored = prepared["msa"][:, prepared[_MSA_CYCLE_INDICES][0]]
    np.testing.assert_array_equal(restored, features["msa"][:, [2, 0]])


def test_native_effective_config_must_match_logged_constructor():
    full = {
        "architecture": {
            "shared": {"num_recycles": 3},
            "msa": {
                "msa_module_embedder": {
                    "subsample_main_msa": False,
                    "subsample_all_msa": True,
                    "min_subsampled_all_msa": 1024,
                    "max_subsampled_all_msa": 1024,
                }
            },
        },
        "settings": {"native": True},
    }
    effective = {"shared": {"num_recycles": 3}, "settings": {"native": True}}
    assert validate_native_config(full, effective) is full["architecture"]
    with pytest.raises(ValueError, match="shared"):
        validate_native_config(full, {**effective, "shared": {"num_recycles": 4}})
    with pytest.raises(ValueError, match="settings"):
        validate_native_config(full, {**effective, "settings": {}})
    full["architecture"]["msa"]["msa_module_embedder"]["min_subsampled_all_msa"] = 512
    with pytest.raises(ValueError, match="fixed-depth"):
        validate_native_config(full, effective)


def test_repeat_provenance_uses_its_own_recorded_checkpoint(tmp_path):
    import json

    from bench.boltz_historical_replay import digest

    checkpoint = tmp_path / "native.pt"
    checkpoint.write_bytes(b"actual checkpoint")
    (tmp_path / "trace.json").write_text(
        json.dumps({"native_source": {"commit": UPSTREAM_COMMIT}, "chunks": []})
    )
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions/experiment_config.json").write_text(
        json.dumps({"inference_ckpt_path": str(checkpoint)})
    )
    assert capture_provenance(tmp_path, digest(checkpoint))["kind"].startswith("repeat")
    with pytest.raises(ValueError, match="checkpoint"):
        capture_provenance(tmp_path, "wrong")


def test_native_triton_is_never_silently_resolved_to_cueq():
    memory = dict(
        use_deepspeed_evo_attention=False,
        use_lma=False,
        use_triton_triangle_kernels=True,
        use_cueq_triangle_kernels=False,
    )
    effective = {"settings": {"memory": {"eval": memory}}}
    assert required_triangle_kernel(effective) == "triton"
    memory["use_cueq_triangle_kernels"] = True
    with pytest.raises(ValueError, match="ambiguous"):
        required_triangle_kernel(effective)
