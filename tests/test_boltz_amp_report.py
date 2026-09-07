import json

import numpy as np
import pytest

from bench.af3_closure_capture import sha
from bench.boltz_amp_report import (
    _STAGES,
    build_report,
    compare_arrays,
    compare_coordinates,
    entity_inputs,
    main,
)
from bench.boltz_foldjax_capture import array_identity


def _write_json(path, value):
    path.write_text(json.dumps(value))


def _write_bundle(root, label, values, *, native=True):
    path = root / f"{label}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    metadata = {
        name: {
            "shape": list(value.shape),
            **(
                {
                    "native_dtype": f"torch.{value.dtype}",
                    "storage_dtype": str(value.dtype),
                }
                if native
                else {"dtype": str(value.dtype)}
            ),
        }
        for name, value in values.items()
    }
    _write_json(root / f"{label}.tree.json", metadata)
    return {"arrays_sha256": sha(path), "tree_sha256": sha(root / f"{label}.tree.json")}


def _features():
    return {
        "atom_pad_mask": np.ones((1, 4), np.float32),
        "token_pad_mask": np.ones((1, 2), np.float32),
        "atom_to_token": np.array([[[1, 0], [1, 0], [1, 0], [0, 1]]], np.int64),
        "entity_id": np.array([[0, 1]], np.int64),
        "mol_type": np.array([[0, 3]], np.int64),
        "asym_id": np.array([[0, 1]], np.int64),
    }


def _coordinates():
    return np.broadcast_to(
        np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], np.float32), (5, 4, 3)
    ).copy()


def _native(root):
    root.mkdir()
    features = _features()
    np.savez(root / "features.npz", **features)
    np.savez(root / "tape.npz", init_noise=np.ones((5, 4, 3), np.float32))
    _write_json(root / "tape.json", {"num_samples": 5})
    _write_json(root / "effective-model-settings.json", {"dtype": "bf16"})
    provenance = {
        name: "a" * 64
        for name in (
            "input_sha256",
            "checkpoint_sha256",
            "wrapper_sha256",
            "legacy_capture_sha256",
        )
    }
    provenance.update(
        {
            "upstream_python_source": {"src/model.py": "b" * 64},
            "helper_source_sha256": {"bench/helper.py": "c" * 64},
            "torch_version": "native-test",
            "cuda_version": "test",
        }
    )
    _write_json(root / "provenance.json", provenance)
    raw = {
        "s": np.ones((1, 2, 3), np.float32),
        "z": np.ones((1, 2, 2, 3), np.float32),
        "sample_atom_coords": _coordinates(),
        "plddt": np.ones((5, 2), np.float32),
        "plddt_logits": np.ones((5, 2, 4), np.float32),
        "pair_chains_iptm.0.1": np.ones(5, np.float32),
    }
    public = {
        "plddt": raw["plddt"],
        "pair_chains_iptm.0.1": raw["pair_chains_iptm.0.1"],
        "coords": raw["sample_atom_coords"],
        "confidence_score": np.ones(5, np.float32),
    }
    artifacts = {
        "forward-output": _write_bundle(root, "forward-output", raw),
        "predict-step-output": _write_bundle(root, "predict-step-output", public),
        "preprocessing-tape": _write_bundle(
            root, "preprocessing-tape", {"draw": np.ones(3)}
        ),
    }
    for stage in _STAGES:
        label = f"trunk-boundaries/{stage}"
        artifacts[label] = _write_bundle(root, label, {"": np.ones(2, np.float32)})
    _write_json(
        root / "capture-complete.json", {"passed": True, "artifacts": artifacts}
    )
    return raw


def _foldjax(root, native, raw, *, trunk_only=False):
    root.mkdir()
    names = (
        "capture-complete.json",
        "provenance.json",
        "features.npz",
        "tape.npz",
        "tape.json",
        "effective-model-settings.json",
        "forward-output.tree.json",
    )
    _write_json(
        root / "provenance.json",
        {
            "arm": "foldjax-core-only",
            "source_files": {"src/foldjax/model.py": "d" * 64},
            "native_artifacts": {name: sha(native / name) for name in names},
        },
    )
    features = {
        name: value.astype(np.int32) if value.dtype == np.int64 else value
        for name, value in _features().items()
    }
    _write_json(
        root / "effective-options.json",
        {
            "features": {
                name: array_identity(value) for name, value in features.items()
            },
            "sampler_tape": {
                "init_noise": array_identity(np.ones((5, 4, 3), np.float32))
            },
        },
    )
    output = {"single": raw["s"], "pair": raw["z"]}
    if not trunk_only:
        output.update(
            {name: value for name, value in raw.items() if name not in {"s", "z"}}
        )
    binding = _write_bundle(root, "prediction", output, native=False)
    stages = {}
    for stage in _STAGES:
        label = f"trunk-boundaries/{stage}"
        stages[label] = _write_bundle(
            root, label, {"": np.ones(2, np.float32)}, native=False
        )
    _write_json(
        root / "capture-complete.json",
        {
            "passed": True,
            "trunk_only": trunk_only,
            "stage_artifacts": stages,
            "prediction_sha256": binding["arrays_sha256"],
            "prediction_tree_sha256": binding["tree_sha256"],
        },
    )


def test_full_report_keeps_sample_order_missing_public_fields_and_portable_ids(
    tmp_path,
):
    native, candidate = tmp_path / "native", tmp_path / "candidate"
    raw = _native(native)
    _foldjax(candidate, native, raw)
    report = build_report(native, candidate)
    assert report["coordinates"]["coordinate_diagnostic"]["coordinate_gate_passed"]
    assert report["raw_confidence"]["strict_pass"]
    assert report["public_confidence"]["missing_from_candidate"] == ["confidence_score"]
    assert not report["public_confidence"]["strict_pass"]
    assert report["reference_control"]["sampler_entry_hashes_verified"]
    assert report["stage_comparisons"]["input_embedder"]["leaves"][""]["strict_pass"]
    encoded = json.dumps(report, allow_nan=False)
    assert str(tmp_path) not in encoded


def test_report_verifies_bound_compiler_options(tmp_path):
    native, candidate = tmp_path / "native", tmp_path / "candidate"
    raw = _native(native)
    _foldjax(candidate, native, raw)
    options_path = candidate / "effective-options.json"
    options = json.loads(options_path.read_text())
    options["compiler_options"] = {"xla_allow_excess_precision": False}
    _write_json(options_path, options)
    complete_path = candidate / "capture-complete.json"
    complete = json.loads(complete_path.read_text())
    complete["effective_options_sha256"] = sha(options_path)
    _write_json(complete_path, complete)
    report = build_report(native, candidate)
    assert report["candidate_compilation"]["effective_options_hash_bound"]
    assert report["candidate_compilation"]["compiler_options"] == {
        "xla_allow_excess_precision": False
    }
    options["compiler_options"]["xla_allow_excess_precision"] = True
    _write_json(options_path, options)
    with pytest.raises(ValueError, match="hash mismatch: effective-options.json"):
        build_report(native, candidate)


def test_trunk_only_does_not_infer_confidence_or_structure_pass(tmp_path):
    native, candidate = tmp_path / "native", tmp_path / "candidate"
    raw = _native(native)
    _foldjax(candidate, native, raw, trunk_only=True)
    report = build_report(native, candidate)
    assert report["trunk_only"]
    assert report["not_evaluated"] == [
        "coordinates",
        "raw confidence",
        "public confidence",
    ]
    assert "coordinates" not in report


def test_native_repeat_requires_exact_feature_sampler_and_preprocessing_bytes(tmp_path):
    native, repeat = tmp_path / "native", tmp_path / "repeat"
    _native(native)
    _native(repeat)
    report = build_report(native, repeat)
    assert report["reference_control"]["controlled_native_repeat"]
    assert report["public_confidence"]["strict_pass"]
    np.savez(repeat / "tape.npz", init_noise=np.zeros((5, 4, 3), np.float32))
    with pytest.raises(ValueError, match="uncontrolled native repeat"):
        build_report(native, repeat)


@pytest.mark.parametrize(
    "defect", ["reference", "output", "feature_entry", "tape_entry"]
)
def test_wrong_reference_or_mutated_capture_is_rejected(tmp_path, defect):
    native, candidate = tmp_path / "native", tmp_path / "candidate"
    raw = _native(native)
    _foldjax(candidate, native, raw)
    if defect == "output":
        np.savez(candidate / "prediction.npz", sample_atom_coords=_coordinates())
    elif defect == "reference":
        value = json.loads((candidate / "provenance.json").read_text())
        value["native_artifacts"]["features.npz"] = "0" * 64
        _write_json(candidate / "provenance.json", value)
    else:
        value = json.loads((candidate / "effective-options.json").read_text())
        value["features" if defect == "feature_entry" else "sampler_tape"] = {}
        _write_json(candidate / "effective-options.json", value)
    with pytest.raises(ValueError, match="hash"):
        build_report(native, candidate)


def test_one_global_fit_does_not_hide_ligand_motion_or_pool_chain_copies():
    features = _features()
    features["entity_id"][:] = 0
    features["mol_type"][:] = 0
    _, _, labels = entity_inputs(features)
    assert len(set(labels)) == 2
    left, right = _coordinates(), _coordinates()
    right[:, 3, 0] += 1
    report = compare_coordinates(left, right, features)
    assert not report["coordinate_diagnostic"]["coordinate_gate_passed"]
    assert len(report["entity_rmsd"]) == 2


@pytest.mark.parametrize(
    "defect", ["samples", "nan", "complex", "onehot", "mask", "padded_token"]
)
def test_coordinate_and_atom_mapping_boundaries_fail_closed(defect):
    features, left, right = _features(), _coordinates(), _coordinates()
    if defect == "samples":
        right = right[:4]
    elif defect == "nan":
        right[0, 0, 0] = np.nan
    elif defect == "complex":
        right = right.astype(complex)
    elif defect == "onehot":
        features["atom_to_token"][0, 0] = 1
    elif defect == "mask":
        features["atom_pad_mask"][0, 0] = 0.5
    else:
        features["token_pad_mask"][0, 1] = 0
    with pytest.raises(ValueError):
        compare_coordinates(left, right, features)


def test_float_undefined_positions_discrete_exactness_and_missing_nested_leaf():
    left = {
        "score": np.array([np.nan, np.inf, -np.inf, 1.0]),
        "pair_chains_iptm.0.1": np.ones(5),
        "count": np.array([2**63 + 1], np.uint64),
    }
    right = {name: value.copy() for name, value in left.items()}
    right["score"][3] += 0.001
    right["count"][0] -= np.uint64(1)
    del right["pair_chains_iptm.0.1"]
    report = compare_arrays(left, right)
    assert report["missing_from_candidate"] == ["pair_chains_iptm.0.1"]
    assert report["leaves"]["score"]["undefined_positions_equal"]
    assert report["leaves"]["score"]["max_abs"] == pytest.approx(0.001)
    assert not report["leaves"]["score"]["strict_pass"]
    assert not report["leaves"]["count"]["strict_pass"]
    right["score"][1] = -np.inf
    assert not compare_arrays(left, right)["leaves"]["score"][
        "undefined_positions_equal"
    ]


def test_original_dtype_is_separate_from_numeric_diagnostic():
    value = {"score": np.ones(5, np.float32)}
    report = compare_arrays(
        value,
        value,
        {"score": {"native_dtype": "torch.bfloat16"}},
        {"score": {"dtype": "float32"}},
    )
    assert report["strict_pass"]
    assert not report["original_dtypes_equal"]


def test_cli_refuses_report_overwrite_before_reading_captures(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        main(["--native", "missing", "--candidate", "missing", "--out", str(report)])
    assert report.read_text() == "existing evidence"
