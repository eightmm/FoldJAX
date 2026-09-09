from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.protenix_closure_capture import (
    NativeRecorder,
    code_child,
    flatten_native,
    host_array,
    input_assets,
    preflight_configs,
)


class Config(SimpleNamespace):
    def __getitem__(self, name):
        return getattr(self, name)


def _config(tmp_path):
    paths = {}
    for name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
        "obsolete_release_data_csv",
    ):
        path = tmp_path / name
        path.write_bytes(b"local fixture")
        paths[name] = str(path)
    (tmp_path / "protenix_base_default_v1.0.0.pt").write_bytes(b"checkpoint")
    return Config(
        model_name="protenix_base_default_v1.0.0",
        sample_diffusion=Config(N_sample=5, N_step=200, guidance=Config(enable=False)),
        model=Config(N_cycle=10, N_model_seed=1),
        seeds=[101],
        use_seeds_in_json=False,
        dtype="bf16",
        use_template=False,
        esm=Config(enable=False),
        num_workers=0,
        data=Config(**paths),
        load_checkpoint_dir=str(tmp_path),
    )


def _complete_recorder(tmp_path):
    recorder = NativeRecorder(tmp_path, samples=2, steps=3, cycles=2)
    for name in (
        "native-input",
        "native-identity",
        "native-derived",
        "input-embedding",
        "trunk",
        "distogram",
        "confidence-input",
        "prediction",
    ):
        recorder.bundle(name, {"array": np.ones(1, dtype=np.float32)})
    recorder.completed = 1
    recorder.mc_dropout = False
    recorder.random_decisions = [0.6]
    recorder.schedule = np.arange(4, dtype=np.float32)
    recorder.draw("init", 0, np.ones((2, 4, 3), np.float32))
    for i in range(3):
        recorder.draw("churn", i, np.full((2, 4, 3), i, np.float32))
        recorder.draw("rotation", i, np.ones((2, 3, 3), np.float32))
        recorder.draw("translation", i, np.ones((2, 1, 3), np.float32))
    recorder.msa = [{"rows": np.arange(3)}, {"rows": np.arange(3)}]
    return recorder


def test_native_flatten_preserves_dtypes_shapes_nulls_and_signed_zero():
    value = {"a": np.array([-0.0, 2.0], np.float32), "b": [None, "chainA"]}
    arrays, metadata = flatten_native(value)
    assert arrays["a"].tobytes() == value["a"].tobytes()
    assert metadata["a"] == {
        "native_dtype": "float32",
        "storage_dtype": "float32",
        "shape": [2],
    }
    assert metadata["b.0"] == {"kind": "none"}
    assert arrays["b.1"].item() == "chainA"
    value["a"][1] = 7
    assert arrays["a"][1] == 2  # native in-place updates cannot alter saved values


def test_native_flatten_rejects_objects_and_path_collisions():
    with pytest.raises(TypeError, match="object payload"):
        host_array(object())
    with pytest.raises(ValueError, match="duplicate flattened path"):
        flatten_native({"a.b": 1, "a": {"b": 2}})


def test_native_text_object_annotations_have_lossless_nonpickle_storage():
    value = np.asarray([np.str_("rna"), np.str_("ligand")], dtype=object)
    arrays, metadata = flatten_native({"chain_mol_type": value})
    np.testing.assert_array_equal(arrays["chain_mol_type"], ["rna", "ligand"])
    assert arrays["chain_mol_type"].dtype.kind == "U"
    assert metadata["chain_mol_type"]["native_dtype"] == "object"
    for unsupported in (["rna", None], ["rna", 1], [object()]):
        with pytest.raises(TypeError, match="object payload"):
            host_array(np.asarray(unsupported, dtype=object))


def test_bf16_storage_mapping_is_explicit():
    class Tensor:
        dtype = "torch.bfloat16"

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return np.array([1.0078125], np.float32)

    arrays, metadata = flatten_native({"bf16": Tensor()})
    assert arrays["bf16"].dtype == np.float32
    assert metadata["bf16"]["native_dtype"] == "torch.bfloat16"


def test_capture_completion_writes_full_index_ordered_tape(tmp_path):
    recorder = _complete_recorder(tmp_path)
    recorder.finish()
    report = json.loads((tmp_path / "capture-complete.json").read_text())
    assert report["passed"] is True
    assert report["mc_dropout_applied"] is False
    with np.load(tmp_path / "sampler-tape.npz", allow_pickle=False) as tape:
        assert tape["step_noises"].shape == (3, 2, 4, 3)
        np.testing.assert_array_equal(tape["step_noises"][:, 0, 0, 0], [0, 1, 2])
        assert tape["translations"].shape == (3, 2, 3)


@pytest.mark.parametrize("defect", ["missing", "extra", "dropout", "completion", "msa"])
def test_capture_fails_closed_on_missing_extra_or_unsupported_events(tmp_path, defect):
    recorder = _complete_recorder(tmp_path)
    if defect == "missing":
        recorder.draws["rotation"].pop(1)
    elif defect == "extra":
        recorder.draws["churn"][3] = recorder.draws["churn"][0]
    elif defect == "dropout":
        recorder.mc_dropout = True
    elif defect == "completion":
        recorder.completed = 0
    elif defect == "msa":
        recorder.msa.pop()
    with pytest.raises(ValueError):
        recorder.finish()
    assert not (tmp_path / "capture-complete.json").exists()


def test_capture_records_complete_native_dropout_masks(tmp_path):
    recorder = _complete_recorder(tmp_path)
    recorder.mc_dropout = True
    recorder.dropout_rate = 0.4
    recorder.dropout_masks = [np.full((2, 2, 3), i == 0, bool) for i in range(2)]
    recorder.finish()
    report = json.loads((tmp_path / "capture-complete.json").read_text())
    assert report["mc_dropout_applied"] is True
    assert report["mc_dropout_mask_calls"] == 2
    assert report["mc_dropout_rate"] == 0.4
    with np.load(tmp_path / "dropout-tape.npz") as tape:
        np.testing.assert_array_equal(tape["keep_masks"], recorder.dropout_masks)


@pytest.mark.parametrize("defect", ["count", "dtype", "shape", "rate", "branch"])
def test_capture_rejects_invalid_dropout_tape(tmp_path, defect):
    recorder = _complete_recorder(tmp_path)
    recorder.mc_dropout = True
    recorder.dropout_rate = 0.4
    recorder.dropout_masks = [np.ones((2, 2, 3), bool)] * 2
    if defect == "count":
        recorder.dropout_masks.pop()
    elif defect == "dtype":
        recorder.dropout_masks[0] = np.ones((2, 2, 3), np.float32)
    elif defect == "shape":
        recorder.dropout_masks[0] = np.ones((2, 3, 3), bool)
    elif defect == "rate":
        recorder.dropout_rate = float("nan")
    else:
        recorder.mc_dropout = False
    with pytest.raises(ValueError, match="dropout"):
        recorder.finish()
    assert not (tmp_path / "capture-complete.json").exists()


def test_capture_duplicate_nonfinite_and_dtype_draws_are_rejected(tmp_path):
    recorder = NativeRecorder(tmp_path)
    recorder.draw("init", 0, np.ones(1, np.float32))
    with pytest.raises(ValueError, match="duplicate"):
        recorder.draw("init", 0, np.ones(1, np.float32))
    for value in (np.array([np.nan], np.float32), np.ones(1, np.float64)):
        with pytest.raises(ValueError, match="non-FP32/nonfinite"):
            recorder.draw("churn", 0, value)


def test_actual_code_identity_not_shape_heuristics_selects_sampler_body():
    def sample():
        def _chunk_sample_diffusion():
            pass

        return _chunk_sample_diffusion

    assert code_child(sample, "_chunk_sample_diffusion") is sample().__code__
    with pytest.raises(ValueError, match="exactly one"):
        code_child(sample, "unrelated_function")


def test_local_asset_preflight_validates_defaults_and_never_downloads(tmp_path):
    configs = _config(tmp_path)
    result = preflight_configs(configs)
    assert set(result) == {
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
        "obsolete_release_data_csv",
        "checkpoint",
    }
    (tmp_path / "pdb_cluster_file").unlink()
    with pytest.raises(FileNotFoundError, match="pdb_cluster_file"):
        preflight_configs(configs)


@pytest.mark.parametrize("change", ["dtype", "samples", "template", "workers"])
def test_preflight_rejects_unrecorded_routes(tmp_path, change):
    configs = _config(tmp_path)
    if change == "dtype":
        configs.dtype = "fp32"
    elif change == "samples":
        configs.sample_diffusion.N_sample = 1
    elif change == "template":
        configs.use_template = True
    else:
        configs.num_workers = 1
    with pytest.raises(ValueError):
        preflight_configs(configs)


def test_input_path_assets_are_hashed_and_missing_paths_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "alignment.a3m").write_text(">query\nAAA\n")
    records = input_assets({"chains": [{"unpairedMsaPath": "alignment.a3m"}]})
    assert len(records["input.chains.0.unpairedMsaPath"]) == 1
    with pytest.raises(FileNotFoundError):
        input_assets({"unpairedMsaPath": "missing.a3m"})
