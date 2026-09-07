import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.boltz_closure_capture import (
    NativeObserver,
    conditioning_tree,
    save_tree,
    writer_order,
)


class Tensor:
    def __init__(self, values, dtype="torch.float32"):
        self.values = np.asarray(values, dtype=np.float32)
        self.dtype = dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return Tensor(self.values)

    def numpy(self):
        if self.dtype == "torch.bfloat16":
            raise TypeError("mock native BF16 cannot be directly exported to NumPy")
        return self.values


def test_save_tree_retains_full_nested_output_and_original_bf16_dtype(tmp_path):
    value = {
        "head": Tensor([1.0078125, -0.0], "torch.bfloat16"),
        "pair_chains_iptm": {0: {1: np.array([0.1, 0.2])}},
        "exception": False,
        "optional": None,
    }
    record = save_tree(tmp_path, "forward-output", value)
    with np.load(tmp_path / "forward-output.npz", allow_pickle=False) as arrays:
        assert set(arrays) == {"head", "pair_chains_iptm.0.1", "exception"}
        assert arrays["head"].tobytes() == value["head"].values.tobytes()
    metadata = json.loads((tmp_path / "forward-output.tree.json").read_text())
    assert metadata["head"] == {
        "native_dtype": "torch.bfloat16",
        "storage_dtype": "float32",
        "shape": [2],
    }
    assert metadata["optional"] == {"kind": "none"}
    assert len(record["arrays_sha256"]) == len(record["tree_sha256"]) == 64
    before = (tmp_path / "forward-output.npz").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        save_tree(tmp_path, "forward-output", {"replacement": np.ones(2)})
    assert (tmp_path / "forward-output.npz").read_bytes() == before


def test_writer_order_uses_supplied_native_tie_order_not_a_new_sort():
    scores = np.array([2, 3, 1, 2, 3], np.float32)
    report = writer_order(scores, np.array([4, 1, 3, 0, 2]))
    assert report["rank_to_sample_index"] == [4, 1, 3, 0, 2]
    assert report["sample_index_to_rank"] == [3, 1, 4, 2, 0]
    assert report["comparison_order"] == "original sample index, never confidence rank"


@pytest.mark.parametrize(
    "scores,order",
    [
        ([0.1, 0.2], [0, 1]),
        ([0.1, 0.2], [1, 1]),
        ([0.1, np.nan], [1, 0]),
        ([0.1, 0.2], [1.0, 0.0]),
        ([0.1, 0.2], [1]),
    ],
)
def test_writer_order_rejects_invalid_or_unobserved_mapping(scores, order):
    with pytest.raises(ValueError):
        writer_order(np.asarray(scores), np.asarray(order))


class Module:
    def __init__(self, fn):
        self.fn = fn
        self.hooks = []

    def register_forward_hook(self, hook, *, with_kwargs):
        assert with_kwargs
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def __call__(self, *args, **kwargs):
        result = self.fn(*args, **kwargs)
        for hook in list(self.hooks):
            assert hook(self, args, kwargs, result) is None
        return result


def single_to_keys(*args, **kwargs):
    raise AssertionError("capture must not call the native callback")


single_to_keys.__module__ = "boltz.model.modules.encodersv2"


def _conditioning(s, z):
    callback = partial(single_to_keys, indexing_matrix=np.eye(2), W=32, H=128)
    return s, z, callback, s, z, z


class Model:
    def __init__(self):
        self.predict_args = {"recycling_steps": 3, "diffusion_samples": 5}
        self.steering_args = {"fk_steering": False}
        self.use_kernels = self.confidence_prediction = self.run_trunk_and_structure = (
            True
        )
        self.use_templates = self.bond_type_feature = self.skip_run_structure = False
        self.is_pairformer_compiled = self.is_msa_compiled = self.training = False
        for name in (
            "input_embedder",
            "s_init",
            "z_init_1",
            "z_init_2",
            "rel_pos",
            "token_bonds",
            "contact_conditioning",
            "s_norm",
            "z_norm",
            "s_recycle",
            "z_recycle",
        ):
            setattr(self, name, Module(lambda value: value + 1))
        self.msa_module = Module(lambda z, *args: z + 0.25)
        self.pairformer_module = Module(lambda s, z: (s + 0.5, z + 0.5))
        self.diffusion_conditioning = Module(_conditioning)

    def forward(self, feats):
        s_inputs = self.input_embedder(feats)
        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs) + self.z_init_2(s_inputs)
        z_init = z_init + self.rel_pos(feats) + self.token_bonds(feats)
        z_init = z_init + self.contact_conditioning(feats)
        s, z = np.zeros_like(s_init), np.zeros_like(z_init)
        for i in range(4):
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))
            z = z + self.msa_module(z, s_inputs, feats)
            s, z = self.pairformer_module(s, z)
        self.diffusion_conditioning(s, z)
        return {
            "s": s,
            "z": z,
            "sample_atom_coords": np.zeros((5, 2, 3)),
            "score": np.array([2, 3, 1, 2, 3], np.float32),
        }

    def predict_step(self, batch):
        result = self.forward(batch)
        return {
            "exception": False,
            "coords": result["sample_atom_coords"],
            "confidence_score": result["score"],
            "nested": {"s": result["s"]},
        }


def test_observer_preserves_native_calls_outputs_and_first_last_boundaries(tmp_path):
    sort_calls = []

    def argsort(scores, *, descending):
        sort_calls.append((scores, descending))
        return np.array([4, 1, 3, 0, 2])

    torch = SimpleNamespace(
        argsort=argsort,
        get_float32_matmul_precision=lambda: "highest",
        is_autocast_enabled=lambda device: True,
        get_autocast_dtype=lambda device: "torch.bfloat16",
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
    )

    class Writer:
        def write_on_batch_end(self, prediction):
            self.observed = prediction
            self.ranks = torch.argsort(prediction["confidence_score"], descending=True)

    original_forward = Model.forward
    observer = NativeObserver(
        tmp_path, samples=5, recycles=3, forward_code=original_forward.__code__
    )
    close = observer.install(Model, Writer, torch)
    model, writer = Model(), Writer()
    try:
        result = model.predict_step(np.ones(2))
        writer.write_on_batch_end(result)
        observer.validate()
        assert writer.observed is result
        assert len(sort_calls) == 1
        assert sort_calls[0][0] is result["confidence_score"]
        assert all(not getattr(model, name).hooks for name in observer.module_names)
    finally:
        close()
    assert Model.forward is original_forward
    assert torch.argsort is argsort
    assert (tmp_path / "forward-output.npz").is_file()
    assert (tmp_path / "predict-step-output.npz").is_file()
    assert (tmp_path / "trunk-boundaries/diffusion_conditioning.npz").is_file()
    with np.load(tmp_path / "trunk-boundaries/diffusion_conditioning.npz") as values:
        np.testing.assert_array_equal(
            values["to_keys.keywords.indexing_matrix"], np.eye(2)
        )
        assert values["to_keys.function"].item().endswith(".single_to_keys")
    assert (tmp_path / "trunk-boundaries/cycle-00/msa_module.npz").is_file()
    assert (tmp_path / "trunk-boundaries/cycle-03/pairformer_module.npz").is_file()
    assert not (tmp_path / "trunk-boundaries/cycle-01").exists()
    with np.load(tmp_path / "trunk-boundaries/initial.npz") as values:
        np.testing.assert_array_equal(values["s_init"], [3, 3])
        np.testing.assert_array_equal(values["z_init"], [12, 12])
    mapping = json.loads((tmp_path / "sample-rank-map.json").read_text())
    assert mapping["rank_to_sample_index"] == [4, 1, 3, 0, 2]
    observer.counts["writer_argsort"] = 0
    with pytest.raises(RuntimeError, match="incomplete native observation"):
        observer.validate()


def test_unknown_output_leaf_is_not_silently_dropped(tmp_path):
    with pytest.raises(TypeError, match="unsupported object"):
        save_tree(tmp_path, "bad-output", {"raw": object()})


def test_conditioning_rejects_unknown_callbacks_without_executing_them():
    with pytest.raises(ValueError, match="unmapped native"):
        conditioning_tree((None, None, partial(lambda: None), None, None, None))


def test_main_refuses_existing_output_before_importing_native_runtime(tmp_path):
    from bench import boltz_closure_capture as capture

    cache, out = tmp_path / "cache", tmp_path / "existing-output"
    (cache / "mols").mkdir(parents=True)
    (cache / "boltz2_conf.ckpt").write_bytes(b"mock checkpoint never loaded")
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("existing evidence")
    job = tmp_path / "input.yaml"
    job.write_text("version: 1\nsequences: []\n")
    with pytest.raises(FileExistsError):
        capture.main(
            [
                "--source-root",
                str(Path(capture.__file__).resolve().parents[1]),
                "--upstream-root",
                str(tmp_path),
                "--input",
                str(job),
                "--out-dir",
                str(out),
                "--cache",
                str(cache),
            ]
        )
    assert marker.read_text() == "existing evidence"
