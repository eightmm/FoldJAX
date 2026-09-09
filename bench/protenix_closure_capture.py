"""Observe a pinned publisher Protenix runner without changing its math policy.

Native-only finite n=5/200-step/10-cycle capture. All heavy publisher imports
are inside ``main`` so the recorder and fail-closed checks have CPU-only tests.
These observer runs are not uninstrumented performance measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import os
import random
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from types import CodeType
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import save, sha

MODELS = (
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix-v2",
)


def host_array(value):
    """Store BF16 values losslessly in FP32, retaining native dtype separately."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    array = np.asarray(value)
    # Biotite's chain_mol_type annotation is an object array of text. Retain
    # every string in pickle-free storage; arbitrary Python objects still fail.
    if array.dtype.kind == "O" and all(isinstance(x, str) for x in array.flat):
        array = array.astype(str)
    if array.dtype.kind == "O":
        raise TypeError(f"unsupported object payload: {type(value)}")
    return array.copy()


def flatten_native(value, prefix="", *, arrays=None, metadata=None):
    arrays = {} if arrays is None else arrays
    metadata = {} if metadata is None else metadata
    if isinstance(value, Mapping):
        children = value.items()
    elif isinstance(value, (tuple, list)):
        children = enumerate(value)
    else:
        if prefix in metadata:
            raise ValueError(f"duplicate flattened path {prefix}")
        if value is None:
            metadata[prefix] = {"kind": "none"}
        else:
            array = host_array(value)
            arrays[prefix] = array
            metadata[prefix] = {
                "native_dtype": str(getattr(value, "dtype", array.dtype)),
                "storage_dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        return arrays, metadata
    for name, child in children:
        flatten_native(
            child,
            f"{prefix}.{name}" if prefix else str(name),
            arrays=arrays,
            metadata=metadata,
        )
    return arrays, metadata


def code_child(function, name):
    candidates = [
        value
        for value in function.__code__.co_consts
        if isinstance(value, CodeType) and value.co_name == name
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one publisher code object: {name}")
    return candidates[0]


def source_identity(directory):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(directory), *args])

    diff = git("diff", "HEAD", "--binary")
    return {
        "resolved_path": str(directory.resolve()),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "changed_files": git("diff", "HEAD", "--name-only").decode().splitlines(),
        "status_sha256": hashlib.sha256(
            git("status", "--porcelain=v1", "--untracked-files=all")
        ).hexdigest(),
    }


def input_assets(value):
    """Native JSON opens paths relative to the execution cwd, not JSON parent."""
    records = {}

    def visit(child, path):
        if isinstance(child, Mapping):
            for name, item in child.items():
                label = f"{path}.{name}"
                if (
                    isinstance(item, str)
                    and item
                    and (name.endswith("Path") or name.endswith("_dir"))
                ):
                    asset = Path(item).resolve(strict=True)
                    files = [asset] if asset.is_file() else sorted(asset.rglob("*"))
                    records[label] = {
                        str(p if p == asset else p.relative_to(asset)): sha(p)
                        for p in files
                        if p.is_file()
                    }
                else:
                    visit(item, label)
        elif isinstance(child, list):
            for index, item in enumerate(child):
                visit(item, f"{path}.{index}")

    visit(value, "input")
    return records


def preflight_configs(configs):
    if configs.model_name not in MODELS:
        raise ValueError("this capture only audits the three non-ESM base profiles")
    if (
        configs.sample_diffusion.N_sample != 5
        or configs.sample_diffusion.N_step != 200
        or configs.model.N_cycle != 10
        or configs.model.N_model_seed != 1
        or list(configs.seeds) != [101]
        or configs.use_seeds_in_json
    ):
        raise ValueError("capture requires native n5/200 steps/10 cycles/seed101")
    if configs.dtype != "bf16":
        raise ValueError("publisher BF16 outer autocast was not selected")
    if (
        configs.use_template
        or configs.esm.enable
        or configs.sample_diffusion.guidance.enable
        or configs.num_workers != 0
    ):
        raise ValueError("template/ESM/guidance/worker routes are not captured")
    records = {}
    paths = {
        name: Path(configs.data[name])
        for name in (
            "ccd_components_file",
            "ccd_components_rdkit_mol_file",
            "pdb_cluster_file",
            "obsolete_release_data_csv",
        )
    }
    paths["checkpoint"] = Path(configs.load_checkpoint_dir) / f"{configs.model_name}.pt"
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required native local asset is missing: {name}")
        records[name] = {"path": str(path.resolve()), "sha256": sha(path)}
    return records


class NativeRecorder:
    def __init__(self, out, *, samples=5, steps=200, cycles=10):
        self.out = out
        self.samples, self.steps, self.cycles = samples, steps, cycles
        self.bundles = set()
        self.draws = {name: {} for name in ("init", "churn", "rotation", "translation")}
        self.msa = []
        self.msa_indices = None
        self.in_sampler = False
        self.in_msa = False
        self.schedule = None
        self.random_decisions = []
        self.mc_dropout = None
        self.dropout_masks = []
        self.dropout_rate = None
        self.completed = 0
        self.fp32_atom_aggregation = False

    def bundle(self, name, values):
        if name in self.bundles:
            raise ValueError(f"unexpected repeated native boundary: {name}")
        arrays, metadata = flatten_native(values)
        np.savez(self.out / f"{name}.npz", **arrays)
        save(self.out / f"{name}-tree.json", metadata)
        self.bundles.add(name)

    def draw(self, name, index, value):
        if index in self.draws[name]:
            raise ValueError(f"duplicate {name} event at index {index}")
        array = host_array(value)
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"non-FP32/nonfinite sampler draw: {name}")
        self.draws[name][index] = array

    def finish(self):
        required = {
            "native-input",
            "native-identity",
            "native-derived",
            "input-embedding",
            "trunk",
            "distogram",
            "confidence-input",
            "prediction",
        }
        if not required <= self.bundles or self.completed != 1:
            raise ValueError(
                "native runner did not capture exactly one completed prediction"
            )
        if type(self.mc_dropout) is not bool or len(self.random_decisions) != 1:
            raise ValueError(
                "MC dropout decision is missing or its mask tape is unsupported"
            )
        if self.mc_dropout:
            if (
                len(self.dropout_masks) != self.cycles
                or self.dropout_rate is None
                or not 0 < self.dropout_rate < 1
            ):
                raise ValueError("MC dropout mask tape is incomplete")
            shapes = {mask.shape for mask in self.dropout_masks}
            if len(shapes) != 1 or any(
                mask.dtype != np.bool_
                or mask.ndim != 3
                or mask.shape[0] != mask.shape[1]
                or 0 in mask.shape
                for mask in self.dropout_masks
            ):
                raise ValueError("MC dropout mask tape has invalid shape/dtype")
            np.savez_compressed(
                self.out / "dropout-tape.npz", keep_masks=np.stack(self.dropout_masks)
            )
        elif self.dropout_masks or self.dropout_rate is not None:
            raise ValueError("unexpected dropout tape on non-dropout branch")
        if self.in_sampler or self.in_msa:
            raise ValueError("unclosed capture context")
        for name, values in self.draws.items():
            expected = {0} if name == "init" else set(range(self.steps))
            if set(values) != expected:
                raise ValueError(f"missing/extra {name} draw indices")
        init = self.draws["init"][0]
        if init.ndim != 3 or init.shape[0] != self.samples or init.shape[-1] != 3:
            raise ValueError("capture requires one unbatched n5 sampler chunk")
        churn = np.stack([self.draws["churn"][i] for i in range(self.steps)])
        rotation = np.stack([self.draws["rotation"][i] for i in range(self.steps)])
        translation = np.stack(
            [self.draws["translation"][i] for i in range(self.steps)]
        )
        if churn.shape != (self.steps, *init.shape):
            raise ValueError("churn tape shape differs from initial-noise shape")
        if rotation.shape != (self.steps, self.samples, 3, 3):
            raise ValueError("rotation tape requires native flat N_augment matrices")
        if translation.shape != (self.steps, self.samples, 1, 3):
            raise ValueError("translation tape requires native N_sample=1 augmentation")
        if self.schedule is None or self.schedule.shape != (self.steps + 1,):
            raise ValueError("missing or wrong-shape native sampler schedule")
        if len(self.msa) != self.cycles:
            raise ValueError("missing/extra native MSA sampling calls")
        np.savez(
            self.out / "sampler-tape.npz",
            init_noise=init,
            step_noises=churn,
            rotations=rotation,
            translations=translation[..., 0, :],
            noise_schedule=self.schedule,
        )
        self.bundle("msa-tape", self.msa)
        save(
            self.out / "capture-complete.json",
            {
                "passed": True,
                "instrumented": True,
                "not_performance": True,
                "samples": self.samples,
                "steps": self.steps,
                "cycles": self.cycles,
                "sampler_calls": 1,
                "initial_draws": 1,
                "churn_draws": self.steps,
                "rotations": self.steps,
                "translations": self.steps,
                "msa_calls": len(self.msa),
                "mc_dropout_applied": self.mc_dropout,
                "mc_dropout_random_draws": self.random_decisions,
                "mc_dropout_mask_calls": len(self.dropout_masks),
                "mc_dropout_rate": self.dropout_rate,
                "bf16_storage_mapping": (
                    "exact numeric widening to FP32; see tree metadata"
                ),
            },
        )


def install_observers(stack, recorder, runner, torch, generator, model_module, utils):
    from protenix.data.inference.infer_dataloader import InferenceDataset
    from protenix.model.modules import pairformer

    original_dataset = InferenceDataset.__getitem__
    original_predict = runner.InferenceRunner.predict
    original_pair = model_module.Protenix.get_pairformer_output
    original_sampler = generator.sample_diffusion
    sampler_code = code_child(original_sampler, "_chunk_sample_diffusion")
    augmentation_code = utils.centre_random_augmentation.__code__
    main_loop_code = model_module.Protenix.main_inference_loop.__code__
    original_randn, original_rotation = torch.randn, utils.uniform_random_rotation
    original_msa, original_indices = (
        pairformer.sample_msa_feature_dict_random_without_replacement,
        utils.sample_indices,
    )
    original_random = random.random

    def dataset(self, index):
        result = original_dataset(self, index)
        data, atoms, error = result
        if error:
            raise RuntimeError(f"native preprocessing failed: {error}")
        recorder.bundle("native-input", data["input_feature_dict"])
        recorder.bundle(
            "native-identity",
            {
                "output_atom_" + name: getattr(atoms, attribute)
                for name, attribute in (
                    ("name", "atom_name"),
                    ("element", "element"),
                    ("res_name", "res_name"),
                    ("chain_id", "chain_id"),
                    ("res_id", "res_id"),
                )
            },
        )
        annotations = {
            name: atoms.get_annotation(name)
            for name in atoms.get_annotation_categories()
        }
        recorder.bundle("native-atom-annotations", annotations)
        return result

    def predict(self, data):
        save(recorder.out / "effective-config.json", self.model.configs.to_dict())
        hooks = []

        def embedded(module, args, result):
            recorder.bundle("input-embedding", {"s_inputs": result})

        def distogram(module, args, result):
            recorder.bundle("distogram", {"input": args[0], "logits": result})

        def confidence(module, args, kwargs):
            keys = ("s_inputs", "s_trunk", "z_trunk", "x_pred_coords", "pair_mask")
            recorder.bundle("confidence-input", {name: kwargs[name] for name in keys})

        hooks.append(self.model.input_embedder.register_forward_hook(embedded))
        hooks.append(self.model.distogram_head.register_forward_hook(distogram))
        hooks.append(
            self.model.confidence_head.register_forward_pre_hook(
                confidence, with_kwargs=True
            )
        )
        save(
            recorder.out / "operator-policy.json",
            {
                "autocast_dtype": str(torch.bfloat16),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "layernorm_type": os.environ.get("LAYERNORM_TYPE", "fast_layernorm"),
                "confidence_skip_amp": self.model.configs.skip_amp.confidence_head,
                "diffusion_skip_amp": self.model.configs.skip_amp.sample_diffusion,
                **(
                    {"fp32_atom_aggregation": True}
                    if recorder.fp32_atom_aggregation
                    else {}
                ),
            },
        )
        try:
            result = original_predict(self, data)
            torch.cuda.synchronize()
            recorder.bundle("prediction", result)
            recorder.completed += 1
            return result
        finally:
            for hook in hooks:
                hook.remove()

    def pair(self, *args, **kwargs):
        bound = inspect.signature(original_pair).bind(self, *args, **kwargs)
        bound.apply_defaults()
        recorder.mc_dropout = bool(bound.arguments["mc_dropout"])
        features = bound.arguments["input_feature_dict"]
        recorder.bundle(
            "native-derived",
            {key: features[key] for key in ("relp", "d_lm", "v_lm", "pad_info")},
        )
        if recorder.mc_dropout:
            from bench.protenix_dropout_tape import capture_native_dropout

            recorder.dropout_rate = float(self.configs.mc_dropout_rate)
            with capture_native_dropout(
                expected_calls=recorder.cycles, rate=recorder.dropout_rate
            ) as masks:
                result = original_pair(self, *args, **kwargs)
            recorder.dropout_masks = masks
        else:
            result = original_pair(self, *args, **kwargs)
        recorder.bundle("trunk", dict(zip(("s_inputs", "s", "z"), result, strict=True)))
        return result

    def sampler(*args, **kwargs):
        if recorder.schedule is not None or recorder.in_sampler:
            raise ValueError("unexpected extra native sampler call")
        bound = inspect.signature(original_sampler).bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["N_sample"] != recorder.samples:
            raise ValueError("wrong native sampler sample count")
        if bound.arguments["diffusion_chunk_size"] not in (None, recorder.samples):
            raise ValueError("native sample chunk route differs from finite capture")
        recorder.schedule = host_array(bound.arguments["noise_schedule"])
        recorder.bundle(
            "sampler-input",
            {key: bound.arguments[key] for key in ("s_inputs", "s_trunk", "z_trunk")},
        )
        recorder.in_sampler = True
        try:
            return original_sampler(*args, **kwargs)
        finally:
            recorder.in_sampler = False

    def randn(*args, **kwargs):
        caller = sys._getframe(1)
        result = original_randn(*args, **kwargs)
        if recorder.in_sampler:
            if caller.f_code is sampler_code:
                kind = "churn" if "step_i" in caller.f_locals else "init"
                recorder.draw(kind, caller.f_locals.get("step_i", 0), result)
            elif caller.f_code is augmentation_code:
                parent = caller.f_back
                if parent.f_code is not sampler_code:
                    raise ValueError("augmentation was not called by native sampler")
                if caller.f_locals["s_trans"] != 1.0:
                    raise ValueError("non-default native translation scale")
                recorder.draw("translation", parent.f_locals["step_i"], result)
            else:
                raise ValueError("unclassified native randn call inside sampler")
        return result

    def rotation(*args, **kwargs):
        caller = sys._getframe(1)
        result = original_rotation(*args, **kwargs)
        if recorder.in_sampler:
            if (
                caller.f_code is not augmentation_code
                or caller.f_back.f_code is not sampler_code
            ):
                raise ValueError(
                    "rotation was not called by native sampler augmentation"
                )
            recorder.draw("rotation", caller.f_back.f_locals["step_i"], result)
        return result

    def indices(*args, **kwargs):
        result = original_indices(*args, **kwargs)
        if recorder.in_msa:
            if recorder.msa_indices is not None:
                raise ValueError("multiple row-index calls in one MSA selection")
            recorder.msa_indices = host_array(result)
        return result

    def msa(*args, **kwargs):
        if recorder.in_msa:
            raise ValueError("nested MSA sample call")
        bound = inspect.signature(original_msa).bind(*args, **kwargs)
        bound.apply_defaults()
        recorder.msa_indices = None
        recorder.in_msa = True
        try:
            result = original_msa(*args, **kwargs)
            rows = recorder.msa_indices
            if rows is None:
                raise ValueError("actual native MSA row indices were not observed")
            cutoff = bound.arguments["cutoff"]
            rows = rows[:cutoff] if cutoff > 0 else rows
            selected = {name: host_array(value) for name, value in result.items()}
            for name, dim in bound.arguments["dim_dict"].items():
                expected = np.take(
                    host_array(bound.arguments["feat_dict"][name]), rows, axis=dim
                )
                if not np.array_equal(selected[name], expected):
                    raise ValueError(f"MSA index/selected-array mismatch: {name}")
            recorder.msa.append({"rows": rows.copy(), "selected": selected})
            return result
        finally:
            recorder.in_msa = False

    def python_random():
        caller = sys._getframe(1)
        result = original_random()
        if caller.f_code is main_loop_code:
            recorder.random_decisions.append(float(result))
        return result

    for owner, name, replacement in (
        (InferenceDataset, "__getitem__", dataset),
        (runner.InferenceRunner, "predict", predict),
        (model_module.Protenix, "get_pairformer_output", pair),
        (model_module, "sample_diffusion", sampler),
        (torch, "randn", randn),
        (utils, "uniform_random_rotation", rotation),
        (pairformer, "sample_msa_feature_dict_random_without_replacement", msa),
        (utils, "sample_indices", indices),
        (random, "random", python_random),
    ):
        stack.enter_context(patch.object(owner, name, replacement))


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", choices=MODELS, default=MODELS[0])
    parser.add_argument("--fp32-atom-aggregation", action="store_true")
    args = parser.parse_args()
    for name in ("input", "native_source", "weights", "assets_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    payload = json.loads(args.input.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("capture requires exactly one native input job")
    asset_records = input_assets(payload)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = out / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / f"{args.model_name}.pt").symlink_to(args.weights)
    sys.path.insert(0, str(args.native_source))
    import torch
    from protenix.model import generator, utils
    from protenix.model import protenix as model_module
    from runner import inference as runner

    for module in (runner, generator, utils, model_module):
        if not Path(module.__file__).resolve().is_relative_to(args.native_source):
            raise ValueError("publisher import did not resolve to the pinned source")
    if not torch.cuda.is_available():
        raise RuntimeError("native closure capture requires its CUDA publisher runtime")
    recorder = NativeRecorder(out)
    recorder.fp32_atom_aggregation = args.fp32_atom_aggregation
    snapshot = Path(__file__).resolve().parents[1]
    provenance = {
        "schema": 1,
        "arm": "native",
        "instrumented": True,
        "input_sha256": sha(args.input),
        "input_assets": asset_records,
        "source": source_identity(args.native_source),
        "wrapper_sha256": sha(Path(__file__)),
        "snapshot_python_source": {
            str(path.relative_to(snapshot)): sha(path)
            for directory in ("src", "bench")
            for path in sorted((snapshot / directory).rglob("*.py"))
        },
        "checkpoint_sha256": sha(args.weights),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "scipy",
                "rdkit",
                "biotite",
                "cuequivariance-torch",
            )
        },
        "device": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "cwd": os.getcwd(),
        "model_name": args.model_name,
        "precision_overrides": [],
        "download_policy": "local-only preflight",
        "environment_policy": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "TORCH_CUDA_ARCH_LIST",
                "LAYERNORM_TYPE",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "CUDNN_DETERMINISTIC",
                "OMP_NUM_THREADS",
                "PROTENIX_ROOT_DIR",
                "CUDA_MODULE_LOADING",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
            )
        },
    }
    save(out / "provenance.json", provenance)

    def local_only(configs):
        provenance["local_assets"] = preflight_configs(configs)
        # Native CCD helpers read configs_data directly rather than the parsed
        # flags. Different source paths must still resolve to identical bytes.
        for name in ("ccd_components_file", "ccd_components_rdkit_mol_file"):
            actual = Path(runner.data_configs[name])
            if sha(actual) != provenance["local_assets"][name]["sha256"]:
                raise ValueError(f"native CCD module/CLI asset differs: {name}")
        save(out / "provenance.json", provenance)
        save(out / "initial-config.json", configs.to_dict())

    argv = [
        "runner.inference",
        "--input_json_path",
        str(args.input),
        "--dump_dir",
        str(out / "prediction-native"),
        "--load_checkpoint_dir",
        str(checkpoint_dir),
        "--model_name",
        args.model_name,
        "--data.ccd_components_file",
        str(args.assets_root / "components.cif"),
        "--data.ccd_components_rdkit_mol_file",
        str(args.assets_root / "components.cif.rdkit_mol.pkl"),
    ]
    if any(any(char.isspace() for char in arg) for arg in argv):
        raise ValueError("publisher parse_sys_args does not preserve paths with spaces")
    save(out / "native-argv.json", argv)
    start = time.monotonic()
    with ExitStack() as stack:
        if args.fp32_atom_aggregation:
            from bench.protenix_atom_reduction_control import fp32_atom_aggregation

            stack.enter_context(fp32_atom_aggregation("native"))
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(
            patch.object(runner, "download_inference_cache", local_only)
        )
        install_observers(
            stack, recorder, runner, torch, generator, model_module, utils
        )
        runner.run()
    # infer_predict catches exceptions, so returning from run() is insufficient.
    if source_identity(args.native_source) != provenance["source"]:
        raise ValueError("publisher source changed during capture")
    recorder.finish()
    save(
        out / "timing.json",
        {
            "instrumented_elapsed_seconds": time.monotonic() - start,
            "not_uninstrumented_performance": True,
        },
    )


if __name__ == "__main__":
    main()
