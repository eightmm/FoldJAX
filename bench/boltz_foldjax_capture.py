"""Instrumented native-tape Boltz replay with bounded first/final trunk captures.

This is a core-only diagnostic using captured native features, not independent
preprocessing evidence. Samples remain in sampler index order. Ordered host
callbacks change the compiled graph and invalidate performance comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import load_legacy, save_new
from bench.protenix_foldjax_capture import save_jax_boundary


def array_identity(value):
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "content_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


class StageObserver:
    """Count actual callback executions, not Python traces of a scan body."""

    CYCLIC = {"msa_module", "pairformer_module"}
    ONCE = {
        "input_embedder",
        "input_embedder/atom_encoder",
        "input_embedder/atom_attention_encoder",
        "rel_pos",
    }

    def __init__(self, out, recycles):
        self.out, self.recycles = out, recycles
        self.counts = Counter()
        self.artifacts = {}

    def callback(self, name, value):
        index = self.counts[name]
        self.counts[name] += 1
        expected = self.recycles + 1 if name in self.CYCLIC else 1
        if self.counts[name] > expected:
            raise RuntimeError(f"unexpected repeated stage {name}")
        if name in self.CYCLIC:
            if index not in {0, self.recycles}:
                return
            label = f"trunk-boundaries/cycle-{index:02d}/{name}"
        else:
            label = f"trunk-boundaries/{name}"
        path = self.out / f"{label}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.with_suffix(".tree.json").exists():
            raise FileExistsError(f"refusing to overwrite stage {label}")
        save_jax_boundary(path, value)
        self.artifacts[label] = {
            "arrays_sha256": sha(path),
            "tree_sha256": sha(path.with_suffix(".tree.json")),
        }

    def wrap(self, name, original):
        import jax

        def observed(*args, **kwargs):
            output = original(*args, **kwargs)
            if name == "msa_module":
                value = {"input_z": args[1], "delta_z": output}
            elif name == "pairformer_module":
                value = {
                    "input_s": args[1],
                    "input_z": args[2],
                    "output_s": output[0],
                    "output_z": output[1],
                }
            else:
                value = output
            jax.debug.callback(
                lambda result: self.callback(name, result), value, ordered=True
            )
            return output

        return observed

    def install(self, trunk, embedder):
        stack = ExitStack()
        for module, attribute, label in (
            (trunk, "input_embedder_forward", "input_embedder"),
            (trunk, "msa_module_forward", "msa_module"),
            (trunk, "pairformer_module_forward", "pairformer_module"),
            (trunk, "relative_position_forward", "rel_pos"),
            (embedder, "atom_encoder_forward", "input_embedder/atom_encoder"),
            (
                embedder,
                "atom_attention_encoder_forward",
                "input_embedder/atom_attention_encoder",
            ),
        ):
            stack.enter_context(
                patch.object(
                    module, attribute, self.wrap(label, getattr(module, attribute))
                )
            )
        return stack

    def validate(self):
        expected = {name: 1 for name in self.ONCE}
        expected.update({name: self.recycles + 1 for name in self.CYCLIC})
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"incomplete stage observation: {dict(self.counts)} != {expected}"
            )


def native_settings(root):
    complete = json.loads((root / "capture-complete.json").read_text())
    meta = json.loads((root / "tape.json").read_text())
    effective = json.loads((root / "effective-model-settings.json").read_text())
    if complete.get("passed") is not True:
        raise ValueError("native capture did not complete")
    if (meta["num_samples"], meta["num_steps"], meta["num_recycles"]) != (5, 200, 3):
        raise ValueError("this native-policy observer requires n5/200/3")
    if (
        meta["precision"] != "bf16-mixed"
        or not meta["kernels"]
        or meta["subsample_msa"]
        or effective["float32_matmul_precision"] != "highest"
        or not effective["cuda_autocast_enabled"]
        or effective["cuda_autocast_dtype"] != "torch.bfloat16"
    ):
        raise ValueError(
            "native capture does not match BF16/highest/kernels/"
            "no-MSA-subsampling policy"
        )
    steering = effective["steering_args"]
    if steering.get("fk_steering", False):
        raise ValueError(
            "FK steering requires additional tapes outside this diagnostic"
        )
    return meta, effective


def load_inputs(legacy, root, meta):
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        for name, array in archive.items():
            if (
                array.dtype == np.int64
                and array.size
                and (
                    array.min() < np.iinfo(np.int32).min
                    or array.max() > np.iinfo(np.int32).max
                )
            ):
                raise ValueError(
                    f"feature {name} cannot be losslessly narrowed to int32"
                )
    with np.load(root / "tape.npz", allow_pickle=False) as archive:
        if set(archive.files) != {
            "sigmas",
            "init_noise",
            "step_noises",
            "rotations",
            "translations",
        }:
            raise ValueError("unknown or incomplete native sampler tape schema")
        if any(
            array.dtype != np.float32 or not np.isfinite(array).all()
            for array in archive.values()
        ):
            raise ValueError("native sampler tape must contain finite FP32 arrays")
    return (
        legacy.load_features(root / "features.npz"),
        legacy.load_tape(root / "tape.npz", meta),
    )


def prediction_options(meta, effective, *, trunk_only, bfactor=False, features=None):
    options = {
        "recycling_steps": int(meta["num_recycles"]),
        "num_sampling_steps": int(meta["num_steps"]),
        "multiplicity": int(meta["num_samples"]),
        "step_scale": float(meta["step_scale"]),
        "gamma_0": float(meta["gamma_0"]),
        "gamma_min": float(meta["gamma_min"]),
        "noise_scale": float(meta["noise_scale"]),
        "sigma_data": float(meta["sigma_data"]),
        "sigma_min": 0.0001,
        "sigma_max": 160.0,
        "rho": 7.0,
        "compute_dtype": "bfloat16",
        "matmul_precision": "highest",
        "triangle_backend": "cueq",
        "attention_backend": "xla",
        "glu_backend": "xla",
        "trunk_atom_attention_backend": "xla",
        "chunk_size": 128,
        "use_scan": True,
        "trunk_use_scan": True,
        "score_use_scan": True,
        "subsample_msa": False,
        "use_template": bool(effective["use_templates"]),
        "augmentation": True,
        "alignment_reverse_diff": True,
        "steering_args": effective["steering_args"],
        "confidence_sequentially": True,
        "return_pair_chains_iptm": True,
        "run_confidence": not trunk_only,
        "run_distogram": not trunk_only,
        "run_bfactor": bfactor and not trunk_only,
        "return_confidence_logits": True,
        "stop_after_trunk": trunk_only,
        "return_representations": ("single_inputs", "single", "pair"),
    }
    if features is not None:
        # Nested metric keys are host metadata, not dynamic JIT output structure.
        # Preserve native IDs (including padding labels); effective-options records
        # the exact tuple alongside the hashes of the input feature arrays.
        options["confidence_chain_ids"] = tuple(
            int(value) for value in np.unique(np.asarray(features["asym_id"]))
        )
    return options


def make_prediction(predict, trunk, options):
    """Keep the native schedule a dynamic JIT argument, not a closed constant."""

    def run(params, features, key, tape):
        def schedule(num_steps, **kwargs):
            if num_steps + 1 != tape["sigmas"].shape[0]:
                raise ValueError("native schedule does not match resolved step count")
            return tape["sigmas"]

        with patch.object(trunk, "_sample_schedule", schedule):
            return predict(
                params,
                features,
                key,
                init_noise=tape["init_noise"],
                step_noises=tape["step_noises"],
                aug_transforms=(tape["rotations"], tape["translations"]),
                **options,
            )

    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--upstream-capture", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trunk-only", action="store_true")
    args = parser.parse_args(argv)
    source = args.source_root.resolve(strict=True)
    if Path(__file__).resolve() != source / "bench/boltz_foldjax_capture.py":
        raise ValueError("execute from the explicitly selected source root")
    native = args.upstream_capture.resolve(strict=True)
    meta, effective = native_settings(native)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.predict import boltz2_predict
    from foldjax.models.boltz2.weights import resolve_native_weight_bundle

    if (
        not Path(inspect.getfile(boltz2_predict))
        .resolve()
        .is_relative_to(source / "src")
    ):
        raise RuntimeError(
            "imported FoldJAX prediction source differs from source root"
        )
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("this native-policy capture requires exactly one GPU")
    trunk = importlib.import_module("foldjax.models.boltz2.models.trunk_blocks.trunk")
    embedder = importlib.import_module(
        "foldjax.models.boltz2.models.trunk_blocks.input_embedder"
    )
    legacy_path = source / "tests/models/boltz2/scripts/parity_matched_tape.py"
    legacy = load_legacy(legacy_path)
    features, tape = load_inputs(legacy, native, meta)
    native_tree = json.loads((native / "forward-output.tree.json").read_text())
    options = prediction_options(
        meta,
        effective,
        trunk_only=args.trunk_only,
        bfactor="pbfactor" in native_tree,
        features=features,
    )
    bundle = resolve_native_weight_bundle(args.weights)
    if bundle is None:
        raise FileNotFoundError(f"native weight bundle not found: {args.weights}")
    save_new(
        args.out_dir / "effective-options.json",
        {
            **options,
            "compiler_options": {"xla_allow_excess_precision": False},
            "triangle_multiplication_backend": "cueq",
            "sampler_tape": {
                name: array_identity(value) for name, value in tape.items()
            },
            "features": {
                name: array_identity(value) for name, value in features.items()
            },
        },
    )
    save_new(
        args.out_dir / "provenance.json",
        {
            "arm": "foldjax-core-only",
            "instrumented": True,
            "source_files": {
                str(path.relative_to(source)): sha(path)
                for directory in ("src", "bench")
                for path in sorted((source / directory).rglob("*.py"))
            },
            "legacy_loader_sha256": sha(legacy_path),
            "weights_sha256": sha(bundle[0]),
            "weight_sidecar_sha256": sha(bundle[1]) if bundle[1].exists() else None,
            "native_artifacts": {
                name: sha(native / name)
                for name in (
                    "capture-complete.json",
                    "provenance.json",
                    "tape.json",
                    "tape.npz",
                    "features.npz",
                    "effective-model-settings.json",
                    "forward-output.tree.json",
                )
            },
            "jax_version": jax.__version__,
            "device": str(jax.devices()[0]),
            "device_kind": jax.devices()[0].device_kind,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "scope": (
                "captured native features; not independent preprocessing "
                "or checkpoint-conversion proof"
            ),
            "output_order": "sample index; no confidence ranking or sample rematching",
            "tape_scope": "sampler not executed"
            if args.trunk_only
            else (
                "dynamic model-input tape; per-step sampler consumer bytes "
                "not independently observed"
            ),
            "callback_scope": (
                "every recycle transferred; first and final persisted; no timing claim"
            ),
        },
    )
    generated = np.asarray(
        trunk._sample_schedule(
            int(meta["num_steps"]),
            sigma_min=0.0001,
            sigma_max=160.0,
            sigma_data=float(meta["sigma_data"]),
            rho=7.0,
        )
    )
    save_new(
        args.out_dir / "schedule-audit.json",
        {
            "generated": array_identity(generated),
            "native": array_identity(tape["sigmas"]),
            "generated_max_abs_error": float(
                np.max(np.abs(generated - tape["sigmas"]))
            ),
            "policy": "native schedule supplied as dynamic model input",
            "sampler_executed": not args.trunk_only,
        },
    )
    params = load_params(args.weights)
    if params.get("affinity", {}).get("modules"):
        raise ValueError("this diagnostic requires confidence-only weights")
    observer = StageObserver(args.out_dir, int(meta["num_recycles"]))
    jax.config.update("jax_compilation_cache_dir", str(args.out_dir / "jax-cache"))
    started = time.monotonic()
    with (
        observer.install(trunk, embedder),
        jax.default_matmul_precision("highest"),
        patch.dict(os.environ, {"BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND": "cueq"}),
    ):
        # Preserve native AMP's explicit FP32 -> BF16 -> FP32 rounding.
        run = jax.jit(
            make_prediction(boltz2_predict, trunk, options),
            compiler_options={"xla_allow_excess_precision": False},
        )
        output = run(
            params,
            features,
            jax.random.PRNGKey(int(meta["seed"])),
            jax.tree.map(jnp.asarray, tape),
        )
        jax.block_until_ready(output)
        jax.effects_barrier()
    observer.validate()
    output_path = args.out_dir / "prediction.npz"
    save_jax_boundary(output_path, output)
    save_new(
        args.out_dir / "capture-complete.json",
        {
            "passed": True,
            "not_parity_admission": True,
            "trunk_only": args.trunk_only,
            "counts": dict(observer.counts),
            "stage_artifacts": observer.artifacts,
            "prediction_sha256": sha(output_path),
            "prediction_tree_sha256": sha(output_path.with_suffix(".tree.json")),
            "effective_options_sha256": sha(args.out_dir / "effective-options.json"),
            "instrumented_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
