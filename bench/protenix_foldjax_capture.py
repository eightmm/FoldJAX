"""Independent Protenix preprocessing and native-tape replay diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import flatten, save, sha

_TAPE_ARGUMENTS = (
    "init_noise",
    "step_noises",
    "rotations",
    "translations",
    "cycle_msa_features",
    "cycle_msa_index_tape",
    "cycle_pair_dropout_keep_masks",
)
_OTHER_DYNAMIC_ARGUMENTS = ("key", "pair_mask", "guidance_features")


def infer_boundary_report(route, noise_schedule, kwargs):
    """Observe concrete wrapper outputs, not traced or device-consumed draws."""
    import jax

    if route not in {"protenix_infer_static", "protenix_infer_compiled"}:
        raise ValueError(f"unrecognized inference boundary: {route}")

    def signature(value):
        if value is None:
            return {"present": False}
        if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value)):
            raise TypeError("inference observer requires concrete arrays, not tracers")
        arrays = flatten(jax.device_get(value))
        return {
            "present": True,
            "pytree": str(jax.tree.structure(value)),
            "leaves": {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "content_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
                for name, array in arrays.items()
            },
        }

    options = {
        name: str(np.dtype(value))
        if name == "trunk_dtype" and value is not None
        else value
        for name, value in kwargs.items()
        if name not in (*_TAPE_ARGUMENTS, *_OTHER_DYNAMIC_ARGUMENTS)
    }
    # Do not stringify unknown values into seemingly valid effective settings.
    options = json.loads(json.dumps(options, allow_nan=False))
    return {
        "schema_version": 1,
        "boundary": f"predict wrapper -> {route} host entry",
        "scope": (
            "concrete arrays after wrapper tape injection and graph/scan selection; "
            "not compiled-pool normalization or sampler/device-consumer evidence"
        ),
        "sampler_consumer_bytes_observed": False,
        "options": options,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "tape_inputs": {
            "noise_schedule": signature(noise_schedule),
            **{name: signature(kwargs.get(name)) for name in _TAPE_ARGUMENTS},
        },
        "not_hashed_here": ["input_feature_dict", "params", *_OTHER_DYNAMIC_ARGUMENTS],
    }


def featurize(input_path, out):
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json

    jobs = json.loads(input_path.read_text())
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("capture requires exactly one input job")
    features = featurize_protein_json(
        jobs[0],
        base_dir=input_path.parent,
        n_queries=32,
        n_keys=128,
        max_msa_depth=16384,
    )
    record_features(input_path, features, out)
    return features


def record_features(input_path, features, out):
    flat = flatten(features)
    np.savez_compressed(out / "foldjax-input.npz", **flat)
    save(
        out / "foldjax-input-tree.json",
        {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in flat.items()
        },
    )
    source = Path(__file__).resolve().parents[1]
    save(
        out / "feature-provenance.json",
        {
            "input_sha256": sha(input_path),
            "scope": (
                "independent preprocessing only; not inference or parity admission"
            ),
            "wrapper_sha256": sha(Path(__file__)),
            "source_files": {
                str(path.relative_to(source)): sha(path)
                for path in sorted(
                    (source / "src/foldjax/models/protenix/data").rglob("*")
                )
                if path.suffix in (".py", ".npz")
            },
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "rdkit")
            },
        },
    )


def msa_cycles(features, tape):
    """Gather native-selected rows from independently verified FoldJAX inputs."""
    fields = ("msa", "has_deletion", "deletion_value")
    expected_keys = {
        f"{i}.{field}"
        for i in range(10)
        for field in ("rows", *(f"selected.{name}" for name in fields))
    }
    if set(tape) != expected_keys:
        raise ValueError("unmapped or incomplete native MSA tape schema")
    cycles = []
    for i in range(10):
        rows = tape[f"{i}.rows"]
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
            raise ValueError("native MSA indices must be a nonempty integer vector")
        if (rows < 0).any() or (rows >= len(features["msa"])).any():
            raise ValueError("native MSA row outside independent alignment")
        cycle = {}
        for name in fields:
            value = np.asarray(features[name])[rows]
            if not np.array_equal(value, tape[f"{i}.selected.{name}"]):
                raise ValueError(f"independent MSA selection differs: {i}.{name}")
            cycle[name] = value
        cycles.append(cycle)
    return tuple(cycles)


def save_jax_boundary(path, value):
    import jax

    arrays = flatten(jax.device_get(value))
    metadata = {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in arrays.items()
    }
    arrays = {
        name: array.astype(np.float32) if str(array.dtype) == "bfloat16" else array
        for name, array in arrays.items()
    }
    np.savez_compressed(path, **arrays)
    save(path.with_suffix(".tree.json"), metadata)


def replay(args):
    import jax
    import jax.numpy as jnp

    from bench.protenix_closure_report import check_inputs
    from foldjax.models.protenix.cli.predict import main as predict_cli

    reference = args.reference
    completion = json.loads((reference / "capture-complete.json").read_text())
    native = json.loads((reference / "provenance.json").read_text())
    evidence = {}
    if getattr(args, "weight_audit", None) is not None:
        audit = json.loads(args.weight_audit.read_text())
        if (
            audit.get("passed") is not True
            or audit["native_checkpoint_sha256"] != native["checkpoint_sha256"]
            or audit["managed_checkpoint_sha256"] != sha(args.weights)
        ):
            raise ValueError("checkpoint audit does not bind both executed weights")
        evidence["weight_audit_sha256"] = sha(args.weight_audit)
    if getattr(args, "calibration", None) is not None:
        from bench.native_repeat_policy import evaluate_candidate
        from bench.protenix_closure_report import native_record

        calibrated = json.loads(args.calibration.read_text())
        record, _ = native_record(reference)
        # Validate the seal and frozen first reference without any candidate data.
        evaluate_candidate(calibrated, record, record)
        evidence["calibration_sha256"] = calibrated["calibration_sha256"]
        evidence["calibration_file_sha256"] = sha(args.calibration)
    if not completion["passed"]:
        raise ValueError("native tape capture did not complete on supported path")
    if native["model_name"] != "protenix_base_default_v1.0.0":
        raise ValueError("this replay currently audits the small released profile only")
    if native["input_sha256"] != sha(args.input):
        raise ValueError("native and FoldJAX input documents differ")
    native_config = json.loads((reference / "effective-config.json").read_text())
    fp32_aggregation = getattr(args, "fp32_atom_aggregation", False)
    policy_file = reference / "operator-policy.json"
    native_aggregation = (
        json.loads(policy_file.read_text()).get("fp32_atom_aggregation", False)
        if policy_file.exists()
        else False
    )
    if type(native_aggregation) is not bool or native_aggregation != fp32_aggregation:
        raise ValueError("native/candidate atom aggregation policies differ")
    from bench.protenix_dropout_tape import load_dropout_tape

    dropout_masks, dropout_rate = load_dropout_tape(
        reference, completion, native_config
    )
    with np.load(reference / "sampler-tape.npz", allow_pickle=False) as archive:
        tape = dict(archive)
    with np.load(reference / "msa-tape.npz", allow_pickle=False) as archive:
        msa = dict(archive)
    if set(tape) != {
        "init_noise",
        "step_noises",
        "rotations",
        "translations",
        "noise_schedule",
    }:
        raise ValueError("unexpected sampler tape schema")
    if tape["init_noise"].shape[0] != 5 or tape["noise_schedule"].shape != (201,):
        raise ValueError("native tape is not n5/200")
    feature_module = importlib.import_module(
        "foldjax.models.protenix.data.featurize_json"
    )
    prediction_module = importlib.import_module(
        "foldjax.models.protenix.models.predict"
    )
    original_features = feature_module.featurize_protein_json
    original_predict = prediction_module.protenix_predict_static
    original_schedule = prediction_module.inference_noise_schedule
    original_infer = {
        name: getattr(prediction_module, name)
        for name in ("protenix_infer_static", "protenix_infer_compiled")
    }
    seen = {"features": 0, "predict": 0, "schedule": 0, "infer_boundary": 0}
    selected = {}

    def observe_infer(route):
        def observed(features, params, noise_schedule, **kwargs):
            if seen["infer_boundary"]:
                raise ValueError("capture expects exactly one concrete inference entry")
            save(
                args.out / "infer-boundary.json",
                infer_boundary_report(route, noise_schedule, kwargs),
            )
            seen["infer_boundary"] += 1
            return original_infer[route](features, params, noise_schedule, **kwargs)

        return observed

    def capture_features(*a, **kw):
        features = original_features(*a, **kw)
        record_features(args.input, features, args.out)
        report = check_inputs(reference, features)
        save(args.out / "input-audit.json", report)
        if not report["passed"]:
            raise ValueError(f"independent input gate failed: {report}")
        selected["cycles"] = msa_cycles(features, msa)
        seen["features"] += 1
        return features

    def schedule(**kwargs):
        generated = np.asarray(original_schedule(**kwargs))
        actual = tape["noise_schedule"]
        if generated.shape != actual.shape:
            raise ValueError("native and FoldJAX schedule shapes differ")
        save(
            args.out / "schedule-audit.json",
            {
                "generated_bitwise_equal": generated.tobytes() == actual.tobytes(),
                "generated_max_abs_error": float(np.max(np.abs(generated - actual))),
                "replay": "actual native FP32 schedule supplied at the public wrapper",
            },
        )
        seen["schedule"] += 1
        return jnp.asarray(actual)

    def capture_predict(params, features, key, **kwargs):
        if (
            kwargs["num_samples"],
            kwargs["num_sampling_steps"],
            kwargs["recycling_steps"],
            kwargs["trunk_dtype"],
        ) != (5, 200, 10, jnp.bfloat16):
            raise ValueError("FoldJAX effective sampling/precision differs")
        # The CLI does not expose this library knob. Match the publisher's
        # cache/fusion policy explicitly for this native-policy comparison.
        kwargs["use_diffusion_efficient_fusion"] = native_config[
            "enable_efficient_fusion"
        ]
        save(
            args.out / "requested-wrapper-options.json",
            {
                name: str(value) if name == "trunk_dtype" else value
                for name, value in kwargs.items()
                if name not in {"cycle_msa_index_tape", "guidance_features"}
            },
        )
        kwargs.update(
            {
                name: jnp.asarray(tape[name])
                for name in ("init_noise", "step_noises", "rotations", "translations")
            }
        )
        kwargs["cycle_msa_index_tape"] = None
        kwargs["cycle_msa_features"] = jax.tree.map(jnp.asarray, selected["cycles"])
        if dropout_masks is not None:
            kwargs["cycle_pair_dropout_keep_masks"] = jnp.asarray(dropout_masks)
            kwargs["pair_dropout_rate"] = dropout_rate
        output = original_predict(params, features, key, **kwargs)
        jax.block_until_ready(output)
        save_jax_boundary(args.out / "prediction.npz", output)
        seen["predict"] += 1
        return output

    argv = [
        "--input-json",
        str(args.input),
        "--weights",
        str(args.weights),
        "--out",
        str(args.out / "cli-output.npz"),
        "--output-format",
        "npz",
        "--model-name",
        "protenix_base_default_v1.0.0",
        "--include-trunk",
        "--trunk-triangle-attention-backend",
        "cueq",
        "--compile-cache",
        str(args.out / "jax-cache"),
    ]
    save(args.out / "foldjax-argv.json", argv)
    started = time.monotonic()
    from bench.protenix_atom_reduction_control import fp32_atom_aggregation

    with (
        fp32_atom_aggregation("foldjax") if fp32_aggregation else nullcontext(),
        patch.object(feature_module, "featurize_protein_json", capture_features),
        patch.object(prediction_module, "protenix_predict_static", capture_predict),
        patch.object(prediction_module, "inference_noise_schedule", schedule),
        patch.object(
            prediction_module,
            "protenix_infer_static",
            observe_infer("protenix_infer_static"),
        ),
        patch.object(
            prediction_module,
            "protenix_infer_compiled",
            observe_infer("protenix_infer_compiled"),
        ),
    ):
        predict_cli(argv)
    if seen != {"features": 1, "predict": 1, "schedule": 1, "infer_boundary": 1}:
        raise ValueError(f"incomplete public FoldJAX lifecycle: {seen}")
    source = Path(__file__).resolve().parents[1]
    save(
        args.out / "provenance.json",
        {
            "arm": "foldjax",
            "fp32_atom_aggregation": fp32_aggregation,
            "instrumented": True,
            "preflight_evidence": evidence,
            "input_sha256": sha(args.input),
            "checkpoint_sha256": sha(args.weights),
            "native_reference_sha256": sha(reference / "provenance.json"),
            "sampler_tape_sha256": sha(reference / "sampler-tape.npz"),
            "msa_tape_sha256": sha(reference / "msa-tape.npz"),
            "snapshot_python_source": {
                str(path.relative_to(source)): sha(path)
                for directory in ("src", "bench")
                for path in sorted((source / directory).rglob("*.py"))
            },
            "jax_version": jax.__version__,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "explicit_kernel": "cueq triangle attention; remaining CLI defaults",
            "tape_consumption_scope": (
                "concrete post-wrapper inference-entry arrays hashed in "
                "infer-boundary.json; sampler/device-consumer bytes not observed"
            ),
            "infer_boundary_sha256": sha(args.out / "infer-boundary.json"),
        },
    )
    save(
        args.out / "capture-complete.json",
        {
            "passed": True,
            "not_parity_admission": True,
            "seen": seen,
            "instrumented_seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weight-audit", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--fp32-atom-aggregation", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.reference is not None:
        if args.weights is None:
            parser.error("--reference replay requires --weights")
        return replay(args)
    features = featurize(args.input.resolve(strict=True), args.out)
    print(json.dumps({"features": len(flatten(features)), "complete": True}))


if __name__ == "__main__":
    main()
