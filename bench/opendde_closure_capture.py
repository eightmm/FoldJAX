"""Capture the native FP32 OpenDDE boundary and public FoldJAX tape replay.

Diagnostic only: successful inference is not parity admission. Run each arm in
its own environment, from an immutable source snapshot, through tsp/run-ledger.
The explicit legacy driver supplies the already audited independent-input
reconciliation; its source hash is retained, not treated as package authority.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import resource
import shutil
import time
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import flatten, save, sha


def numpy_tree(value):
    if isinstance(value, dict):
        return {key: numpy_tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_tree(child) for child in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def load_module(path):
    spec = importlib.util.spec_from_file_location("opendde_closure_driver", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def confidence_feature_boundary(features):
    # Only these features are read by the pinned native ConfidenceHead. The
    # full model input may now contain lazy trunk-only relative-position data.
    required = ("distogram_rep_atom_mask", "atom_to_token_idx", "atom_to_tokatom_idx")
    result = {key: numpy_tree(features[key]) for key in required}
    if "structural_distogram_rep_atom_mask" in features:
        result["structural_distogram_rep_atom_mask"] = numpy_tree(
            features["structural_distogram_rep_atom_mask"]
        )
    return result


def native_deterministic_policy(reference, provenance):
    config = json.loads((reference / "effective-config.json").read_text())
    value = config["deterministic"]
    if type(value) is not bool:
        raise ValueError("native deterministic policy must be explicit boolean")
    if provenance.get("native_deterministic", value) != value:
        raise ValueError("native deterministic policy conflicts with provenance")
    return value


def native_consumer_cycles(msa):
    """Finite native-to-FoldJAX MSA storage mapping for this replay route."""
    mask = np.asarray(msa["input_msa_mask"])
    if mask.dtype != bool or mask.ndim != 2 or not mask.all():
        raise ValueError("consumer audit requires native all-valid MSA rows")
    rows = np.asarray(msa["rows"])
    if rows.ndim != 2 or rows.shape[0] != 10 or rows.dtype.kind not in "iu":
        raise ValueError("native MSA row indices must have ten cycles")
    if np.any(rows < 0) or np.any(rows >= len(mask)):
        raise ValueError("native MSA row index out of range")
    result = []
    for index in range(10):
        cycle = {
            name: np.asarray(msa[f"selected_{name}"][index])
            for name in ("msa", "has_deletion", "deletion_value")
        }
        ids = cycle["msa"]
        if (
            ids.ndim != 2
            or ids.dtype.kind not in "iu"
            or np.any((ids < 0) | (ids > 31))
        ):
            raise ValueError("native MSA ids outside the audited categorical range")
        for name in ("has_deletion", "deletion_value"):
            if cycle[name].shape != ids.shape or cycle[name].dtype != np.float32:
                raise ValueError("native deletion fields must be shape-matched FP32")
        # build_cycle_msa returns JAX arrays: host-only compaction deliberately
        # leaves these at int32/FP32. The consumer observer verifies that fact.
        cycle["msa"] = ids.astype(np.int32)
        cycle["msa_mask"] = mask[rows[index]].astype(np.float32)
        result.append(cycle)
    return tuple(result)


def audit_request(args):
    from foldjax.schema import PredictionRequest

    return PredictionRequest(
        model="opendde",
        input=args.input,
        input_format="native",
        weights=args.repo / ".foldjax/weights/opendde/opendde.jax",
        output_dir=args.out / "prediction",
        cache_dir=args.out / "cache",
        seed=101,
        options={
            "include_raw": True,
            "matmul_precision": getattr(args, "jax_matmul_precision", "high"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--legacy-driver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable-native-tf32", action="store_true")
    parser.add_argument("--native-deterministic", action="store_true")
    parser.add_argument("--capture-confidence-boundary", action="store_true")
    parser.add_argument("--capture-linear-policy", action="store_true")
    parser.add_argument("--capture-trunk-boundary", action="store_true")
    parser.add_argument("--capture-ffi-policy", action="store_true")
    parser.add_argument("--capture-consumed-tape", action="store_true")
    parser.add_argument(
        "--jax-matmul-precision", choices=("high", "highest"), default="high"
    )
    args = parser.parse_args()
    if args.capture_consumed_tape and args.arm != "foldjax":
        parser.error("--capture-consumed-tape observes the FoldJAX consumer only")
    args.out.mkdir(parents=True, exist_ok=False)
    driver = load_module(args.legacy_driver)
    driver.REPO = args.repo
    driver.UP = args.native_source
    snapshot = Path(__file__).resolve().parents[1]
    from foldjax.paths import assets_dir

    assets = assets_dir()
    driver.module = lambda path: load_module(
        snapshot / "tests/models/opendde/scripts" / path.name
    )
    provenance = {
        "arm": args.arm,
        "scope": (
            "FP32 audit; TF32-off/highest/deterministic are controls, "
            "not native defaults"
        ),
        "input_sha256": sha(args.input),
        "input_assets": driver.input_assets(
            json.loads(args.input.read_text()), args.input.parent
        ),
        "managed_assets_sha256": {
            name: sha(assets / name)
            for name in ("components.cif", "components.cif.rdkit_mol.pkl")
        },
        "native_source": driver.source_identity(args.native_source),
        "source_files": {
            str(path.relative_to(snapshot)): sha(path)
            for directory in ("src", "bench", "tests/models/opendde/scripts")
            for path in sorted((snapshot / directory).rglob("*.py"))
        },
        "legacy_driver_sha256": sha(args.legacy_driver),
        "wrapper_sha256": sha(Path(__file__)),
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "rdkit")
        },
        "samples": 5,
        "steps": 200,
        "cycles": 10,
        "seed": 101,
        "trunk_dtype": "fp32",
        "native_tf32": not args.disable_native_tf32,
        "native_deterministic": args.native_deterministic,
        "instrumented": True,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_persistent_cache_enable_xla_caches": os.environ.get(
            "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"
        ),
        "backend_environment": {
            name: os.environ.get(name)
            for name in (
                "PROTENIX_TRIANGLE_BACKEND",
                "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND",
            )
        },
    }
    driver.SCOPE = (
        provenance["scope"] + "; independent inputs; raw/extracted confidence"
    )
    save(args.out / "provenance.json", provenance)
    started = time.perf_counter()
    if args.arm == "native":
        import torch
        from opendde.model.modules.confidence import ConfidenceHead
        from opendde.model.opendde import OpenDDE

        forward = OpenDDE.forward
        confidence_forward = ConfidenceHead.forward
        confidence_calls = []
        original_module = driver.module

        def configured_module(path):
            runner = original_module(path)
            parse = runner.parse_args

            def configured_args():
                value = parse()
                value.disable_tf32 = args.disable_native_tf32
                value.deterministic = args.native_deterministic
                return value

            runner.parse_args = configured_args
            return runner

        driver.module = configured_module

        @wraps(forward)
        def captured_forward(self, *a, **kw):
            save(args.out / "effective-config.json", self.configs.to_dict())
            provenance["torch_policy_at_forward_entry"] = {
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            }
            provenance["actual_parameter_dtypes"] = sorted(
                {str(value.dtype) for value in self.parameters()}
            )
            result = forward(self, *a, **kw)
            np.savez_compressed(args.out / "raw.npz", **flatten(numpy_tree(result[0])))
            return result

        @wraps(confidence_forward)
        def captured_confidence(self, *a, **kw):
            if not args.capture_confidence_boundary:
                return confidence_forward(self, *a, **kw)
            arguments = inspect.signature(confidence_forward).bind(self, *a, **kw)
            arguments.apply_defaults()
            values = arguments.arguments
            index = len(confidence_calls)
            boundary = {
                key: numpy_tree(values[key])
                for key in (
                    "s_inputs",
                    "s_trunk",
                    "z_trunk",
                    "x_pred_coords",
                    "pair_mask",
                )
                if values[key] is not None
            }
            boundary["input_feature_dict"] = confidence_feature_boundary(
                values["input_feature_dict"]
            )
            boundary["selected_distogram_rep_atom_mask"] = numpy_tree(
                self._select_distogram_rep_atom_mask(
                    values["input_feature_dict"], values["s_inputs"].shape[-2]
                )
            )
            np.savez_compressed(
                args.out / f"confidence-input-{index}.npz", **flatten(boundary)
            )
            metadata = {
                "index": index,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "autocast_enabled": torch.is_autocast_enabled("cuda"),
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "kwargs": {
                    key: values[key]
                    for key in (
                        "triangle_attention",
                        "triangle_multiplicative",
                        "inplace_safe",
                        "chunk_size",
                    )
                },
            }
            confidence_calls.append(metadata)
            result = confidence_forward(self, *a, **kw)
            np.savez_compressed(
                args.out / f"confidence-output-{index}.npz",
                **dict(
                    zip(
                        ("plddt", "pae", "pde", "resolved"),
                        numpy_tree(result),
                        strict=True,
                    )
                ),
            )
            return result

        from bench.opendde_linear_policy import LinearPolicyObserver

        with (
            LinearPolicyObserver(args.out)
            if args.capture_linear_policy
            else nullcontext(),
            patch.object(OpenDDE, "forward", captured_forward),
            patch.object(ConfidenceHead, "forward", captured_confidence),
        ):
            status = driver.native(args.input, args.out)
        if status:
            raise RuntimeError(f"native lifecycle failed: {status}")
        provenance["versions"]["torch"] = torch.__version__
        provenance["cuda"] = torch.version.cuda
        provenance["confidence_calls"] = confidence_calls
        if args.capture_confidence_boundary and not confidence_calls:
            raise ValueError("native confidence boundary was not reached")
        provenance["checkpoint_sha256"] = sha(
            args.repo / ".foldjax/downloads/opendde/opendde.pt"
        )
        memory = {"torch_max_memory_allocated": torch.cuda.max_memory_allocated()}
    else:
        if args.reference is None:
            parser.error("FoldJAX requires --reference")
        import jax

        from foldjax import predict
        from foldjax.models.opendde.cli import predict as cli

        for name in ("native-input.npz", "native-identity.npz", "native-derived.npz"):
            shutil.copy2(args.reference / name, args.out / name)
        (args.out / "torch").mkdir()
        for name in ("tape.json", "tape.npz", "msa.npz"):
            shutil.copy2(args.reference / "torch" / name, args.out / "torch" / name)
        runner = driver.module(Path("parity_matched_tape.py"))
        tape_meta = json.loads((args.reference / "torch/tape.json").read_text())
        reference_provenance = json.loads(
            (args.reference / "provenance.json").read_text()
        )
        provenance["native_tf32"] = reference_provenance["native_tf32"]
        provenance["native_deterministic"] = native_deterministic_policy(
            args.reference, reference_provenance
        )
        assert (
            tape_meta["num_steps"],
            tape_meta["num_samples"],
            tape_meta["num_recycles"],
        ) == (200, 5, 10)
        msa = driver.arrays(args.reference / "torch/msa.npz")
        featurize, infer, score = cli._featurize, cli._predict, cli._score
        seen = {"input": 0, "predict": 0, "score": 0}
        selected = {}
        consumed = None

        @wraps(featurize)
        def checked_features(*a, **kw):
            nonlocal consumed
            if kw.get("seed") != 101:
                raise ValueError("public preprocessing seed differs from audit seed")
            features = featurize(*a, **kw)
            driver.gate(features, args.out)
            cycles, agreement = runner.build_cycle_msa(
                features, msa, source="indices", num_recycles=10
            )
            if not agreement.get("featurizer_msa_rows_identical"):
                raise ValueError("native MSA tape cannot index different alignments")
            for index, cycle in enumerate(cycles):
                for name in ("msa", "has_deletion", "deletion_value"):
                    expected = runner._squeeze_batch(msa[f"selected_{name}"][index], 2)
                    if not np.array_equal(expected, cycle[name]):
                        raise ValueError(f"MSA replay mismatch {index}:{name}")
            _, tape = runner.load_tape(
                args.reference / "torch/tape.npz",
                num_steps=200,
                num_samples=5,
                n_atom=len(features["ref_pos"]),
            )
            selected.update(cycle_msa_features=cycles, **tape)
            if args.capture_consumed_tape:
                from bench.opendde_consumed_tape import ConsumedTape, expected_events

                consumed = ConsumedTape(
                    expected_events(
                        driver.arrays(args.reference / "torch/tape.npz"),
                        native_consumer_cycles(msa),
                    )
                )
            save(args.out / "msa-audit.json", agreement)
            seen["input"] += 1
            return features

        @wraps(infer)
        def replay(*a, **kw):
            if (
                kw.get("seed"),
                kw.get("num_samples"),
                kw.get("num_steps"),
                kw.get("num_recycles"),
            ) != (101, 5, 200, 10):
                raise ValueError("public sampling profile differs from native audit")
            if kw.get("trunk_dtype") is not None:
                raise ValueError("native FP32 must reach the public model boundary")
            kw.update(selected)
            if args.capture_trunk_boundary:
                kw["capture_names"] = ("single_inputs", "single", "pair")
            observer_context = nullcontext()
            if args.capture_consumed_tape:
                from bench.opendde_consumed_tape import observe_consumption

                observer_context = observe_consumption(consumed)
            with observer_context as observed_sources:
                output = infer(*a, **kw)
            if args.capture_consumed_tape:
                audit = consumed.finish()
                audit["source_sha256"] = observed_sources
                audit["native_msa_representation_mapping"] = (
                    "categorical int64 to int32; all-valid bool mask to FP32; "
                    "FP32 deletion fields unchanged; no consumer host compaction"
                )
                save(args.out / "consumed-tape-audit.json", audit)
                if not audit["passed"]:
                    raise ValueError(f"actual consumed tape mismatch: {audit}")
            if args.capture_trunk_boundary:
                boundary = {
                    native: output.pop(port)
                    for native, port in (
                        ("s_inputs", "single_inputs"),
                        ("s_trunk", "single"),
                        ("z_trunk", "pair"),
                    )
                }
                np.savez_compressed(
                    args.out / "trunk-boundary.npz", **jax.device_get(boundary)
                )
            np.savez_compressed(args.out / "raw.npz", **flatten(jax.device_get(output)))
            seen["predict"] += 1
            return output

        @wraps(score)
        def captured_scores(*a, **kw):
            output = score(*a, **kw)
            np.savez_compressed(
                args.out / "scored.npz", **flatten(jax.device_get(output))
            )
            seen["score"] += 1
            return output

        ffi_context = nullcontext()
        ffi_policy = []
        if args.capture_ffi_policy:
            ffi = importlib.import_module("cuequivariance_ops_jax._triangle_attention")
            original_policy = ffi.use_tf32

            def observed_policy(precision, dtype):
                selected_policy = original_policy(precision, dtype)
                record = {
                    "precision": str(precision),
                    "dtype": str(dtype),
                    "use_tf32": selected_policy,
                }
                if record not in ffi_policy:
                    ffi_policy.append(record)
                return selected_policy

            ffi_context = patch.object(ffi, "use_tf32", observed_policy)
        with (
            ffi_context,
            patch.object(cli, "_featurize", checked_features),
            patch.object(cli, "_predict", replay),
            patch.object(cli, "_score", captured_scores),
        ):
            predict(audit_request(args))
        if seen != {"input": 1, "predict": 1, "score": 1}:
            raise RuntimeError(f"incomplete public lifecycle: {seen}")
        provenance["versions"].update(
            jax=jax.__version__, jaxlib=importlib.metadata.version("jaxlib")
        )
        provenance["jax_matmul_precision"] = args.jax_matmul_precision
        provenance["triangle_attention_ffi_trace_attributes"] = ffi_policy
        provenance["versions"].update(
            {
                name: importlib.metadata.version(name)
                for name in ("cuequivariance-jax", "cuequivariance-ops-jax-cu13")
            }
        )
        provenance["checkpoint_sha256"] = sha(
            args.repo / ".foldjax/weights/opendde/opendde.jax"
        )
        provenance["native_capture_sha256"] = sha(args.reference / "provenance.json")
        memory = {"jax_device_memory_stats": jax.devices()[0].memory_stats()}
    if provenance["native_source"] != driver.source_identity(args.native_source):
        raise RuntimeError("native source changed during capture")
    save(args.out / "provenance.json", provenance)
    save(
        args.out / "finished.json",
        {
            "instrumented": True,
            "lifecycle_seconds": time.perf_counter() - started,
            "host_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "memory": memory,
        },
    )
    print(json.dumps({"arm": args.arm, "finished": True}), flush=True)


if __name__ == "__main__":
    main()
