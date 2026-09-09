"""Shared-input OpenBind replay; backend deviation is explicit, not admission."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.boltz_historical_replay import digest, save_new, source_hashes
from bench.openbind_tape_adapter import (
    capture_provenance,
    load_capture,
    prepare_core_features,
    required_triangle_kernel,
)


def model_feature_batch(captured):
    """Keep the declared model ABI; native atom-array annotations stay on host."""
    from foldjax.models.openfold3.data.featurize import (
        MODEL_FEATURES,
        OPTIONAL_MODEL_FEATURES,
    )

    missing = set(MODEL_FEATURES) - captured.keys()
    if missing:
        raise ValueError(f"missing model features: {sorted(missing)}")
    return {name: captured[name] for name in (*MODEL_FEATURES, *OPTIONAL_MODEL_FEATURES)
            if name in captured}


def positive_count(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("repeat count must be positive")
    return count


def compiler_environment():
    """Record only compiler controls, never the unrestricted process environment."""
    return {name: os.environ.get(name) for name in (
        "XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
        "JAX_DEFAULT_MATMUL_PRECISION", "XLA_PYTHON_CLIENT_PREALLOCATE",
    )}


def native_trunk_arrays(capture, n_token):
    trace = json.loads((capture / "trace.json").read_text())
    records = trace["trunk_arrays"]
    if len(records) != 1 or records[0]["file"] != "trunk-00.npz":
        raise ValueError("expected exactly one native trunk capture")
    path = capture / records[0]["file"]
    identity = records[0]["sha256"]
    if digest(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
        raise ValueError("native trunk identity mismatch")
    shapes = ((1, n_token, 449), (1, n_token, 384), (1, n_token, n_token, 128))
    with np.load(path, allow_pickle=False) as archive:
        arrays = tuple(archive[str(i)] for i in range(3))
    if any(a.shape != s or a.dtype != np.float32 or not np.isfinite(a).all()
           for a, s in zip(arrays, shapes, strict=True)):
        raise ValueError("invalid native trunk shape, dtype or values")
    return arrays, identity


def candidate_trunk_arrays(candidate, expected, n_token):
    preflight = json.loads((candidate / "preflight.json").read_text())
    expected = json.loads(json.dumps(expected))
    if any(preflight.get(k) != v for k, v in expected.items()):
        raise ValueError("candidate trunk provenance mismatch")
    if (preflight.get("native_trunk_injection")
            or preflight.get("candidate_trunk_injection")):
        raise ValueError("candidate trunk must come from a non-injected run")
    finished = json.loads((candidate / "finished.json").read_text())
    path = candidate / "prediction.npz"
    if digest(path) != finished["prediction_sha256"]:
        raise ValueError("candidate trunk prediction identity mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = tuple(archive[k] for k in ("single_inputs", "single", "pair"))
    shapes = ((1, n_token, 449), (1, n_token, 384), (1, n_token, n_token, 128))
    if any(a.shape != s or a.dtype != np.float32 or not np.isfinite(a).all()
           for a, s in zip(arrays, shapes, strict=True)):
        raise ValueError("invalid candidate trunk shape, dtype or values")
    return arrays, {"prediction_sha256": digest(path),
                    "preflight_sha256": digest(candidate / "preflight.json")}


def replace_trunk_components(candidate, native, components):
    if components not in ("single", "pair"):
        raise ValueError("unknown native trunk components")
    indices = (0, 1) if components == "single" else (2,)
    return tuple(native[i] if i in indices else candidate[i] for i in range(3))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("cueq", "cueq-full", "xla", "native-private"),
        required=True,
    )
    parser.add_argument("--capture-trunk", action="store_true")
    parser.add_argument("--capture-plddt-logits", action="store_true")
    parser.add_argument("--private-pair-operators", action="store_true")
    parser.add_argument("--repeats", type=positive_count, default=1)
    injection = parser.add_mutually_exclusive_group()
    injection.add_argument("--inject-native-trunk", action="store_true")
    injection.add_argument("--inject-candidate-trunk", type=Path)
    parser.add_argument("--native-components", choices=("single", "pair"))
    args = parser.parse_args(argv)
    if args.private_pair_operators and (
        args.backend != "xla" or args.inject_native_trunk
        or args.inject_candidate_trunk is not None
    ):
        parser.error("private pair control requires XLA and no trunk injection")
    if args.native_components and args.inject_candidate_trunk is None:
        parser.error("--native-components requires --inject-candidate-trunk")

    import jax

    from foldjax.models.openfold3.bridge.checkpoint import load_checkpoint
    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table
    from foldjax.models.openfold3.bridge.torch_mapping import (
        map_inference_params,
        prune_sample_diffusion_aliases,
        resolve_model_prefix,
    )
    from foldjax.models.openfold3.inference import compile_predict, released_config

    tape, effective = load_capture(args.capture)
    checkpoint_hash = digest(args.checkpoint)
    provenance = capture_provenance(args.capture, checkpoint_hash)
    with np.load(args.capture / "input.npz", allow_pickle=False) as archive:
        features = prepare_core_features(dict(archive), max_atoms_per_token=23)
    features = tape.prepare_features(model_feature_batch(features))
    config = released_config(
        n_token=features["token_mask"].shape[-1],
        n_atom=features["atom_mask"].shape[-1],
        num_recycles=4, num_samples=5, num_steps=200,
        msa_depth=tape.msa_indices.shape[1], max_array_bytes=None,
        returned_representations=(
            ("single_inputs", "single", "pair") if args.capture_trunk else ()
        ),
        return_plddt_logits=args.capture_plddt_logits,
        per_sample_token_cutoff=effective["settings"]["memory"]["eval"][
            "per_sample_token_cutoff"
        ],
    )
    source = source_hashes(args.source_root)
    injected, injected_identity = None, None
    if args.inject_native_trunk:
        injected, injected_identity = native_trunk_arrays(args.capture, config.n_token)
    candidate_identity = None
    if args.inject_candidate_trunk is not None:
        injected, candidate_identity = candidate_trunk_arrays(
            args.inject_candidate_trunk, {
                "input_sha256": digest(args.capture / "input.npz"),
                "tape_sha256": digest(args.capture / "tape.npz"),
                "checkpoint_sha256": checkpoint_hash,
                "source": source, "config": config._asdict(),
                "candidate_backend": args.backend,
            }, config.n_token,
        )
        if args.native_components:
            native_arrays, injected_identity = native_trunk_arrays(
                args.capture, config.n_token
            )
            injected = replace_trunk_components(
                injected, native_arrays, args.native_components
            )
    compiler_controls = compiler_environment()
    harness_paths = [
        Path(__file__), Path(__file__).with_name("openbind_tape_adapter.py")
    ]
    if args.private_pair_operators:
        harness_paths.append(Path(__file__).with_name("openbind_real_pair_replay.py"))
    harness = {path.name: digest(path) for path in harness_paths}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    save_new(args.out_dir / "preflight.json", {
        "scope": (
            "trunk injection; diagnostic only, never ordinary execution"
            if injected is not None else
            "shared native inputs; no intermediate injection; diagnostic only"
        ),
        "native_trunk_injection": injected_identity,
        "candidate_trunk_injection": candidate_identity,
        "native_trunk_components": args.native_components,
        "native_backend": required_triangle_kernel(effective),
        "candidate_backend": args.backend,
        "private_pair_operators": args.private_pair_operators,
        "capture_provenance": provenance,
        "checkpoint_sha256": checkpoint_hash,
        "source": source,
        "harness_sha256": harness,
        "compiler_environment": compiler_controls,
        "requested_calls": args.repeats,
        "config": config._asdict(),
        "input_sha256": digest(args.capture / "input.npz"),
        "tape_sha256": digest(args.capture / "tape.npz"),
        "not_verified": ["independent preprocessing", "device tape consumption",
                         "native tuned chunks", "warm performance"],
    })
    state = load_checkpoint(args.checkpoint)
    prefix = resolve_model_prefix(state, None)
    prune_sample_diffusion_aliases(state, prefix=prefix)
    params = map_inference_params(state, prefix)
    injection_context = (
        patch(
            "foldjax.models.openfold3.inference.trunk",
            side_effect=lambda *a, **kw: tuple(jax.numpy.asarray(x) for x in injected),
        )
        if injected is not None else nullcontext()
    )
    # The patch is needed during the first JIT trace, not subsequent executions.
    pair_context = nullcontext()
    if args.private_pair_operators:
        from bench.openbind_real_pair_replay import private_pair_operators

        pair_context = private_pair_operators()
    with injection_context, pair_context:
        run = compile_predict(config, representative_atom_table(),
                              triangle_kernel=args.backend)
        started = time.perf_counter()
        result = run(jax.random.key(101), features, params, noise_tape=tape.noise,
                     augmentation_tape=tape.augmentation())
    result = jax.device_get(result)
    elapsed = time.perf_counter() - started
    arrays = {k: v for k, v in result._asdict().items() if v is not None}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite prediction")
    with (args.out_dir / "prediction.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    repeats = []
    for index in range(1, args.repeats):
        repeated = jax.device_get(run(
            jax.random.key(101), features, params, noise_tape=tape.noise,
            augmentation_tape=tape.augmentation(),
        ))
        repeated_arrays = {
            k: v for k, v in repeated._asdict().items() if v is not None
        }
        if not all(np.isfinite(v).all() for v in repeated_arrays.values()):
            raise ValueError("nonfinite repeated prediction")
        path = args.out_dir / f"prediction-repeat-{index}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **repeated_arrays)
        repeats.append({
            "file": path.name, "sha256": digest(path),
            "array_equal_to_first": {
                k: bool(np.array_equal(arrays[k], v))
                for k, v in repeated_arrays.items()
            },
        })
    if source_hashes(args.source_root) != source:
        raise RuntimeError("candidate source changed during replay")
    if {path.name: digest(path) for path in harness_paths} != harness:
        raise RuntimeError("candidate replay harness changed during replay")
    if compiler_environment() != compiler_controls:
        raise RuntimeError("compiler environment changed during replay")
    save_new(args.out_dir / "finished.json", {
        "compile_and_infer_seconds": elapsed,
        "prediction_sha256": digest(args.out_dir / "prediction.npz"),
        "same_callable_repeats": repeats,
        "parity_admitted": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
