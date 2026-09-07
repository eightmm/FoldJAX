"""Substituted native-trunk Boltz sampler counterfactual, NOT model parity.

Both arms skip the FoldJAX trunk and confidence heads. The optional native
conditioning arm also substitutes q/c and all three materialized bias tensors.
Captured arrays and the complete sampler tape enter as dynamic JIT arguments;
no publisher callback is imported or executed. This is not preprocessing,
weight-conversion, performance, or all-input admission evidence.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_amp_report import compare_coordinates
from bench.boltz_closure_capture import load_legacy, save_new
from bench.boltz_foldjax_capture import array_identity, load_inputs, native_settings
from bench.protenix_foldjax_capture import save_jax_boundary

_TRUNK = ("s", "z", "s_inputs", "relative_position_encoding")
_CONDITIONING = ("q", "c", "atom_enc_bias", "atom_dec_bias", "token_trans_bias")
_CALLBACK = {
    "to_keys.native_type": "functools.partial",
    "to_keys.function": "boltz.model.modules.encodersv2.single_to_keys",
    "to_keys.positional_argument_count": 0,
    "to_keys.keywords.W": 32,
    "to_keys.keywords.H": 128,
}


def bound_arrays(root, label, names=None):
    complete = json.loads((root / "capture-complete.json").read_text())
    if complete.get("passed") is not True:
        raise ValueError("native capture is incomplete")
    binding = complete["artifacts"][label]
    path, tree_path = root / f"{label}.npz", root / f"{label}.tree.json"
    if (
        sha(path) != binding["arrays_sha256"]
        or sha(tree_path) != binding["tree_sha256"]
    ):
        raise ValueError(f"native artifact hash mismatch: {label}")
    tree = json.loads(tree_path.read_text())
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        if available != {
            key for key, info in tree.items() if info.get("kind") != "none"
        }:
            raise ValueError(f"native archive/tree schema mismatch: {label}")
        if names is not None and not set(names).issubset(available):
            raise ValueError(f"missing native leaves: {label}")
        arrays = {key: archive[key] for key in (available if names is None else names)}
    for key, value in arrays.items():
        if (
            list(value.shape) != tree[key]["shape"]
            or str(value.dtype) != tree[key]["storage_dtype"]
        ):
            raise ValueError(f"native array metadata mismatch: {label}/{key}")
    return arrays, {key: tree[key] for key in arrays}


def finite_fp32(value, name, shape):
    if (
        value.dtype != np.float32
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"invalid native FP32 storage at {name}: expected {shape}")


def load_trunk(root, meta):
    """Cross-bind the legacy sampler archive to observed native module outputs."""
    n = int(meta["n_token"])
    if n < 1:
        raise ValueError("native token count must be positive")
    with np.load(root / "trunk.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(_TRUNK):
            raise ValueError("native sampler trunk is missing or has unknown leaves")
        trunk = {key: archive[key] for key in _TRUNK}
    forward, forward_tree = bound_arrays(root, "forward-output", ("s", "z"))
    embedder, embedder_tree = bound_arrays(
        root, "trunk-boundaries/input_embedder", ("",)
    )
    relpos, relpos_tree = bound_arrays(root, "trunk-boundaries/rel_pos", ("",))
    references = {
        **forward,
        "s_inputs": embedder[""],
        "relative_position_encoding": relpos[""],
    }
    metadata = {
        **forward_tree,
        "s_inputs": embedder_tree[""],
        "relative_position_encoding": relpos_tree[""],
    }
    for key, value in trunk.items():
        shape = (1, n, 384) if key in {"s", "s_inputs"} else (1, n, n, 128)
        finite_fp32(value, key, shape)
        if not np.array_equal(value, references[key]):
            raise ValueError(
                f"legacy trunk differs from bound native observation: {key}"
            )
    return trunk, metadata


def load_conditioning(root, meta, *, native_indexing):
    arrays, tree = bound_arrays(root, "trunk-boundaries/diffusion_conditioning")
    expected = (
        set(_CONDITIONING)
        | set(_CALLBACK)
        | {"to_keys.function_source_sha256", "to_keys.keywords.indexing_matrix"}
    )
    if set(arrays) != expected:
        raise ValueError("unknown native conditioning/callback schema")
    for name, value in _CALLBACK.items():
        actual = arrays[name]
        if actual.shape != () or actual.item() != value:
            raise ValueError(f"unmapped native callback metadata: {name}")
        if isinstance(value, int) and actual.dtype.kind not in "iu":
            raise ValueError(f"noninteger native callback metadata: {name}")
    provenance = json.loads((root / "provenance.json").read_text())
    expected_source = provenance["upstream_python_source"][
        "src/boltz/model/modules/encodersv2.py"
    ]
    if arrays["to_keys.function_source_sha256"].item() != expected_source:
        raise ValueError("native callback source is not bound to captured upstream")
    atoms, tokens = int(meta["n_atom"]), int(meta["n_token"])
    if atoms < 32 or atoms % 32:
        raise ValueError("native conditioning requires complete 32-atom windows")
    shapes = {
        "q": (1, atoms, 128),
        "c": (1, atoms, 128),
        "atom_enc_bias": (1, atoms // 32, 32, 128, 12),
        "atom_dec_bias": (1, atoms // 32, 32, 128, 12),
        "token_trans_bias": (1, tokens, tokens, 384),
    }
    payload = {}
    for name in _CONDITIONING:
        value = arrays[name]
        finite_fp32(value, name, shapes[name])
        dtype = "torch.float32" if name in {"q", "c"} else "torch.bfloat16"
        if tree[name]["native_dtype"] != dtype:
            raise ValueError(f"unexpected original native conditioning dtype: {name}")
        if dtype == "torch.bfloat16" and np.any(value.view(np.uint32) & 0xFFFF):
            raise ValueError(f"native BF16 storage contains unrounded values: {name}")
        payload[name] = value
    matrix = arrays["to_keys.keywords.indexing_matrix"]
    finite_fp32(matrix, "indexing_matrix", (atoms // 16, atoms // 4))
    if not np.array_equal(matrix, native_indexing):
        raise ValueError("captured native key-window matrix differs from JAX mapper")
    payload["indexing_matrix"] = matrix
    return payload, tree


def restore_conditioning(payload):
    import jax.numpy as jnp

    return {
        key: jnp.asarray(
            value, dtype=jnp.bfloat16 if key.endswith("bias") else jnp.float32
        )
        for key, value in payload.items()
    }


def sampler_options(meta, effective, *, materialized_token_bias=False):
    return {
        "recycling_steps": int(meta["num_recycles"]),
        "num_sampling_steps": int(meta["num_steps"]),
        "multiplicity": int(meta["num_samples"]),
        **{
            name: float(meta[name])
            for name in (
                "step_scale",
                "gamma_0",
                "gamma_min",
                "noise_scale",
                "sigma_data",
            )
        },
        "sigma_min": 0.0001,
        "sigma_max": 160.0,
        "rho": 7.0,
        "compute_dtype": "bfloat16",
        "matmul_precision": "highest",
        "attention_backend": "xla",
        "triangle_backend": "cueq",
        "glu_backend": "xla",
        "chunk_size": 128,
        "augmentation": True,
        "alignment_reverse_diff": True,
        "use_scan": True,
        "trunk_use_scan": True,
        "score_use_scan": True,
        "steering_args": effective["steering_args"],
        "lazy_token_trans_bias": not materialized_token_bias,
    }


def make_sampler(
    module,
    options,
    *,
    native_conditioning,
    single_to_keys,
    conditioning_norm_control="production",
):
    """Patches only trace-time dispatch; every substituted array is a runtime input."""

    def run(params, features, key, tape, native_trunk, conditioning):
        def schedule(num_steps, **kwargs):
            if tape["sigmas"].shape != (num_steps + 1,):
                raise ValueError(
                    "captured schedule length differs from requested steps"
                )
            return tape["sigmas"]

        def substitute(*args, **kwargs):
            def to_keys(value):
                return single_to_keys(
                    value, conditioning["indexing_matrix"], w=32, h_keys=128
                )

            return {
                **{name: conditioning[name] for name in _CONDITIONING},
                "to_keys": to_keys,
            }

        def forbidden_trunk(*args, **kwargs):
            raise RuntimeError("downstream counterfactual must never compute the trunk")

        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "_sample_schedule", schedule))
            stack.enter_context(
                patch.object(module, "boltz2_trunk_forward", forbidden_trunk)
            )
            if native_conditioning:
                stack.enter_context(
                    patch.object(module, "diffusion_conditioning_forward", substitute)
                )
            elif conditioning_norm_control == "explicit_fma":
                from bench.boltz_native_layer_norm import conditioning_override

                stack.enter_context(
                    patch.object(
                        module,
                        "diffusion_conditioning_forward",
                        conditioning_override(module.diffusion_conditioning_forward),
                    )
                )
            elif conditioning_norm_control != "production":
                raise ValueError("unknown conditioning norm control")
            return module.boltz2_sample_forward(
                params,
                features,
                key,
                trunk=native_trunk,
                init_noise=tape["init_noise"],
                step_noises=tape["step_noises"],
                aug_transforms=(tape["rotations"], tape["translations"]),
                **options,
            )

    return run


def source_identity(source):
    return {
        str(path.relative_to(source)): sha(path)
        for directory in ("src", "bench")
        for path in sorted((source / directory).rglob("*.py"))
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--upstream-capture", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--conditioning-source", choices=("foldjax", "native"), default="foldjax"
    )
    parser.add_argument(
        "--materialized-token-bias",
        action="store_true",
        help="Keep FoldJAX conditioning but precompute token bias, as native does",
    )
    parser.add_argument(
        "--conditioning-norm-control",
        choices=("production", "explicit_fma"),
        default="production",
    )
    args = parser.parse_args(argv)
    if args.conditioning_norm_control != "production" and (
        args.conditioning_source != "foldjax" or not args.materialized_token_bias
    ):
        parser.error(
            "explicit-FMA control requires FoldJAX and materialized token bias"
        )
    if args.out_dir.exists():
        raise FileExistsError("--out-dir must be new")
    source, native = (
        args.source_root.resolve(strict=True),
        args.upstream_capture.resolve(strict=True),
    )
    if Path(__file__).resolve() != source / "bench/boltz_downstream_probe.py":
        raise ValueError("execute from the explicitly selected source root")
    meta, effective = native_settings(native)
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.compile_policy import compiler_options, jit
    from foldjax.models.boltz2.models.diffusion.atom import (
        get_indexing_matrix,
        single_to_keys,
    )
    from foldjax.models.boltz2.weights import resolve_native_weight_bundle

    module = importlib.import_module("foldjax.models.boltz2.models.trunk_blocks.trunk")
    if (
        not Path(inspect.getfile(module.boltz2_sample_forward))
        .resolve()
        .is_relative_to(source / "src")
    ):
        raise ValueError("imported sampler source differs from selected source root")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("this sampler counterfactual requires exactly one GPU")
    bundle = resolve_native_weight_bundle(args.weights)
    if bundle is None:
        raise FileNotFoundError("native FoldJAX weight bundle not found")
    legacy_path = source / "tests/models/boltz2/scripts/parity_matched_tape.py"
    legacy = load_legacy(legacy_path)
    features, tape = load_inputs(legacy, native, meta)
    native_trunk, trunk_tree = load_trunk(native, meta)
    native_conditioning = args.conditioning_source == "native"
    conditioning, conditioning_tree = {}, {}
    if native_conditioning:
        matrix = np.asarray(get_indexing_matrix(int(meta["n_atom"]) // 32, 32, 128))
        conditioning, conditioning_tree = load_conditioning(
            native, meta, native_indexing=matrix
        )
    reference, _ = bound_arrays(native, "forward-output", ("sample_atom_coords",))
    with np.load(native / "features.npz", allow_pickle=False) as archive:
        coordinate_features = {
            name: archive[name]
            for name in (
                "atom_pad_mask",
                "token_pad_mask",
                "atom_to_token",
                "entity_id",
                "mol_type",
                "asym_id",
            )
        }
    compare_coordinates(
        reference["sample_atom_coords"],
        reference["sample_atom_coords"],
        coordinate_features,
    )
    filenames = [
        "capture-complete.json",
        "provenance.json",
        "features.npz",
        "tape.npz",
        "tape.json",
        "effective-model-settings.json",
        "trunk.npz",
        "forward-output.npz",
        "forward-output.tree.json",
        "trunk-boundaries/input_embedder.npz",
        "trunk-boundaries/input_embedder.tree.json",
        "trunk-boundaries/rel_pos.npz",
        "trunk-boundaries/rel_pos.tree.json",
    ]
    if native_conditioning:
        filenames += [
            "trunk-boundaries/diffusion_conditioning.npz",
            "trunk-boundaries/diffusion_conditioning.tree.json",
        ]
    hashes = {name: sha(native / name) for name in filenames}
    sources = source_identity(source)
    weight_hashes = {
        "weights_sha256": sha(bundle[0]),
        "sidecar_sha256": sha(bundle[1]) if bundle[1].exists() else None,
    }
    options = sampler_options(
        meta,
        effective,
        materialized_token_bias=native_conditioning or args.materialized_token_bias,
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    device_conditioning = restore_conditioning(conditioning)
    policy = {
        **options,
        "compiler_options": compiler_options("bfloat16"),
        "conditioning_source": args.conditioning_source,
        "conditioning_norm_control": args.conditioning_norm_control,
        "native_trunk_entry": {
            name: array_identity(value) for name, value in native_trunk.items()
        },
        "native_trunk_original_metadata": trunk_tree,
        "native_conditioning_entry": {
            name: array_identity(value) for name, value in device_conditioning.items()
        },
        "native_conditioning_original_metadata": conditioning_tree,
        "native_conditioning_consumption": (
            "full materialized token bias; no lazy projection"
            if native_conditioning or args.materialized_token_bias
            else "FoldJAX lazy token bias"
        ),
        "sampler_tape": {name: array_identity(value) for name, value in tape.items()},
        "features": {name: array_identity(value) for name, value in features.items()},
    }
    save_new(args.out_dir / "effective-options.json", policy)
    save_new(
        args.out_dir / "provenance.json",
        {
            "scope": __doc__,
            "not_parity_admission": True,
            "substituted_diagnostic": True,
            "substitutions": ["native_trunk"]
            + (["native_diffusion_conditioning"] if native_conditioning else []),
            "source_files": sources,
            "native_artifacts": hashes,
            **weight_hashes,
            "legacy_loader_sha256": sha(legacy_path),
            "jax_version": jax.__version__,
            "device_kind": jax.devices()[0].device_kind,
            "output_order": "original sample index 0..4; no rank rematching",
            "tape_scope": (
                "dynamic sampler entry arrays; not device-consumer observation"
            ),
            "callback_mapping": (
                "JAX single_to_keys with validated captured native matrix; "
                "publisher callback never executed"
            ),
        },
    )
    params = load_params(args.weights)
    if params.get("affinity", {}).get("modules"):
        raise ValueError("counterfactual requires confidence-only weights")
    started = time.monotonic()
    jax.config.update("jax_compilation_cache_dir", str(args.out_dir / "jax-cache"))
    with (
        jax.default_matmul_precision("highest"),
        patch.dict(os.environ, {"BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND": "cueq"}),
    ):
        run = jit(
            make_sampler(
                module,
                options,
                native_conditioning=native_conditioning,
                single_to_keys=single_to_keys,
                conditioning_norm_control=args.conditioning_norm_control,
            ),
            compute_dtype="bfloat16",
        )
        output = run(
            params,
            features,
            jax.random.PRNGKey(int(meta["seed"])),
            jax.tree.map(jnp.asarray, tape),
            legacy.captured_sampler_trunk(native_trunk),
            device_conditioning,
        )
        jax.block_until_ready(output)
    comparison = compare_coordinates(
        reference["sample_atom_coords"],
        np.asarray(output["sample_atom_coords"]),
        coordinate_features,
    )
    if sources != source_identity(source) or hashes != {
        name: sha(native / name) for name in filenames
    }:
        raise RuntimeError(
            "source or native input artifacts changed during counterfactual"
        )
    if weight_hashes != {
        "weights_sha256": sha(bundle[0]),
        "sidecar_sha256": sha(bundle[1]) if bundle[1].exists() else None,
    }:
        raise RuntimeError("weight bundle changed during counterfactual")
    output_path = args.out_dir / "prediction.npz"
    save_jax_boundary(output_path, output)
    save_new(
        args.out_dir / "comparison.json",
        {
            "not_parity_admission": True,
            "substituted_diagnostic": True,
            "coordinates": comparison,
        },
    )
    save_new(
        args.out_dir / "capture-complete.json",
        {
            "passed": True,
            "not_parity_admission": True,
            "substituted_diagnostic": True,
            "trunk_executed": False,
            "confidence_executed": False,
            "native_callback_executed": False,
            "diagnostic_seconds": time.monotonic() - started,
            "artifacts": {
                name: sha(args.out_dir / name)
                for name in (
                    "prediction.npz",
                    "prediction.tree.json",
                    "comparison.json",
                    "provenance.json",
                    "effective-options.json",
                )
            },
        },
    )


if __name__ == "__main__":
    main()
