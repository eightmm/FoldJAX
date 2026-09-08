"""Same-operand atom feature projection diagnostic, not model admission."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from bench.af3_closure_capture import sha
from bench.boltz_amp_report import verify_capture
from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison, profiled_call
from bench.boltz_relpos_probe import torch_policy


def atom_features(features):
    pos = features["ref_pos"]
    return np.concatenate([
        pos, features["ref_charge"][..., None], features["ref_element"],
        features["ref_atom_name_chars"].reshape(*pos.shape[:2], 256),
    ], axis=-1).astype(np.float32)


def compile_profiles(backend_controls=False):
    profiles = {"baseline": {"xla_allow_excess_precision": False}}
    if backend_controls:
        for name, extra in {
            "split_k_1": {"xla_gpu_experimental_force_split_k": 1},
            "no_triton": {"xla_gpu_enable_triton_gemm": False},
            "no_triton_no_lt": {
                "xla_gpu_enable_triton_gemm": False,
                "xla_gpu_enable_cublaslt": False,
            },
        }.items():
            profiles[name] = {**profiles["baseline"], **extra}
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend-controls", action="store_true")
    parser.add_argument("--ffi-library", type=Path)
    args = parser.parse_args()
    if args.backend_controls and args.arm != "foldjax":
        parser.error("backend controls require foldjax arm")
    if args.ffi_library and args.arm != "foldjax":
        parser.error("FFI requires foldjax arm")
    root = args.reference.resolve(strict=True)
    verify_capture(root, foldjax=False)
    paths = [root / "features.npz", args.weights.resolve(strict=True),
             root / "trunk-boundaries/input_embedder.atom_encoder.npz",
             Path(__file__).resolve()]
    if args.ffi_library:
        paths.extend([
            args.ffi_library.resolve(strict=True),
            Path(__file__).with_name("native_cublaslt_ffi.py").resolve(),
            Path(__file__).with_name("native_cublaslt_ffi.cc").resolve(),
        ])
    bindings = {str(p): sha(p) for p in paths}
    with np.load(paths[0], allow_pickle=False) as features:
        x = atom_features(features)
    with safe_open(paths[1], framework="numpy") as weights:
        prefix = "d:trunk/d:input_embedder/d:atom_encoder/d:embed_atom_features/d:"
        w, b = (weights.get_tensor(prefix + k) for k in ("kernel", "bias"))
    with np.load(paths[2], allow_pickle=False) as reference:
        target = reference["1"]
    if (x.shape, w.shape, b.shape, target.shape) != (
        (1, 3104, 388), (388, 128), (128,), (1, 3104, 128)
    ) or any(a.dtype != np.float32 or not np.isfinite(a).all()
             for a in (x, w, b, target)):
        raise ValueError("requires finite FP32 captured 5SAK projection")
    results = {}
    if args.arm == "native":
        import torch

        xx, ww, bb = (torch.from_numpy(a.copy()).cuda() for a in (x, w.T, b))
        transposed_storage = ww.T.contiguous().T
        calls = {
            "linear": lambda: torch.nn.functional.linear(xx, ww, bb),
            "matmul_then_bias": lambda: xx @ ww.T + bb,
            "linear_transposed_storage": lambda: torch.nn.functional.linear(
                xx, transposed_storage, bb
            ),
        }
        with torch_policy(torch, precision="highest"), torch.inference_mode():
            with torch.autocast("cuda", enabled=False):
                for name, fn in calls.items():
                    fn()
                    output, profile = profiled_call(torch, fn)
                    repeat = fn()
                    actual = output.cpu().numpy()
                    results[name] = {
                        "comparison": bitwise_comparison(actual, target),
                        "repeat": bitwise_comparison(actual, repeat.cpu().numpy()),
                        "profile": profile,
                    }
        runtime = {
            "torch": torch.__version__, "device": torch.cuda.get_device_name(),
            "weight_strides": {
                "original": list(ww.stride()),
                "transposed_storage": list(transposed_storage.stride()),
            },
            "weight_values_equal": bool(torch.equal(ww, transposed_storage)),
        }
    else:
        import jax
        import jax.numpy as jnp
        from jax.experimental.layout import Layout, with_layout_constraint

        from foldjax.models.boltz2.models.primitives._common import linear

        with jax.default_matmul_precision("highest"):
            operands = tuple(jnp.asarray(a) for a in (x, w, b))
            def separated_projection(x, w, b):
                return jax.lax.optimization_barrier(x @ w) + b

            def native_storage_projection(x, w, b):
                # The managed kernel is [input, output]; native Linear stores
                # [output, input] contiguously, hence input must be minor here.
                w = with_layout_constraint(w, Layout((1, 0)))
                return linear(x, w, b)

            def transposed_gemm(x, w, b):
                return (w.T @ x.reshape(-1, x.shape[-1]).T).T.reshape(
                    *x.shape[:-1], w.shape[-1]
                ) + b

            calls = {
                "linear": linear, "barrier_then_bias": separated_projection,
                "native_storage": native_storage_projection,
                "transposed_gemm": transposed_gemm,
            }
            if args.ffi_library:
                from bench.native_cublaslt_ffi import linear as ffi_linear
                from bench.native_cublaslt_ffi import register

                target_name = register(args.ffi_library, fp32=True)

                def native_ffi(x, w, b):
                    return ffi_linear(x, w.T, target=target_name, fp32=True) + b

                calls["native_ffi"] = native_ffi
            for name, function in calls.items():
                for profile, options in compile_profiles(args.backend_controls).items():
                    executable = jax.jit(function, compiler_options=options).lower(
                        *operands
                    ).compile()
                    actual = np.asarray(executable(*operands))
                    repeat = np.asarray(executable(*operands))
                    results[f"{name}/{profile}"] = {
                        "comparison": bitwise_comparison(actual, target),
                        "repeat": bitwise_comparison(actual, repeat),
                        "compiler_options": options,
                        "lowering_lines": [
                            line.strip() for line in executable.as_text().splitlines()
                            if any(token in line for token in (
                                "custom_call_target=", " dot(", " reduce(",
                                " parameter(", " copy(", " bitcast(", " transpose(",
                            ))
                        ],
                    }
        runtime = {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()]}
    if any(sha(Path(p)) != digest for p, digest in bindings.items()):
        raise ValueError("bound inputs changed during probe")
    save_new(args.out, {
        "arm": args.arm, "results": results, "runtime": runtime,
        "bindings": bindings, "bindings_unchanged": True,
        "scope": (
            "managed weights and captured native features; not weight conversion proof"
        ),
        "not_model_parity_admission": True,
    })
    print(json.dumps(results))


if __name__ == "__main__":
    main()
