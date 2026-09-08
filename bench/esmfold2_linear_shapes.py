"""Synthetic small/degenerate native BF16 linear controls, not model admission."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from bench.esmfold2_tape import _sha256


def compiler_options(profile):
    if profile == "default":
        return {}
    if profile == "split-k-1":
        return {"xla_gpu_experimental_force_split_k": 1}
    if profile == "no-triton":
        return {"xla_gpu_enable_triton_gemm": False}
    raise ValueError("unknown compiler profile")


def operands():
    rng = np.random.default_rng(123)
    for rows, width, inner in (
        (1, 1, 1024),
        (1, 7, 33),
        (3, 1, 257),
        (2, 3, 1024),
        (17, 16, 31),
        (65, 256, 1024),
        (1, 256, 1024),
        (64, 256, 1024),
    ):
        yield (
            f"m{rows}_n{width}_k{inner}",
            rng.normal(size=(rows, inner)).astype(np.float32),
            rng.normal(size=(width, inner)).astype(np.float32),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("native", "jax"), required=True)
    p.add_argument("--ffi-library", type=Path)
    p.add_argument(
        "--compiler-profile",
        choices=("default", "split-k-1", "no-triton"),
        default="default",
    )
    p.add_argument("--native-reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.engine == "native" and args.compiler_profile != "default":
        raise ValueError("compiler profile is a candidate-only control")
    if args.ffi_library and (
        args.engine != "jax" or args.compiler_profile != "default"
    ):
        raise ValueError("FFI requires the JAX engine and default compiler profile")
    options = compiler_options(args.compiler_profile)
    reference_hash = _sha256(args.native_reference)
    source_hash = _sha256(Path(__file__))
    reference = json.loads(args.native_reference.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    arrays, runtime = {}, {}
    if args.engine == "native":
        import torch
        import torch.nn.functional as functional

        if torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one native GPU required")
        if (str(torch.__version__), torch.version.git_version) != (
            reference["torch"],
            reference["torch_git"],
        ):
            raise ValueError("native runtime identity mismatch")
        torch.set_float32_matmul_precision(reference["matmul"])
        torch.backends.cuda.matmul.allow_tf32 = reference["allow_tf32"]
        reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if reduction != reference["allow_bf16_reduced_precision_reduction"]:
            raise ValueError("native reduction policy mismatch")
        runtime = {
            "torch": str(torch.__version__),
            "torch_git": torch.version.git_version,
            "reduced_bf16": reduction,
        }

        def run(x, w, name):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
                return [functional.linear(a, b).float().cpu().numpy() for _ in range(2)]
    else:
        import jax
        import jax.numpy as jnp

        from foldjax.models.esmfold2.models import trunk

        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise RuntimeError("exactly one candidate GPU required")
        jax.config.update("jax_default_matmul_precision", "highest")
        helper = Path(inspect.getfile(trunk))
        runtime = {"jax": jax.__version__, "helper_sha256": _sha256(helper)}
        def linear_impl(a, b):
            return trunk._autocast_linear(a, {"p.weight": b}, "p")
        ffi_bindings = {}
        if args.ffi_library:
            from bench import native_cublaslt_ffi

            ffi_bindings = {
                str(path.resolve()): _sha256(path)
                for path in (
                    args.ffi_library,
                    Path(native_cublaslt_ffi.__file__),
                    Path(__file__).with_name("native_cublaslt_ffi.cc"),
                )
            }
            target = native_cublaslt_ffi.register(args.ffi_library)
            def linear_impl(a, b):
                return native_cublaslt_ffi.linear(a, b, target=target)
        runtime["ffi_bindings"] = ffi_bindings
        runtime["linear_implementation"] = "ffi" if args.ffi_library else "carried"

        def run(x, w, name):
            fn = jax.jit(
                linear_impl,
                compiler_options=options,
            )
            executable = fn.lower(jnp.asarray(x), jnp.asarray(w)).compile()
            (args.output / f"{name}.hlo.txt").write_text(executable.as_text())
            return [
                np.asarray(
                    executable(jnp.asarray(x), jnp.asarray(w)).astype(jnp.float32)
                )
                for _ in range(2)
            ]

    for name, x, w in operands():
        arrays[name + ".input"], arrays[name + ".weight"] = x, w
        results = run(x, w, name)
        for i, result in enumerate(results):
            if (
                result.shape != (x.shape[0], w.shape[0])
                or not np.isfinite(result).all()
            ):
                raise ValueError("invalid output")
            arrays[f"{name}.output{i}"] = result
    if source_hash != _sha256(Path(__file__)) or reference_hash != _sha256(
        args.native_reference
    ):
        raise ValueError("bound source/reference changed")
    if args.engine == "jax" and runtime["helper_sha256"] != _sha256(helper):
        raise ValueError("candidate helper changed")
    if args.engine == "jax" and any(
        _sha256(Path(p)) != h for p, h in ffi_bindings.items()
    ):
        raise ValueError("FFI artifacts changed")
    with (args.output / "arrays.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "scope": "synthetic same-operand native BF16 linear shape control",
                "model_admission": None,
                "engine": args.engine,
                "runtime": runtime,
                "compiler_profile": args.compiler_profile,
                "compiler_options": options,
                "production_compiler_change": False,
                "benchmark_sha256": source_hash,
                "reference_sha256": reference_hash,
                "archive_sha256": _sha256(args.output / "arrays.npz"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
