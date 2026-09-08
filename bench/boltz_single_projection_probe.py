"""Native-operand single projections, separated versus production concatenation."""

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open

from bench.af3_closure_capture import sha
from bench.boltz_amp_report import verify_capture
from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "weights", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ffi-library", type=Path)
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    verify_capture(root, foldjax=False)
    if args.out.exists():
        raise FileExistsError(args.out)
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives import _common

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("requires one GPU")
    prefix = "d:trunk/d:pairformer_module/d:layers/i:0/d:attention/d:proj_"
    with safe_open(args.weights, framework="numpy") as handle:
        weights = {k: handle.get_tensor(prefix + k + "/d:kernel") for k in "qkvg"}
        bias = handle.get_tensor(prefix + "q/d:bias")
    paths = sorted(root.glob(
        "trunk-boundaries/cycle-*/pairformer_module.layers.0.attention.proj_*.npz"
    ))
    if len(paths) != 8:
        raise ValueError("requires four projections at first and last recycle")
    bindings = {str(p): sha(p) for p in (
        *paths, args.weights.resolve(strict=True), Path(__file__).resolve(),
        Path(_common.__file__).resolve(),
    )}
    target = None
    if args.ffi_library is not None:
        from bench import native_cublaslt_ffi as ffi

        library = args.ffi_library.resolve(strict=True)
        bindings[str(library)] = sha(library)
        bindings[str(Path(ffi.__file__).resolve())] = sha(Path(ffi.__file__))
        target = ffi.register(library, fp32=True)
    results = {}
    for cycle in sorted({p.parent for p in paths}):
        inputs, expected = {}, {}
        for key in "qkvg":
            path = cycle / f"pairformer_module.layers.0.attention.proj_{key}.npz"
            with np.load(path, allow_pickle=False) as archive:
                inputs[key], expected[key] = archive["input"], archive["output"]
        arrays = [*inputs.values(), *expected.values(), *weights.values(), bias]
        if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays):
            raise ValueError("requires finite FP32 inputs, outputs and weights")
        if not all(np.array_equal(inputs["q"], inputs[k]) for k in "kvg"):
            raise ValueError("native QKVG operands differ")

        def separated(x, w, b):
            return tuple(_common.linear(x, w[k], b if k == "q" else None)
                         for k in "qkvg")

        def combined(x, w, b):
            q, g = jnp.split(_common.linear(
                x, jnp.concatenate((w["q"], w["g"]), axis=-1)), 2, axis=-1)
            k, v = jnp.split(_common.linear(
                x, jnp.concatenate((w["k"], w["v"]), axis=-1)), 2, axis=-1)
            return q + b, k, v, g

        def native_gemm(x, w, b):
            outputs = tuple(ffi.linear(x, w[k].T, target=target, fp32=True)
                            for k in "qkvg")
            return outputs[0] + b, *outputs[1:]

        operands = (jnp.asarray(inputs["q"]),
                    {k: jnp.asarray(v) for k, v in weights.items()}, jnp.asarray(bias))
        with jax.default_matmul_precision("highest"):
            calls = [("separated", separated), ("combined", combined)]
            if target is not None:
                calls.append(("native_gemm", native_gemm))
            arm_outputs = {}
            for arm, call in calls:
                fn = jax.jit(
                    call, compiler_options={"xla_allow_excess_precision": False}
                )
                actual, repeat = fn(*operands), fn(*operands)
                arm_outputs[arm] = tuple(np.asarray(a) for a in actual)
                for key, a, r in zip("qkvg", actual, repeat, strict=True):
                    results[f"{cycle.name}/{arm}/{key}"] = {
                        "comparison": bitwise_comparison(np.asarray(a), expected[key]),
                        "repeat": bitwise_comparison(np.asarray(a), np.asarray(r)),
                    }
            for key, a, b in zip("qkvg", arm_outputs["separated"],
                                 arm_outputs["combined"], strict=True):
                results[f"{cycle.name}/separated_vs_combined/{key}"] = (
                    bitwise_comparison(a, b)
                )
    if any(sha(Path(p)) != value for p, value in bindings.items()):
        raise ValueError("bound inputs changed")
    save_new(args.out, {"results": results, "bindings": bindings,
                        "not_model_parity_admission": True})


if __name__ == "__main__":
    main()
