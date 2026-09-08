"""Same-operand FP32 atom-pair module diagnostic, not full-model admission."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from bench.af3_closure_capture import sha
from bench.boltz_amp_report import verify_capture
from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison

NAMES = (
    "embed_atompair_ref_pos", "embed_atompair_ref_dist", "embed_atompair_mask",
    "c_to_p_trans_q", "c_to_p_trans_k", "p_mlp",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "weights", "ffi-library", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    verify_capture(root, foldjax=False)
    paths = [args.weights.resolve(strict=True), args.ffi_library.resolve(strict=True),
             Path(__file__).resolve(),
             Path(__file__).with_name("native_cublaslt_ffi.py").resolve(),
             Path(__file__).with_name("native_cublaslt_ffi.cc").resolve()]
    captures = {name: root / f"trunk-boundaries/input_embedder.atom_encoder.{name}.npz"
                for name in NAMES}
    bindings = {str(p): sha(p) for p in [*paths, *captures.values()]}

    import jax
    import jax.numpy as jnp

    from bench.native_cublaslt_ffi import linear as ffi_linear
    from bench.native_cublaslt_ffi import register
    from foldjax.models.boltz2.models.primitives._common import linear

    if len(jax.devices()) != 1 or jax.default_backend() != "gpu":
        raise RuntimeError("requires one GPU")
    target = register(args.ffi_library, fp32=True)
    from foldjax.models.boltz2.models.primitives.native_atom_geometry import (
        inverse_squared_distance,
    )
    results = {}
    with np.load(captures["embed_atompair_ref_pos"], allow_pickle=False) as archive:
        displacement = archive["input"]
    with np.load(captures["embed_atompair_ref_dist"], allow_pickle=False) as archive:
        distance_target = archive["input"]

    def ordinary_distance(d):
        return 1.0 / (1.0 + jnp.sum(d * d, axis=-1, keepdims=True))

    def separated_distance(d, explicit_division=False, order=(0, 1, 2)):
        squares = jax.lax.optimization_barrier(d * d)
        partial = jax.lax.optimization_barrier(
            squares[..., order[0]] + squares[..., order[1]]
        )
        total = jax.lax.optimization_barrier(partial + squares[..., order[2]])
        denominator = jax.lax.optimization_barrier(1.0 + total)[..., None]
        if explicit_division:
            from foldjax.models.boltz2.models.primitives.native_pwa_weights import (
                _cuda_rn_divide,
            )
            return _cuda_rn_divide(jnp.ones_like(denominator), denominator)
        return 1.0 / denominator

    for name, call in (
        ("distance/ordinary", ordinary_distance),
        ("distance/production", inverse_squared_distance),
        ("distance/separated", separated_distance),
        ("distance/rn_divide", lambda d: separated_distance(d, True)),
        ("distance/tree_021", lambda d: separated_distance(d, order=(0, 2, 1))),
        ("distance/tree_120", lambda d: separated_distance(d, order=(1, 2, 0))),
        ("distance/tree_021_rn", lambda d: separated_distance(d, True, (0, 2, 1))),
        ("distance/tree_120_rn", lambda d: separated_distance(d, True, (1, 2, 0))),
    ):
        fn = jax.jit(call, compiler_options={"xla_allow_excess_precision": False})
        actual = np.asarray(fn(jnp.asarray(displacement)))
        repeat = np.asarray(fn(jnp.asarray(displacement)))
        results[name] = {"comparison": bitwise_comparison(actual, distance_target),
                         "repeat": bitwise_comparison(actual, repeat)}
    with safe_open(paths[0], framework="numpy") as handle:
        for name, path in captures.items():
            with np.load(path, allow_pickle=False) as archive:
                x, expected = archive["input"], archive["output"]
            prefix = f"d:trunk/d:input_embedder/d:atom_encoder/d:{name}/"
            suffixes = [f"i:{i}/d:kernel" for i in range(3)] if name == "p_mlp" else [
                "d:kernel"
            ]
            weights = [handle.get_tensor(prefix + key) for key in suffixes]
            if any(a.dtype != np.float32 or not np.isfinite(a).all()
                   for a in (x, expected, *weights)):
                raise ValueError("requires finite FP32 native operands")
            for arm in ("jax", "ffi"):
                def call(x, weights):
                    for w in weights:
                        if name == "p_mlp" or name.startswith("c_to_p"):
                            x = jax.nn.relu(x)
                        x = (linear(x, w) if arm == "jax" else
                             ffi_linear(x, w.T, target=target, fp32=True))
                    return x

                with jax.default_matmul_precision("highest"):
                    fn = jax.jit(
                        call, compiler_options={"xla_allow_excess_precision": False}
                    )
                    operands = (jnp.asarray(x), tuple(jnp.asarray(w) for w in weights))
                    actual = np.asarray(fn(*operands))
                    repeat = np.asarray(fn(*operands))
                results[f"{name}/{arm}"] = {
                    "comparison": bitwise_comparison(actual, expected),
                    "repeat": bitwise_comparison(actual, repeat),
                }
    if any(sha(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("bound inputs changed")
    save_new(args.out, {"results": results, "bindings": bindings,
                        "bindings_unchanged": True, "not_model_parity_admission": True})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
