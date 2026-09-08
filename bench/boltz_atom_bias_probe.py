"""Same-native-p atom attention bias control; not model parity admission."""

import argparse
import json
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
    paths = [root / "trunk-boundaries/input_embedder.atom_encoder.npz",
             root / "trunk-boundaries/input_embedder.atom_enc_proj_z.npz",
             args.weights.resolve(strict=True), Path(__file__).resolve()]
    if args.ffi_library:
        paths.extend([
            args.ffi_library.resolve(strict=True),
            Path(__file__).with_name("native_cublaslt_ffi.py").resolve(),
            Path(__file__).with_name("native_cublaslt_ffi.cc").resolve(),
        ])
    bindings = {str(p): sha(p) for p in paths}
    with np.load(paths[0], allow_pickle=False) as archive:
        p = archive["2"]
    with np.load(paths[1], allow_pickle=False) as archive:
        expected = archive[""]
    with safe_open(paths[2], framework="numpy") as handle:
        prefix = "d:trunk/d:input_embedder/d:atom_enc_proj_z/d:"
        scale = handle.get_tensor(prefix + "norm/d:scale")
        bias = handle.get_tensor(prefix + "norm/d:bias")
        weight = handle.get_tensor(prefix + "linear/d:kernel")
    if p.shape != (1, 97, 32, 128, 16) or expected.shape != (*p.shape[:-1], 12):
        raise ValueError("requires observed atom bias profile")
    if any(a.dtype != np.float32 or not np.isfinite(a).all()
           for a in (p, expected, scale, bias, weight)):
        raise ValueError("requires finite lossless FP32 capture/weights")

    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives._common import layer_norm, linear
    from foldjax.models.boltz2.models.primitives.native_amp_norm import amp_layer_norm

    if len(jax.devices()) != 1 or jax.default_backend() != "gpu":
        raise RuntimeError("requires one GPU")
    operands = tuple(jnp.asarray(a) for a in (p, scale, bias, weight))
    results = {}
    profiles = [("ordinary", layer_norm, None), ("native_amp", amp_layer_norm, None)]
    if args.ffi_library:
        from bench.native_cublaslt_ffi import linear as ffi_linear
        from bench.native_cublaslt_ffi import register

        profiles.append(("native_amp_ffi", amp_layer_norm, register(args.ffi_library)))
    with jax.default_matmul_precision("highest"):
        for name, norm, target in profiles:
            def call(p, scale, bias, weight):
                normalized = norm(p, scale, bias, 1e-5)
                if target is not None:
                    return ffi_linear(normalized, weight.T, target=target)
                return linear(normalized, weight.astype(jnp.bfloat16))

            fn = jax.jit(call, compiler_options={"xla_allow_excess_precision": False})
            actual = np.asarray(fn(*operands).astype(jnp.float32))
            repeat = np.asarray(fn(*operands).astype(jnp.float32))
            results[name] = {
                "comparison": bitwise_comparison(actual, expected),
                "repeat": bitwise_comparison(actual, repeat),
            }
    if any(sha(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("bound source/inputs changed")
    save_new(args.out, {
        "results": results, "bindings": bindings, "bindings_unchanged": True,
        "scope": "same native atom p; managed weights; not input/conversion proof",
        "not_model_parity_admission": True,
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
    })
    print(json.dumps(results))


if __name__ == "__main__":
    main()
