"""Same-operand FP32 single prenorm replay; not full-model parity admission."""

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
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    verify_capture(root, foldjax=False)
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = sorted(root.glob(
        "trunk-boundaries/cycle-*/pairformer_module.layers.0.pre_norm_s.npz"
    ))
    if len(paths) != 2:
        raise ValueError("requires first and last recycle captures")
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives import _common

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("requires one GPU")
    bindings = {str(p): sha(p) for p in (
        *paths, args.weights.resolve(strict=True), Path(__file__).resolve(),
        Path(_common.__file__).resolve(),
    )}
    prefix = "d:trunk/d:pairformer_module/d:layers/i:0/d:pre_norm_s"
    with safe_open(args.weights, framework="numpy") as handle:
        scale = handle.get_tensor(prefix + "/d:scale")
        bias = handle.get_tensor(prefix + "/d:bias")
    fn = jax.jit(_common.layer_norm,
                 compiler_options={"xla_allow_excess_precision": False})
    results = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            x, expected = archive["input"], archive["output"]
        if x.shape != expected.shape or x.shape[-1] != 384:
            raise ValueError("requires matching width384 operands")
        if any(a.dtype != np.float32 or not np.isfinite(a).all()
               for a in (x, expected, scale, bias)):
            raise ValueError("requires finite FP32 operands")
        operands = tuple(jnp.asarray(a) for a in (x, scale, bias))
        actual = np.asarray(fn(*operands, 1e-5))
        repeat = np.asarray(fn(*operands, 1e-5))
        results[path.parent.name] = {
            "comparison": bitwise_comparison(actual, expected),
            "repeat": bitwise_comparison(actual, repeat),
        }
    if any(sha(Path(p)) != value for p, value in bindings.items()):
        raise ValueError("bound inputs changed")
    save_new(args.out, {"results": results, "bindings": bindings,
                        "not_model_parity_admission": True})


if __name__ == "__main__":
    main()
