"""Same-operand MSA embedding contraction diagnostic, not model admission."""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison
from bench.esmfold2_lm_encoder_candidate import compiler_control
from bench.esmfold2_tape import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opm-wout", action="store_true")
    for name in ("reference", "weights", "ffi-library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.reference / "report.json").read_text())
    if report.get("engine") != "native":
        raise ValueError("native capture required")
    bindings = dict(report["bindings"])
    if bindings.get(str(args.weights.resolve())) != _sha256(args.weights):
        raise ValueError("checkpoint binding differs")
    bindings[str((args.reference / "prefix.npz").resolve())] = report["archive_sha256"]
    bindings[str((args.reference / "report.json").resolve())] = _sha256(
        args.reference / "report.json"
    )
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("native capture binding changed")
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from bench import native_cublaslt_ffi as ffi

    if jax.default_backend() != "gpu":
        raise ValueError("CUDA replay required")
    for path in (
        Path(__file__),
        Path(__file__).with_name("esmfold2_lm_encoder_candidate.py"),
        Path(__file__).with_name("boltz_pwa_averaging_probe.py"),
        Path(ffi.__file__),
        Path(ffi.__file__).with_suffix(".cc"),
        args.ffi_library,
    ):
        bindings[str(path.resolve())] = _sha256(path)
    key = "blocks.0.outer_product_mean.Wout" if args.opm_wout else "embed"
    prefix = "msa_encoder." + key
    with np.load(args.reference / "prefix.npz") as archive:
        x = archive[key + ".input"]
        expected = archive[key + ".output"]
    with safe_open(args.weights, framework="numpy") as handle:
        if not args.opm_wout and prefix + ".bias" in handle.keys():
            raise ValueError("bias-free embedding required")
        weight = handle.get_tensor(prefix + ".weight")
        bias = handle.get_tensor(prefix + ".bias") if args.opm_wout else None
    if x.shape[-1] != weight.shape[-1] or expected.shape != (
        *x.shape[:-1],
        weight.shape[0],
    ):
        raise ValueError("embedding shapes differ")
    if not all(np.isfinite(v).all() for v in (x, weight, expected)):
        raise ValueError("nonfinite operands")
    target = ffi.register(args.ffi_library)
    jax.config.update("jax_default_matmul_precision", "highest")
    operands = tuple(jnp.asarray(v).astype(jnp.bfloat16) for v in (x, weight))
    arms = {
        "bf16_destination": lambda a, w: a @ w.T,
        "fp32_accumulate": lambda a, w: jnp.matmul(
            a, w.T, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16),
        "native_ffi": lambda a, w: ffi.linear(a, w, target=target),
    }
    if args.opm_wout:
        from foldjax.models.esmfold2.models import trunk

        bindings[str(Path(trunk.__file__).resolve())] = _sha256(Path(trunk.__file__))
        if bias.shape != (weight.shape[0],) or not np.isfinite(bias).all():
            raise ValueError("invalid OPM bias")
        operands += (jnp.asarray(bias).astype(jnp.bfloat16),)
        target32 = ffi.register(args.ffi_library, fp32=True)
        arms = {
            "runtime": lambda a, w, b: trunk._autocast_linear(
                a, {prefix + ".weight": w, prefix + ".bias": b}, prefix
            ),
            "bf16_before_bias": lambda a, w, b: (
                jax.lax.optimization_barrier(a @ w.T) + b
            ),
            "fp32_ffi_then_bias": lambda a, w, b: (
                ffi.linear(
                    a.astype(jnp.float32),
                    w.astype(jnp.float32),
                    target=target32,
                    fp32=True,
                )
                + b.astype(jnp.float32)
            ).astype(jnp.bfloat16),
        }
    results = {}
    for name, fn in arms.items():
        run = jax.jit(
            fn, compiler_options=compiler_control("native-chunks-strict-rounding")
        )
        output = np.asarray(run(*operands).astype(jnp.float32))
        repeat = np.asarray(run(*operands).astype(jnp.float32))
        results[name] = {
            "native": bitwise_comparison(output, expected),
            "repeat": bitwise_comparison(output, repeat),
        }
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("binding changed during replay")
    save_new(
        args.output,
        {
            "operator": key,
            "arms": results,
            "bindings": bindings,
            "full_model_admission": False,
        },
    )


if __name__ == "__main__":
    main()
