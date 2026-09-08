"""Compare existing norm implementations on the full captured first LM norm.

Development-only same-operand control. No ESM runtime routing is changed.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from bench.esmfold2_lm_encoder_candidate import (
    compare_boundaries,
    validate_capture,
    validate_first_norm,
)
from bench.esmfold2_tape import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "weights", "source-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    captured = {
        str(p.resolve()): _sha256(p)
        for p in (args.native / "report.json", args.native / "native.npz")
    }
    report, arrays = validate_capture(args.native, args.weights)
    validate_first_norm(report, arrays)

    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.boltz2.models.primitives import native_amp_norm
    from foldjax.models.esmfold2.models import primitives

    source = args.source_root.resolve() / "src"
    for module in (native_amp_norm, primitives):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError("runtime imported from a different source")
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("norm control requires one CUDA GPU")
    paths = [
        args.native / "report.json",
        args.native / "native.npz",
        args.weights / "model.safetensors",
        Path(__file__),
        Path(inspect.getfile(validate_capture)),
        Path(inspect.getfile(_sha256)),
        *sorted((source / "foldjax/models/boltz2/models/primitives").glob("*.py")),
        Path(primitives.__file__),
    ]
    before = {str(p.resolve()): _sha256(p) for p in paths}
    if any(before[path] != digest for path, digest in captured.items()):
        raise RuntimeError("native capture changed while loading")
    prefix = "lm_encoder.blocks.0.tri_mul_out._engine.norm_start"
    with safe_open(args.weights / "model.safetensors", framework="numpy") as handle:
        scale, bias = (
            jnp.asarray(handle.get_tensor(prefix + "." + name))
            for name in ("weight", "bias")
        )
    x = jnp.asarray(arrays["block.input"], jnp.bfloat16).astype(jnp.float32)
    if not np.array_equal(np.asarray(x), arrays["block.input"]):
        raise ValueError("native BF16 input was not stored losslessly")
    target = arrays["first_norm.output"]
    del arrays
    args.output.mkdir(parents=True, exist_ok=False)
    outputs, results = {}, {}
    for name, fn in (
        ("generic", primitives.layer_norm),
        ("existing_cuda_welford", native_amp_norm.amp_layer_norm),
    ):
        compiled = (
            jax.jit(lambda a, s, b: fn(a, s, b, 1e-5)).lower(x, scale, bias).compile()
        )
        value = compiled(x, scale, bias)
        value.block_until_ready()
        repeated = compiled(x, scale, bias)
        repeated.block_until_ready()
        actual = np.asarray(value)
        if not np.isfinite(actual).all():
            raise ValueError("nonfinite norm output")
        outputs[name] = actual
        hlo = args.output / f"{name}.hlo.txt"
        hlo.write_text(compiled.as_text())
        results[name] = {
            "repeat_bitwise": actual.tobytes() == np.asarray(repeated).tobytes()
        }
        for label, dtype in (
            ("raw_fp32", "float32"),
            ("effective_bfloat16", "bfloat16"),
        ):
            left = np.asarray(jnp.asarray(target).astype(dtype).astype(jnp.float32))
            right = np.asarray(value.astype(dtype).astype(jnp.float32))
            results[name][label] = compare_boundaries(
                {"norm": left},
                {"norm": right},
                {"norm": {"original_dtype": dtype}},
                {"norm": {"dtype": dtype}},
            )
        results[name]["hlo_sha256"] = _sha256(hlo)
    output_path = args.output / "outputs.npz"
    np.savez_compressed(output_path, **outputs)
    if before != {str(p.resolve()): _sha256(p) for p in paths}:
        raise RuntimeError("bound artifacts changed during norm control")
    result = {
        "scope": "full first outgoing norm; same-operand implementation control",
        "model_admission": False,
        "runtime_routing_changed": False,
        "bindings": before,
        "native_scope": report["scope"],
        "jax": jax.__version__,
        "results": results,
        "outputs_sha256": _sha256(output_path),
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
