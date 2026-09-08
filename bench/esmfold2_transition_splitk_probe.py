"""Explicit split-K hypotheses for the measured three-part native launch."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from bench.boltz_amp_report import compare_arrays
from bench.esmfold2_lm_encoder_candidate import (
    compare_boundaries,
    validate_capture,
    validate_transition_projection,
)
from bench.esmfold2_tape import (
    _save_npz,
    _sha256,
    configure_autotune_control,
    finish_autotune_control,
)


def split_k_ranges(k, tile, partitions):
    if any(type(v) is not int or v <= 0 for v in (k, tile, partitions)):
        raise ValueError("positive integer K, tile and partition count required")
    span = ((k + tile - 1) // tile + partitions - 1) // partitions * tile
    return [(start, min(start + span, k)) for start in range(0, k, span)]


def serial_splitk(x, weight, *, partitions=3, alignment=32):
    import jax.numpy as jnp

    carry = jnp.zeros((*x.shape[:-1], weight.shape[0]), jnp.float32)
    for start, stop in split_k_ranges(x.shape[-1], alignment, partitions):
        part = jnp.matmul(
            x[..., start:stop].astype(jnp.bfloat16),
            weight[:, start:stop].astype(jnp.bfloat16).T,
            preferred_element_type=jnp.float32,
        )
        carry = (part + carry).astype(jnp.bfloat16).astype(jnp.float32)
    return carry.astype(jnp.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserve-rounding", action="store_true")
    parser.add_argument("--partition-alignment", type=int, choices=(8, 32), default=32)
    parser.add_argument("--native-output", action="store_true")
    parser.add_argument("--ffi-library", type=Path)
    parser.add_argument("--first-heuristic-only", action="store_true")
    parser.add_argument("--autotune-level", type=int, choices=(0, 4))
    parser.add_argument("--capture-autotune", action="store_true")
    parser.add_argument("--autotune-load", type=Path)
    parser.add_argument(
        "--operand-precision", choices=("highest", "default"), default="highest"
    )
    for name in ("native", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.ffi_library and (
        not args.native_output
        or args.preserve_rounding
        or args.first_heuristic_only
        or args.autotune_level is not None
        or args.capture_autotune
        or args.autotune_load
    ):
        raise ValueError(
            "FFI requires native-output without compiler/autotune controls"
        )
    autotune = configure_autotune_control(
        args.output, enabled=args.capture_autotune, load=args.autotune_load
    )
    report, arrays = validate_capture(args.native, args.weights)
    validate_transition_projection(report, arrays)
    if report["precision_control"]:
        raise ValueError("reference must retain native default")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        args.native / "report.json",
        args.native / "native.npz",
        args.weights / "model.safetensors",
        Path(__file__),
        Path(inspect.getfile(validate_transition_projection)),
        Path(inspect.getfile(compare_arrays)),
        Path(inspect.getfile(_sha256)),
    ]
    before = {str(p.resolve()): _sha256(p) for p in paths}
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("one CUDA GPU required")
    jax.config.update("jax_default_matmul_precision", "highest")
    prefix = "lm_encoder.blocks.0.pair_transition.ffn.w3"
    with safe_open(args.weights / "model.safetensors", framework="numpy") as handle:
        if prefix + ".bias" in handle.keys():
            raise ValueError("measured native w3 control is bias-free")
        weight = jnp.asarray(handle.get_tensor(prefix + ".weight"))
    value = jnp.asarray(arrays["transition_projection.input"], jnp.bfloat16)
    if value.shape != (1, 64, 437, 1024) or weight.shape != (256, 1024):
        raise ValueError("unmeasured w3 shape for split-K=3 control")
    expected = arrays["transition_projection.output"]
    del arrays
    options = {
        "xla_gpu_enable_triton_gemm": False,
        "xla_disable_hlo_passes": (
            "cublas-pad-for-gemms,dynamic-slice-fusion-rewriter-v2,dot-merger"
        ),
    }
    if args.preserve_rounding:
        options["xla_allow_excess_precision"] = False
    if args.first_heuristic_only:
        options["xla_gpu_autotune_max_solutions"] = 1
    if args.autotune_level is not None:
        options["xla_gpu_autotune_level"] = args.autotune_level
    if args.ffi_library:
        from bench import native_cublaslt_ffi

        options = {}
        extra = [
            args.ffi_library,
            Path(native_cublaslt_ffi.__file__),
            Path(__file__).with_name("native_cublaslt_ffi.cc"),
        ]
        paths.extend(extra)
        before.update({str(p.resolve()): _sha256(p) for p in extra})
        target = native_cublaslt_ffi.register(args.ffi_library)

    def run(x, w):
        if args.ffi_library:
            return native_cublaslt_ffi.linear(x, w, target=target)
        if args.native_output:
            return jnp.matmul(
                x.astype(jnp.bfloat16),
                w.astype(jnp.bfloat16).T,
                precision=args.operand_precision,
            )
        return serial_splitk(x, w, alignment=args.partition_alignment)

    compiled = (
        jax.jit(
            run,
            compiler_options=options,
        )
        .lower(value, weight)
        .compile()
    )
    outputs = {}
    for repeat in range(2):
        result = compiled(value, weight)
        result.block_until_ready()
        outputs[str(repeat)] = np.asarray(result.astype(jnp.float32))
    (args.output / "compiled.hlo.txt").write_text(compiled.as_text())
    comparison = compare_boundaries(
        {key: expected for key in outputs},
        outputs,
        {key: {"original_dtype": "torch.bfloat16"} for key in outputs},
        {key: {"dtype": "bfloat16"} for key in outputs},
    )
    if before != {str(p.resolve()): _sha256(p) for p in paths}:
        raise ValueError("reference/checkpoint/runner changed during probe")
    _save_npz(args.output / "outputs.npz", outputs)
    result = {
        "scope": "same_complete_first_w3_native_output_counterfactual"
        if args.native_output
        else "same_complete_first_w3_serial_splitk_counterfactual",
        "model_admission": None,
        "bindings": before,
        "ranges": [(0, value.shape[-1])]
        if args.native_output
        else split_k_ranges(value.shape[-1], args.partition_alignment, 3),
        "partition_alignment": None if args.native_output else args.partition_alignment,
        "native_output": args.native_output,
        "ffi_native_dispatch": bool(args.ffi_library),
        "ffi_compute_policy": "BF16 inputs/output; FP32 compute; first heuristic"
        if args.ffi_library
        else None,
        "operand_precision": args.operand_precision
        if args.native_output
        else "highest",
        "compiler_options": options,
        "autotune_control": finish_autotune_control(autotune),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "comparison": comparison,
        "repeat_bytes_equal": outputs["0"].tobytes() == outputs["1"].tobytes(),
        "hlo_sha256": _sha256(args.output / "compiled.hlo.txt"),
        "outputs_sha256": _sha256(args.output / "outputs.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
