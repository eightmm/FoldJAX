"""Current-policy JAX first LM block on the exact captured native BF16 input."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

from bench.boltz_amp_report import compare_arrays
from bench.esmfold2_lm_shim_probe import boundary_slice
from bench.esmfold2_tape import (
    _npz,
    _save_npz,
    _sha256,
    configure_autotune_control,
    finish_autotune_control,
)


def compare_boundaries(left, right, native_schema, candidate_schema):
    """NPZ storage is FP32; retain the actual producer dtypes separately."""
    return compare_arrays(
        left,
        right,
        {key: {"dtype": native_schema[key]["original_dtype"]} for key in left},
        {key: {"dtype": candidate_schema[key]["dtype"]} for key in right},
    )


def compiler_control(profile):
    if profile == "native-chunks-strict-rounding":
        return {
            **compiler_control("no-triton-native-chunks"),
            "xla_allow_excess_precision": False,
        }
    if profile == "default":
        return {}
    if profile == "no-triton-gemm":
        return {"xla_gpu_enable_triton_gemm": False}
    if profile == "no-triton-no-padding":
        # Installed CUDA pass name; verify actual K437 preservation in HLO.
        return {
            "xla_gpu_enable_triton_gemm": False,
            "xla_disable_hlo_passes": "cublas-pad-for-gemms",
        }
    if profile == "no-triton-no-padding-no-slice-fusion":
        return {
            "xla_gpu_enable_triton_gemm": False,
            "xla_disable_hlo_passes": (
                "cublas-pad-for-gemms,dynamic-slice-fusion-rewriter-v2"
            ),
        }
    if profile == "no-triton-native-chunks":
        return {
            "xla_gpu_enable_triton_gemm": False,
            "xla_disable_hlo_passes": (
                "cublas-pad-for-gemms,dynamic-slice-fusion-rewriter-v2,dot-merger"
            ),
        }
    raise ValueError("unknown block compiler control")


def first_difference_indices(left, right, limit=16):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("effective operand comparison requires identical shape/dtype")
    differing = np.flatnonzero(left.ravel() != right.ravel())
    return {
        "different_entries": int(differing.size),
        "first_indices": [
            list(map(int, np.unravel_index(i, left.shape))) for i in differing[:limit]
        ],
    }


def validate_first_norm(report, arrays):
    key = "first_norm.output"
    schema, value = report["boundaries"][key], arrays[key]
    expected = report["boundaries"]["block.input"]["shape"]
    if (
        not report.get("capture_first_norm")
        or schema["scope"] != "full"
        or schema["original_dtype"] != "torch.float32"
        or schema["shape"] != expected
        or schema["stored_shape"] != expected
        or list(value.shape) != expected
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("native first norm must be full original FP32 output")


def validate_first_projection(report, arrays):
    validate_first_norm(report, arrays)
    if (
        not report.get("capture_first_projection")
        or report.get("instrumentation_output_bytes_equal") is not True
    ):
        raise ValueError("native first projection requires an unchanged captured block")
    shape = report["boundaries"]["block.input"]["shape"]
    for suffix, width, dtype in (
        ("input", shape[-1], "torch.float32"),
        ("output", 4 * shape[-1], "torch.bfloat16"),
    ):
        key = f"first_projection.{suffix}"
        schema, value = report["boundaries"][key], arrays[key]
        expected = [*shape[:-1], width]
        if (
            schema["scope"] != "full"
            or schema["shape"] != expected
            or schema["stored_shape"] != expected
            or schema["original_dtype"] != dtype
            or list(value.shape) != expected
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("invalid full first projection shape/dtype/storage")
        if suffix == "output" and np.any(value.view(np.uint32) & 0xFFFF):
            raise ValueError("first projection must retain lossless BF16 output")
    if not np.array_equal(
        arrays["first_projection.input"].view(np.uint32),
        arrays["first_norm.output"].view(np.uint32),
    ):
        raise ValueError("first projection did not receive the captured norm")


def validate_capture(root, weights):
    report = json.loads((root / "report.json").read_text())
    if (
        report["scope"] != "teacher_forced_first_lm_encoder_block_dropout_disabled"
        or report["entry_cast"] != "float32_to_bfloat16"
        or report["chunk_size"] != 64
        or report["kernel_backend"] is not None
        or report["archive_sha256"] != _sha256(root / "native.npz")
    ):
        raise ValueError("native block capture policy/artifact differs")
    for path, digest in report["bindings"].items():
        if _sha256(Path(path)) != digest:
            raise ValueError("native block binding changed")
    checkpoint_hash = _sha256(weights / "model.safetensors")
    if report["input_shim_policy"]["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("candidate checkpoint differs from native")
    arrays = _npz(root / "native.npz")
    for name in ("block.input", "block.output", "pair_mask"):
        value = arrays[name]
        schema = report["boundaries"][name]
        if (
            schema["scope"] != "full"
            or list(value.shape) != schema["shape"]
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("invalid full block archive schema")
    if (
        report["boundaries"]["block.input"]["original_dtype"] != "torch.bfloat16"
        or arrays["block.input"].shape != arrays["block.output"].shape
        or arrays["pair_mask"].shape != arrays["block.input"].shape[:-1]
    ):
        raise ValueError("native block input dtype/shape differs")
    return report, arrays


def validate_incoming_boundary(report, arrays):
    if (
        not report.get("capture_incoming_boundary")
        or report.get("instrumentation_output_bytes_equal") is not True
    ):
        raise ValueError("incoming capture requires an unchanged full native block")
    expected = report["boundaries"]["block.input"]["shape"]
    for leaf in ("input", "left", "right", "output"):
        key = "incoming." + leaf
        value, schema = arrays[key], report["boundaries"][key]
        dtype = "torch.float32" if leaf in ("left", "right") else "torch.bfloat16"
        if (
            schema["scope"] != "full"
            or schema["shape"] != expected
            or schema["stored_shape"] != expected
            or schema["original_dtype"] != dtype
            or list(value.shape) != expected
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("incoming capture shape/dtype/storage differs")
        if dtype == "torch.bfloat16" and np.any(value.view(np.uint32) & 0xFFFF):
            raise ValueError("incoming BF16 capture is not lossless")


def assemble_incoming_chunks(lefts, outputs, tokens):
    return assemble_triangle_chunks(lefts, outputs, tokens, left_axis=2)


def assemble_triangle_chunks(lefts, outputs, tokens, *, left_axis):
    import jax.numpy as jnp

    if left_axis not in (1, 2):
        raise ValueError("triangle output-i must slice left axis 1 or 2")
    sizes = [min(64, tokens - start) for start in range(0, tokens, 64)]
    if [x.shape[left_axis] for x in lefts] != sizes or [
        x.shape[1] for x in outputs
    ] != sizes:
        raise ValueError("incoming chunks do not retain native output-i order")
    if any(x.dtype != jnp.bfloat16 for x in (*lefts, *outputs)):
        raise ValueError("incoming consumer operands/results must be BF16")
    return jnp.concatenate(lefts, axis=left_axis), jnp.concatenate(outputs, axis=1)


def validate_outgoing_boundary(report, arrays):
    validate_first_projection(report, arrays)
    validate_incoming_boundary(report, arrays)
    if not report.get("capture_outgoing_boundary"):
        raise ValueError("full outgoing boundary was not captured")
    shape = report["boundaries"]["block.input"]["shape"]
    for leaf in ("left", "right", "output", "norm_mix", "proj_emit", "proj_gate"):
        key = "outgoing." + leaf
        value, schema = arrays[key], report["boundaries"][key]
        dtype = (
            "torch.float32"
            if leaf in ("left", "right", "norm_mix")
            else "torch.bfloat16"
        )
        if (
            schema["scope"] != "full"
            or schema["shape"] != shape
            or schema["stored_shape"] != shape
            or schema["original_dtype"] != dtype
            or list(value.shape) != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("outgoing capture shape/dtype/storage differs")
        if dtype == "torch.bfloat16" and np.any(value.view(np.uint32) & 0xFFFF):
            raise ValueError("outgoing BF16 capture is not lossless")


def validate_transition_projection(report, arrays):
    if (
        not report.get("capture_transition_projection")
        or report.get("instrumentation_output_bytes_equal") is not True
    ):
        raise ValueError(
            "transition projection requires an unchanged full native capture"
        )
    batch, rows, columns, channels = report["boundaries"]["block.input"]["shape"]
    for suffix, width in (("input", 4 * channels), ("output", channels)):
        key = "transition_projection." + suffix
        value, schema = arrays[key], report["boundaries"][key]
        shape = [batch, min(rows, 64), columns, width]
        if (
            schema["scope"] != "full"
            or schema["original_dtype"] != "torch.bfloat16"
            or schema["shape"] != shape
            or schema["stored_shape"] != shape
            or list(value.shape) != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or np.any(value.view(np.uint32) & 0xFFFF)
        ):
            raise ValueError("invalid full BF16 transition projection boundary")


def native_output_linear(x, params, prefix, *, preserve_boundary=False):
    import jax
    import jax.numpy as jnp

    if prefix + ".bias" in params:
        raise ValueError("native-output linear control only covers bias-free GEMMs")
    result = jnp.matmul(
        x.astype(jnp.bfloat16),
        params[prefix + ".weight"].astype(jnp.bfloat16).T,
        precision="default",
    )
    return jax.lax.optimization_barrier(result) if preserve_boundary else result


def ffi_output_linear(x, params, prefix, *, target):
    from bench.native_cublaslt_ffi import linear

    if prefix + ".bias" in params:
        raise ValueError(
            "native FFI block control is verified only for bias-free linears"
        )
    return linear(x, params[prefix + ".weight"], target=target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-autocast", action="store_true")
    parser.add_argument("--native-linear-output", action="store_true")
    parser.add_argument("--linear-output-barrier", action="store_true")
    parser.add_argument("--ffi-library", type=Path)
    parser.add_argument("--capture-autotune", action="store_true")
    parser.add_argument("--autotune-load", type=Path)
    parser.add_argument("--capture-first-operands", action="store_true")
    parser.add_argument("--capture-first-norm", action="store_true")
    parser.add_argument("--capture-first-projection", action="store_true")
    parser.add_argument("--capture-incoming-boundary", action="store_true")
    parser.add_argument("--capture-outgoing-boundary", action="store_true")
    parser.add_argument("--capture-transition-projection", action="store_true")
    parser.add_argument("--uncompressed", action="store_true")
    parser.add_argument(
        "--compiler-profile",
        choices=(
            "default",
            "no-triton-gemm",
            "no-triton-no-padding",
            "no-triton-no-padding-no-slice-fusion",
            "no-triton-native-chunks",
            "native-chunks-strict-rounding",
        ),
        default="default",
    )
    for name in ("native", "weights", "candidate-source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.native_linear_output and not args.native_autocast:
        raise ValueError("native linear output requires native autocast")
    if args.linear_output_barrier and not args.native_linear_output:
        raise ValueError("linear output barrier requires native linear output")
    if args.ffi_library and (
        not args.native_autocast or args.native_linear_output or args.autotune_load
    ):
        raise ValueError(
            "FFI needs native autocast without another linear or autotune-load control"
        )
    autotune = configure_autotune_control(
        args.output, enabled=args.capture_autotune, load=args.autotune_load
    )
    compile_options = compiler_control(args.compiler_profile)
    args.capture_incoming_boundary |= args.capture_outgoing_boundary
    args.capture_first_projection |= args.capture_outgoing_boundary
    args.capture_first_norm |= (
        args.capture_first_projection
        or args.capture_incoming_boundary
        or args.capture_transition_projection
    )
    if (
        args.capture_first_operands or args.capture_first_norm
    ) and not args.native_autocast:
        raise ValueError("full effective operand capture requires native-autocast")
    args.output.mkdir(parents=True, exist_ok=False)
    native_manifest_hash = _sha256(args.native / "report.json")
    report, native = validate_capture(args.native, args.weights)
    if args.capture_first_norm:
        validate_first_norm(report, native)
    if args.capture_first_projection:
        validate_first_projection(report, native)
    if args.capture_incoming_boundary:
        validate_incoming_boundary(report, native)
    if args.capture_outgoing_boundary:
        validate_outgoing_boundary(report, native)
    if args.capture_transition_projection:
        validate_transition_projection(report, native)
    if args.capture_first_operands:
        from bench.esmfold2_contraction_probe import validate_operands

        validate_operands(report, native)
    source = args.candidate_source / "src/foldjax/models/esmfold2/models"
    paths = [
        *sorted(source.rglob("*.py")),
        args.candidate_source
        / "src/foldjax/models/boltz2/models/primitives/native_amp_norm.py",
        args.weights / "model.safetensors",
        Path(__file__),
        Path(inspect.getfile(compare_arrays)),
        Path(inspect.getfile(boundary_slice)),
        Path(inspect.getfile(_npz)),
    ]
    if args.ffi_library:
        from bench import native_cublaslt_ffi

        paths.extend(
            [
                args.ffi_library,
                Path(native_cublaslt_ffi.__file__),
                Path(__file__).with_name("native_cublaslt_ffi.cc"),
            ]
        )
    before = {str(path.resolve()): _sha256(path) for path in paths}
    if args.capture_first_operands:
        dependency = Path(inspect.getfile(validate_operands))
        paths.append(dependency)
        before[str(dependency.resolve())] = _sha256(dependency)
    sys.path.insert(0, str(args.candidate_source / "src"))
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.esmfold2.models import model, trunk

    for module in (model, trunk):
        if Path(module.__file__).resolve().parent != source.resolve():
            raise ValueError("candidate imported from a different source")
    if args.native_linear_output:
        # Process-local control; retain/probe wrappers below observe this same
        # helper in both baseline and instrumented executables.
        trunk._autocast_linear = partial(
            native_output_linear, preserve_boundary=args.linear_output_barrier
        )
    if args.ffi_library:
        target = native_cublaslt_ffi.register(args.ffi_library)
        trunk._autocast_linear = partial(ffi_output_linear, target=target)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("candidate block probe requires exactly one GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    prefix = "lm_encoder.blocks.0"
    with safe_open(args.weights / "model.safetensors", framework="numpy") as handle:
        original = {
            key: jnp.asarray(handle.get_tensor(key))
            for key in handle.keys()
            if key.startswith(prefix + ".")
        }
    params = (
        original
        if args.native_autocast
        else model._cast(original, model.TRUNK_PREFIXES, jnp.bfloat16)
    )
    value = jnp.asarray(native["block.input"], jnp.bfloat16)
    if not np.array_equal(np.asarray(value.astype(jnp.float32)), native["block.input"]):
        raise ValueError("native BF16 archive storage is not lossless")
    mask = jnp.asarray(native["pair_mask"])

    def run(x, p, m):
        return trunk.pair_update_block(
            x, p, prefix, mask=m, native_autocast=args.native_autocast
        )

    baseline_lowered = jax.jit(run, compiler_options=compile_options).lower(
        value, params, mask
    )
    baseline_executable = baseline_lowered.compile()
    baseline = baseline_executable(value, params, mask)
    baseline.block_until_ready()
    # Return selected actual helper outputs from the traced graph. Retaining
    # intermediates may change compiler fusion; compare its final output too.
    normalizer, linear = trunk.layer_norm, trunk.linear
    autocast_norm, autocast_linear = trunk._autocast_norm, trunk._autocast_linear
    original_einsum = jnp.einsum
    records = {}

    def instrumented(x, p, m):
        captured = {}
        counts = {}
        incoming_lefts, incoming_outputs, incoming_rhs_checks = [], [], []
        incoming_rhs = None
        outgoing_lefts, outgoing_outputs, outgoing_rhs_checks = [], [], []
        outgoing_rhs = None
        norm_names = {id(v): k[:-7] for k, v in p.items() if k.endswith(".weight")}

        def retain(name, inp, out):
            count = counts.get(name, 0)
            counts[name] = count + 1
            name = name.removeprefix(prefix + ".") + f".{count}"
            if (
                args.capture_transition_projection
                and name == "pair_transition.ffn.w3.0"
            ):
                for suffix, array in (("input", inp), ("output", out)):
                    key = "transition_projection." + suffix
                    captured[key] = array.astype(jnp.float32)
                    records[key] = {
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "scope": "full",
                    }
            if args.capture_outgoing_boundary:
                for leaf in ("norm_mix", "proj_emit", "proj_gate"):
                    if name == f"tri_mul_out._engine.{leaf}.0":
                        key = "outgoing." + leaf
                        captured[key] = out.astype(jnp.float32)
                        records[key] = {
                            "dtype": str(out.dtype),
                            "shape": list(out.shape),
                            "scope": "full",
                        }
            if (
                args.capture_incoming_boundary
                and name == "tri_mul_in._engine.norm_start.0"
            ):
                captured["incoming.input"] = inp.astype(jnp.float32)
                records["incoming.input"] = {
                    "dtype": str(inp.dtype),
                    "shape": list(inp.shape),
                    "scope": "full",
                }
            if args.capture_first_norm and name == "tri_mul_out._engine.norm_start.0":
                captured["first_norm.output"] = out.astype(jnp.float32)
                records["first_norm.output"] = {
                    "dtype": str(out.dtype),
                    "shape": list(out.shape),
                    "scope": "full",
                }
            for suffix, array in (("input", inp), ("output", out)):
                if (
                    args.capture_first_projection
                    and name == "tri_mul_out._engine.proj_bundle.0"
                ):
                    key = f"first_projection.{suffix}"
                    captured[key] = array.astype(jnp.float32)
                    records[key] = {
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "scope": "full",
                    }
                key = name + "." + suffix
                captured[key] = boundary_slice(array).astype(jnp.float32)
                records[key] = {"dtype": str(array.dtype), "shape": list(array.shape)}

        def norm(x, weight, bias=None, **kwargs):
            out = normalizer(x, weight, bias, **kwargs)
            retain(norm_names[id(weight)], x, out)
            return out

        def project(x, params, name):
            out = linear(x, params, name)
            retain(name, x, out)
            return out

        def native_norm(x, params, name, eps=1e-5):
            out = autocast_norm(x, params, name, eps)
            retain(name, x, out)
            return out

        def native_project(x, params, name):
            out = autocast_linear(x, params, name)
            retain(name, x, out)
            return out

        def observe_einsum(equation, lhs, rhs, **kwargs):
            nonlocal incoming_rhs, outgoing_rhs
            if args.capture_first_operands and "effective_first.lhs" not in captured:
                if equation != "bikd,bjkd->bijd":
                    raise ValueError("first contraction must be outgoing")
                for name, operand in (("lhs", lhs), ("rhs", rhs)):
                    key = "effective_first." + name
                    if operand.dtype != jnp.bfloat16:
                        raise ValueError("candidate contraction operand is not BF16")
                    captured[key] = operand.astype(jnp.float32)
                    records[key] = {
                        "dtype": str(operand.dtype),
                        "shape": list(operand.shape),
                        "scope": "full",
                    }
            result = original_einsum(equation, lhs, rhs, **kwargs)
            if args.capture_incoming_boundary and equation == "bkid,bkjd->bijd":
                if incoming_rhs is None:
                    incoming_rhs = rhs
                incoming_rhs_checks.append(
                    jnp.all(
                        jax.lax.bitcast_convert_type(rhs, jnp.uint16)
                        == jax.lax.bitcast_convert_type(incoming_rhs, jnp.uint16)
                    )
                )
                incoming_lefts.append(lhs)
                incoming_outputs.append(result.astype(jnp.bfloat16))
            if args.capture_outgoing_boundary and equation == "bikd,bjkd->bijd":
                if outgoing_rhs is None:
                    outgoing_rhs = rhs
                outgoing_rhs_checks.append(
                    jnp.all(
                        jax.lax.bitcast_convert_type(rhs, jnp.uint16)
                        == jax.lax.bitcast_convert_type(outgoing_rhs, jnp.uint16)
                    )
                )
                outgoing_lefts.append(lhs)
                outgoing_outputs.append(result.astype(jnp.bfloat16))
            return result

        if args.native_autocast:
            trunk._autocast_norm, trunk._autocast_linear = native_norm, native_project
        else:
            trunk.layer_norm, trunk.linear = norm, project
        if args.capture_first_operands or args.capture_incoming_boundary:
            jnp.einsum = observe_einsum
        try:
            output = run(x, p, m)
        finally:
            jnp.einsum = original_einsum
            trunk.layer_norm, trunk.linear = normalizer, linear
            trunk._autocast_norm, trunk._autocast_linear = (
                autocast_norm,
                autocast_linear,
            )
        if args.capture_incoming_boundary:
            left, incoming_output = assemble_incoming_chunks(
                incoming_lefts, incoming_outputs, x.shape[1]
            )
            for key, array in (
                ("effective_incoming.left", left),
                ("effective_incoming.right", incoming_rhs),
                ("incoming.output", incoming_output),
            ):
                captured[key] = array.astype(jnp.float32)
                records[key] = {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "scope": "full",
                }
            captured["incoming.rhs_consistent"] = jnp.stack(incoming_rhs_checks)
        if args.capture_outgoing_boundary:
            left, outgoing_output = assemble_triangle_chunks(
                outgoing_lefts, outgoing_outputs, x.shape[1], left_axis=1
            )
            for key, array in (
                ("effective_outgoing.left", left),
                ("effective_outgoing.right", outgoing_rhs),
                ("outgoing.output", outgoing_output),
            ):
                captured[key] = array.astype(jnp.float32)
                records[key] = {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "scope": "full",
                }
            captured["outgoing.rhs_consistent"] = jnp.stack(outgoing_rhs_checks)
        return output, captured

    observed, slices = jax.jit(instrumented, compiler_options=compile_options)(
        value, params, mask
    )
    observed.block_until_ready()
    arrays = {key: np.asarray(array) for key, array in slices.items()}
    incoming_comparison = None
    outgoing_comparison = None
    if args.capture_outgoing_boundary:
        from bench.boltz_relpos_probe import bf16_round

        if not arrays.pop("outgoing.rhs_consistent").all():
            raise ValueError("outgoing RHS changed between native-sized chunks")
        mapped = {
            "outgoing." + leaf: native["outgoing." + leaf]
            for leaf in ("output", "norm_mix", "proj_emit", "proj_gate")
        }
        mapped.update(
            {
                "effective_outgoing." + leaf: bf16_round(native["outgoing." + leaf])
                for leaf in ("left", "right")
            }
        )
        outgoing_comparison = compare_boundaries(
            mapped,
            {key: arrays[key] for key in mapped},
            {
                key: {
                    "original_dtype": "float32"
                    if key == "outgoing.norm_mix"
                    else "bfloat16"
                }
                for key in mapped
            },
            records,
        )
    if args.capture_incoming_boundary:
        from bench.boltz_relpos_probe import bf16_round

        if not arrays.pop("incoming.rhs_consistent").all():
            raise ValueError("incoming RHS changed between native-sized chunks")
        mapped = {key: native[key] for key in ("incoming.input", "incoming.output")}
        mapped.update(
            {
                "effective_incoming." + leaf: bf16_round(native["incoming." + leaf])
                for leaf in ("left", "right")
            }
        )
        incoming_comparison = compare_boundaries(
            mapped,
            {key: arrays[key] for key in mapped},
            {key: {"original_dtype": "bfloat16"} for key in mapped},
            records,
        )
    norm_comparison = None
    if args.capture_first_norm:
        left, right = native["first_norm.output"], arrays["first_norm.output"]
        effective_left, effective_right = (
            np.asarray(jnp.asarray(v, jnp.bfloat16).astype(jnp.float32))
            for v in (left, right)
        )
        norm_comparison = {
            "scope": "full_first_outgoing_norm_before_projection",
            "raw": compare_boundaries(
                {"norm": left},
                {"norm": right},
                {"norm": report["boundaries"]["first_norm.output"]},
                {"norm": records["first_norm.output"]},
            ),
            "effective_bfloat16": compare_boundaries(
                {"norm": effective_left},
                {"norm": effective_right},
                {"norm": {"original_dtype": "bfloat16"}},
                {"norm": {"dtype": "bfloat16"}},
            ),
            "effective_first_differences": first_difference_indices(
                effective_left, effective_right
            ),
            "projection_capture": (
                "full input/output captured"
                if args.capture_first_projection
                else "not captured; check effective norm equality first"
            ),
        }
    projection_comparison = None
    if args.capture_first_projection:
        keys = ("first_projection.input", "first_projection.output")
        projection_comparison = compare_boundaries(
            {key: native[key] for key in keys},
            {key: arrays[key] for key in keys},
            report["boundaries"],
            records,
        )
    effective_comparison = None
    if args.capture_first_operands:
        mapped = {
            "effective_first." + name: np.asarray(
                jnp.asarray(native["first_contraction." + name], jnp.bfloat16).astype(
                    jnp.float32
                )
            )
            for name in ("lhs", "rhs")
        }
        effective_comparison = {
            "scope": "complete first64chunk effective BF16 operands",
            "native_producer_dtype": "float32",
            "native_effective_dtype": "bfloat16",
            "mapping": (
                "native captured FP32 operand explicitly cast BF16 "
                "then lossless FP32 storage"
            ),
            "comparison": compare_boundaries(
                mapped,
                {key: arrays[key] for key in mapped},
                {key: {"original_dtype": "bfloat16"} for key in mapped},
                records,
            ),
            "first_differences": {
                key: first_difference_indices(mapped[key], arrays[key])
                for key in mapped
            },
        }
    shared = sorted(set(arrays) & set(native))
    comparisons = compare_boundaries(
        {key: native[key] for key in shared},
        {key: arrays[key] for key in shared},
        report["boundaries"],
        records,
    )
    arrays["block.output"] = np.asarray(baseline.astype(jnp.float32))
    arrays["block.instrumented_output"] = np.asarray(observed.astype(jnp.float32))
    if args.uncompressed:
        with (args.output / "jax.npz").open("xb") as stream:
            np.savez(stream, **arrays)
    else:
        _save_npz(args.output / "jax.npz", arrays)
    (args.output / "baseline.hlo.txt").write_text(baseline_lowered.as_text())
    (args.output / "baseline.optimized.hlo.txt").write_text(
        baseline_executable.as_text()
    )
    validate_capture(args.native, args.weights)
    if native_manifest_hash != _sha256(args.native / "report.json") or before != {
        str(path.resolve()): _sha256(path) for path in paths
    }:
        raise ValueError("bound candidate/native sources changed")
    result = {
        "scope": "same_native_block_input_candidate_operator_policy",
        "native_autocast": args.native_autocast,
        "model_admission": None,
        "bindings": before,
        "native_manifest_sha256": native_manifest_hash,
        "native_precision_control": report["precision_control"],
        "parameter_policy": (
            "original_parameters_explicit_native_autocast"
            if args.native_autocast
            else "model._cast(original, TRUNK_PREFIXES, bfloat16)"
        ),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "matmul": "highest",
        "compiler_profile": args.compiler_profile,
        "compiler_options": compile_options,
        "native_linear_output": args.native_linear_output,
        "linear_output_barrier": args.linear_output_barrier,
        "ffi_native_dispatch": bool(args.ffi_library),
        "autotune_control": finish_autotune_control(autotune),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "compiler_control_is_production_change": False,
        "boundaries": records,
        "full_block": compare_boundaries(
            {"output": native["block.output"]},
            {"output": arrays["block.output"]},
            {"output": report["boundaries"]["block.output"]},
            {"output": {"dtype": str(baseline.dtype)}},
        ),
        "instrumentation_effect": compare_boundaries(
            {"output": arrays["block.output"]},
            {"output": arrays["block.instrumented_output"]},
            {"output": {"original_dtype": str(baseline.dtype)}},
            {"output": {"dtype": str(observed.dtype)}},
        ),
        "matched_boundary_slices": comparisons,
        "effective_first_operands": effective_comparison,
        "full_first_norm": norm_comparison,
        "full_first_projection": projection_comparison,
        "full_transition_projection": compare_boundaries(
            {
                key: native[key]
                for key in (
                    "transition_projection.input",
                    "transition_projection.output",
                )
            },
            {
                key: arrays[key]
                for key in (
                    "transition_projection.input",
                    "transition_projection.output",
                )
            },
            report["boundaries"],
            records,
        )
        if args.capture_transition_projection
        else None,
        "full_incoming_boundary": incoming_comparison,
        "full_outgoing_boundary": outgoing_comparison,
        "archive_compression": "stored" if args.uncompressed else "deflate",
        "uncaptured_native_boundaries": sorted(set(native) - set(slices)),
        "archive_sha256": _sha256(args.output / "jax.npz"),
        "hlo_sha256": _sha256(args.output / "baseline.hlo.txt"),
        "optimized_hlo_sha256": _sha256(args.output / "baseline.optimized.hlo.txt"),
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
