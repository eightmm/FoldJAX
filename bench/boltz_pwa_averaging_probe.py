"""Same-operand full-row PWA contraction, not full-PWA/model admission.

Replay one captured head's averaging before testing JAX einsum versus the
native wide-BMM layout. Only the selected NPZ members are materialized.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_probe import source_hashes, verify_bound_file
from bench.boltz_relpos_probe import bf16_round, comparison, torch_policy


def validate_operands(weights, values, expected):
    if any(x.dtype != np.float32 for x in (weights, values, expected)):
        raise ValueError("captures must use lossless FP32 storage")
    if weights.ndim != 4 or weights.shape[:2] != (1, 1):
        raise ValueError("one head and singleton batch are required")
    n = weights.shape[-1]
    if (
        weights.shape[-2] != n
        or values.ndim != 4
        or values.shape[0] != 1
        or values.shape[2] != n
        or min(values.shape) < 1
        or expected.shape != values.shape
    ):
        raise ValueError("PWA operand shapes do not match")
    if not all(np.isfinite(x).all() for x in (weights, values, expected)):
        raise ValueError("nonfinite PWA operand")
    if not np.array_equal(bf16_round(values), values) or not np.array_equal(
        bf16_round(expected), expected
    ):
        raise ValueError("native values and average must be BF16-representable")


def load_reference(root, msa_root, head):
    if not 0 <= head < 8:
        raise ValueError("released PWA has eight heads")
    report = json.loads((root / "report.json").read_text())
    msa = json.loads((msa_root / "report.json").read_text())
    if (
        report.get("arm") != "native"
        or report.get("passed") is not True
        or report.get("native_decomposition", {}).get("values_equal") is not True
        or report.get("row_slice_vs_full", {}).get("values_equal") is not True
        or msa.get("arm") != "native"
        or msa.get("passed") is not True
    ):
        raise ValueError("full native PWA/MSA reproduction must pass first")
    verify_bound_file(msa_root / "report.json", report["native_msa_report_sha256"])
    if report["native_source"] != msa["native_source"]:
        raise ValueError("PWA/MSA source identities differ")
    if report["torch"] != msa["runtime"]["torch"]:
        raise ValueError("PWA/MSA Torch versions differ")
    tree = msa_root / "layers/00/input_m.tree.json"
    verify_bound_file(tree, msa["stages"]["layers/00/input_m"]["tree_sha256"])
    metadata = json.loads(tree.read_text())[""]
    shape = metadata.get("shape", [])
    if (
        len(shape) != 4
        or shape != [1, report["rows"], shape[2], 64]
        or metadata["native_dtype"] != "torch.bfloat16"
        or metadata["storage_dtype"] != "float32"
    ):
        raise ValueError("native PWA must retain the complete released MSA shape")
    archive_path = root / "stages.npz"
    verify_bound_file(archive_path, report["artifacts"]["stages.npz"])
    names = [f"head{head}/{name}" for name in ("weights", "v", "averaged")]
    with np.load(archive_path, allow_pickle=False) as archive:
        if not set(names).issubset(archive.files):
            raise ValueError("selected native head is missing")
        weights, values, expected = (archive[name] for name in names)
    validate_operands(weights, values, expected)
    if values.shape != (1, report["rows"], metadata["shape"][2], 32):
        raise ValueError("selected head does not match the full native MSA")
    if values.shape[2] <= 384:
        raise ValueError("probe requires native head-wise PWA above 384 tokens")
    return report, msa, weights, values, expected


def einsum_average(weights, values):
    """Values enter in the captured [batch, MSA, token, channel] layout."""
    import jax.numpy as jnp

    return jnp.einsum("bhij,bhsjd->bhsid", weights, values[:, None])[:, 0]


def wide_average(weights, values):
    import jax.numpy as jnp

    batch, rows, n, channels = values.shape
    rhs = values.transpose(0, 2, 1, 3).reshape(batch, n, rows * channels)
    out = jnp.matmul(weights[:, 0], rhs)
    return out.reshape(batch, n, rows, channels).transpose(0, 2, 1, 3)


def fp32_accumulator_average(weights, values):
    import jax.numpy as jnp

    result = jnp.einsum(
        "bhij,bhsjd->bhsid",
        weights,
        values[:, None],
        preferred_element_type=jnp.float32,
    )
    return result.astype(values.dtype)[:, 0]


def jax_profiles(accumulation_controls=False):
    profiles = [("einsum", einsum_average, {}), ("native_layout_bmm", wide_average, {})]
    if accumulation_controls:
        no_triton = {"xla_gpu_enable_triton_gemm": False}
        profiles.extend(
            [
                ("einsum_fp32_accumulator", fp32_accumulator_average, {}),
                ("einsum_no_triton", einsum_average, no_triton),
                (
                    "einsum_no_triton_fp32_accumulator",
                    fp32_accumulator_average,
                    no_triton,
                ),
            ]
        )
    return profiles


def native_profiles(reduction, reduction_controls=False):
    profiles = [("einsum", reduction), ("native_layout_bmm", reduction)]
    if reduction_controls:
        profiles.append(("einsum_reduction_disabled", False))
    return profiles


def lowering_evidence(hlo):
    """Report emitted instructions, not the backend requested by a flag."""
    return {
        # Generic Triton conversion/transpose fusions can follow a cuBLAS GEMM.
        "contains_triton_gemm": bool(
            re.search(
                r'(?:\bkind\s*=|"kind"\s*:)\s*'
                r'"__triton_(?:gemm|nested_gemm_fusion)"',
                hlo,
            )
        ),
        "custom_call_targets": sorted(
            set(re.findall(r'custom_call_target="([^"]+)"', hlo))
        ),
        "dot_instructions": [
            line.strip() for line in hlo.splitlines() if " dot(" in line
        ],
    }


def save_array(path, **values):
    with path.open("xb") as stream:
        np.savez(stream, **values)
    return sha(path)


def loaded_cublas_libraries(maps_path=Path("/proc/self/maps")):
    """Bind mapped library files; return private execution metadata only."""
    libraries = {}
    for line in maps_path.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not re.match(
            r"libcublas(?:Lt)?\.so(?:\.|$)", Path(fields[5]).name
        ):
            continue
        path = Path(fields[5])
        if not path.is_absolute() or path.name.endswith(" (deleted)"):
            raise ValueError("mapped cuBLAS file identity is unavailable")
        if str(path) in libraries:
            continue
        before = path.stat()
        device = f"{os.major(before.st_dev):x}:{os.minor(before.st_dev):x}"
        mapped_device = ":".join(f"{int(x, 16):x}" for x in fields[3].split(":"))
        if before.st_ino != int(fields[4]) or device != mapped_device:
            raise ValueError("mapped cuBLAS file was replaced")
        digest = sha(path)
        after = path.stat()
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in identity):
            raise ValueError("cuBLAS file changed while hashing")
        libraries[str(path)] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": before.st_size,
            "mapped_device": fields[3],
            "mapped_inode": int(fields[4]),
        }
    if not libraries:
        raise ValueError("no loaded cuBLAS library in process maps")
    return [libraries[path] for path in sorted(libraries)]


def tensor_layout(tensor):
    return {
        "shape": list(tensor.shape),
        "strides_elements": list(tensor.stride()),
        "storage_offset_elements": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
    }


def profiler_evidence(events):
    kernels, operators = set(), []
    for event in events:
        device_type = str(event.device_type).rsplit(".", 1)[-1]
        if device_type == "CUDA":
            kernels.add(event.name)
        else:
            kernels.update(kernel.name for kernel in event.kernels)
            operators.append({"name": event.name, "input_shapes": event.input_shapes})
    return {
        "kernel_capture_complete": bool(kernels),
        "kernel_names": sorted(kernels),
        "cpu_events": operators,
    }


def profiled_call(torch, function):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        result = function()
        torch.cuda.synchronize()
    return result, profiler_evidence(profile.events())


def bitwise_comparison(actual, expected):
    if (
        actual.dtype != np.float32
        or expected.dtype != np.float32
        or actual.shape != expected.shape
    ):
        raise ValueError("bridge must preserve original shape and FP32 storage")
    finite = bool(np.isfinite(actual).all())
    unequal = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    return {
        "finite": finite,
        "bitwise_equal": finite and unequal == 0,
        "bitwise_unequal": unequal,
        "numerical": comparison(actual, expected) if finite else None,
    }


def raw_fp32_output_control(torch, w, v, expected, runtime, out, *, provenance):
    """A matching bridge is necessary, not proof of the hidden accumulator."""
    batch, rows, n, channels = v.shape
    lhs = w[:, 0].bfloat16()
    rhs = v.permute(0, 2, 1, 3).reshape(batch, n, rows * channels)
    if lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16:
        raise ValueError("raw FP32 control requires unchanged BF16 operands")

    def stored_layout(result):
        return result.reshape(batch, n, rows, channels).permute(0, 2, 1, 3)

    metadata = {
        "private_execution_metadata": True,
        "provenance": provenance,
        "input_layouts": {"lhs": tensor_layout(lhs), "rhs": tensor_layout(rhs)},
        "autocast_enabled": False,
        "torch_source_git": torch.version.git_version,
        "blas_preference": str(torch.backends.cuda.preferred_blas_library()),
        "tunable_enabled": torch.cuda.tunable.is_enabled(),
        "internal_accumulator_identity_proven": False,
    }
    with (
        torch_policy(
            torch,
            precision="highest",
            reduction=runtime["bf16_reduced_precision_reduction"],
        ),
        torch.autocast("cuda", enabled=False),
    ):
        metadata["bf16_reduced_precision_reduction"] = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )
        metadata["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        baseline, baseline_profile = profiled_call(torch, lambda: torch.bmm(lhs, rhs))
        baseline_stored = stored_layout(baseline).float().cpu().numpy()
        metadata["baseline"] = {
            "comparison": bitwise_comparison(baseline_stored, expected),
            "output_layout": tensor_layout(baseline),
            "profile": baseline_profile,
            "output_sha256": save_array(
                out / "raw_fp32_baseline.npz", output=baseline_stored
            ),
        }
        if (
            baseline.dtype != torch.bfloat16
            or not metadata["baseline"]["comparison"]["bitwise_equal"]
        ):
            save_new(out / "raw_fp32_failed_reproduction.json", metadata)
            raise ValueError("same-BF16-operand baseline did not reproduce original")
        raw, raw_profile = profiled_call(
            torch, lambda: torch.bmm(lhs, rhs, out_dtype=torch.float32)
        )
        if raw.dtype != torch.float32:
            raise ValueError("native raw-output control did not return FP32")
        raw_stored = stored_layout(raw).cpu().numpy()
        rounded = stored_layout(raw.bfloat16()).float().cpu().numpy()
    bridge = bitwise_comparison(rounded, expected)
    output_hash = save_array(
        out / "raw_fp32_output.npz", raw_output=raw_stored, rounded_output=rounded
    )
    metadata.update(
        {
            "capture_complete": True,
            "profiling_complete": (
                baseline_profile["kernel_capture_complete"]
                and raw_profile["kernel_capture_complete"]
            ),
            "bridge_gate_passed": bridge["bitwise_equal"],
            "bridge": bridge,
            "raw_output_layout": tensor_layout(raw),
            "raw_profile": raw_profile,
            "loaded_cublas_libraries": loaded_cublas_libraries(),
            "output_sha256": output_hash,
        }
    )
    path = out / "raw_fp32_control.private.json"
    save_new(path, metadata)
    return {
        "capture_complete": True,
        "profiling_complete": metadata["profiling_complete"],
        "bridge_gate_passed": bridge["bitwise_equal"],
        "bridge": bridge,
        "internal_accumulator_identity_proven": False,
        "private_metadata_sha256": sha(path),
        "output_sha256": metadata["output_sha256"],
    }


def source_binding(root):
    functions = {
        "bench/boltz_pwa_averaging_probe.py": source_binding,
        "bench/af3_closure_capture.py": sha,
        "bench/boltz_closure_capture.py": save_new,
        "bench/boltz_msa_probe.py": verify_bound_file,
        "bench/boltz_relpos_probe.py": comparison,
    }
    for name, function in functions.items():
        if Path(inspect.getfile(function)).resolve() != root / name:
            raise ValueError(f"imported another source snapshot: {name}")
    return {name: sha(root / name) for name in functions}


def native(args):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("native probe requires one GPU")
    report, msa, weights, values, expected = load_reference(
        args.reference, args.msa_reference, args.head
    )
    if torch.__version__ != report["torch"]:
        raise ValueError("Torch version differs from native capture")
    before = source_hashes(args.upstream, "src")
    if before != report["native_source"]:
        raise ValueError("upstream source differs from native capture")
    runtime = msa["runtime"]
    if runtime["float32_matmul_precision"] != "highest":
        raise ValueError("native capture must use highest matmul precision")
    w = torch.from_numpy(weights).cuda()
    v = torch.from_numpy(values).cuda().bfloat16()
    batch, rows, n, channels = v.shape
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    arms = {}
    with torch.inference_mode():
        for label, reduction in native_profiles(
            runtime["bf16_reduced_precision_reduction"], args.reduction_controls
        ):
            with (
                torch_policy(torch, precision="highest", reduction=reduction),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                if label == "native_layout_bmm":
                    rhs = v.permute(0, 2, 1, 3).reshape(batch, n, rows * channels)
                    result = torch.bmm(w[:, 0], rhs)
                    result = result.reshape(batch, n, rows, channels).permute(
                        0, 2, 1, 3
                    )
                else:
                    # Keep the original singleton-head dimension in native replay.
                    result = torch.einsum("bhij,bhsjd->bhsid", w, v[:, None])[:, 0]
                actual_reduction = (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                )
                actual_precision = torch.get_float32_matmul_precision()
            stored = result.float().cpu().numpy()
            metric = comparison(stored, expected)
            arms[label] = {
                "native_original": metric,
                "result_dtype": str(result.dtype),
                "bf16_reduced_precision_reduction": actual_reduction,
                "float32_matmul_precision": actual_precision,
                "output_sha256": save_array(args.out / f"{label}.npz", output=stored),
            }
            if label == "einsum" and not metric["values_equal"]:
                save_new(args.out / "failed-reproduction.json", arms)
                raise ValueError("native einsum did not reproduce captured averaging")
        consumer_weights = w.bfloat16().float().cpu().numpy()
    if not np.array_equal(consumer_weights, bf16_round(weights)):
        raise ValueError("native BF16 consumer conversion differs from recorded input")
    inputs_hash = save_array(
        args.out / "inputs.npz", weights=consumer_weights, values=values
    )
    with torch.inference_mode():
        raw_control = (
            raw_fp32_output_control(
                torch,
                w,
                v,
                expected,
                runtime,
                args.out,
                provenance={
                    "source": args.source_binding,
                    "inputs_sha256": inputs_hash,
                    "reference_sha256": sha(args.reference / "report.json"),
                    "msa_report_sha256": sha(args.msa_reference / "report.json"),
                    "selected_head": args.head,
                },
            )
            if args.raw_fp32_control
            else None
        )
    if source_hashes(args.upstream, "src") != before:
        raise ValueError("upstream source changed during native replay")
    if source_binding(args.source_root) != args.source_binding:
        raise ValueError("benchmark source changed during native replay")
    save_new(
        args.out / "report.json",
        {
            "arm": "native",
            "passed": True,
            "not_model_parity_admission": True,
            "head": args.head,
            "reduction_controls": args.reduction_controls,
            "raw_fp32_control": raw_control,
            "arms": arms,
            "inputs_sha256": inputs_hash,
            "reference_sha256": sha(args.reference / "report.json"),
            "reference_artifacts": report["artifacts"],
            "msa_report_sha256": sha(args.msa_reference / "report.json"),
            "selected_stage_keys": [
                f"head{args.head}/{s}" for s in ("weights", "v", "averaged")
            ],
            "consumer_dtypes": {"weights": "bfloat16", "values": "bfloat16"},
            "consumer_checks": {
                "native_weights_bf16_cast_unequal": 0,
                "values_bf16_storage_unequal": 0,
            },
            "input_shapes": {
                "weights": list(weights.shape),
                "values": list(values.shape),
            },
            "native_bmm_shapes": [[batch, n, n], [batch, n, rows * channels]],
            "native_source": before,
            "native_settings": runtime,
            "cuda_autocast_enabled": True,
            "cuda_autocast_dtype": "torch.bfloat16",
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "source": args.source_binding,
        },
    )
    print(json.dumps(arms))


def foldjax(args):
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("JAX probe requires one GPU")
    root = args.reference
    report = json.loads((root / "report.json").read_text())
    if (
        report.get("arm") != "native"
        or report.get("passed") is not True
        or report.get("arms", {})
        .get("einsum", {})
        .get("native_original", {})
        .get("values_equal")
        is not True
    ):
        raise ValueError("native contraction reproduction must pass first")
    verify_bound_file(root / "inputs.npz", report["inputs_sha256"])
    verify_bound_file(root / "einsum.npz", report["arms"]["einsum"]["output_sha256"])
    with np.load(root / "inputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"weights", "values"}:
            raise ValueError("unreviewed native consumer schema")
        weights, values = archive["weights"], archive["values"]
    with np.load(root / "einsum.npz", allow_pickle=False) as archive:
        expected = archive["output"]
    targets = {"original": expected}
    reduction_arm = report["arms"].get("einsum_reduction_disabled")
    if args.accumulation_controls and reduction_arm is None:
        raise ValueError(
            "accumulation controls require the native reduction-disabled arm"
        )
    if reduction_arm is not None:
        path = root / "einsum_reduction_disabled.npz"
        verify_bound_file(path, reduction_arm["output_sha256"])
        if reduction_arm["bf16_reduced_precision_reduction"] is not False:
            raise ValueError(
                "native reduction-disabled metadata does not match its arm"
            )
        with np.load(path, allow_pickle=False) as archive:
            targets["native_reduction_disabled"] = archive["output"]
    validate_operands(weights, values, expected)
    if report["input_shapes"] != {
        "weights": list(weights.shape),
        "values": list(values.shape),
    } or report["consumer_dtypes"] != {"weights": "bfloat16", "values": "bfloat16"}:
        raise ValueError("consumer metadata differs from captured arrays")
    if not np.array_equal(bf16_round(weights), weights):
        raise ValueError("native consumer weights must be BF16-representable")
    args.out.mkdir(parents=True, exist_ok=False)
    operands = jnp.asarray(weights, jnp.bfloat16), jnp.asarray(values, jnp.bfloat16)
    options = {"xla_allow_excess_precision": False}
    arms = {}
    for name, function, extra_options in jax_profiles(args.accumulation_controls):
        arm_options = {**options, **extra_options}
        with jax.default_matmul_precision("highest"):
            executable = (
                jax.jit(function, compiler_options=arm_options)
                .lower(*operands)
                .compile()
            )
            output = executable(*operands)
            output.block_until_ready()
        stored = np.asarray(output.astype(jnp.float32))
        hlo = args.out / f"{name}.hlo.txt"
        hlo_text = executable.as_text()
        with hlo.open("x") as stream:
            stream.write(hlo_text)
        arms[name] = {
            "native_original": comparison(stored, expected),
            "comparisons": {
                label: comparison(stored, target) for label, target in targets.items()
            },
            "result_dtype": str(output.dtype),
            "compiler_options": arm_options,
            "fp32_accumulator_output_requested": function is fp32_accumulator_average,
            "lowering_evidence": lowering_evidence(hlo_text),
            "output_sha256": save_array(args.out / f"{name}.npz", output=stored),
            "hlo_sha256": sha(hlo),
        }
    if source_binding(args.source_root) != args.source_binding:
        raise ValueError("benchmark source changed during JAX replay")
    save_new(
        args.out / "report.json",
        {
            "arm": "foldjax",
            "capture_complete": True,
            "not_model_parity_admission": True,
            "head": report["head"],
            "accumulation_controls": args.accumulation_controls,
            "arms": arms,
            "reference_sha256": sha(root / "report.json"),
            "inputs_sha256": report["inputs_sha256"],
            "input_shapes": report["input_shapes"],
            "consumer_dtypes": report["consumer_dtypes"],
            "native_bmm_shapes": report["native_bmm_shapes"],
            "matmul_precision": "highest",
            "compiler_options": options,
            "jax": jax.__version__,
            "device": str(jax.devices()[0]),
            "source": args.source_binding,
        },
    )
    print(json.dumps(arms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--msa-reference", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--head", type=int, default=2)
    parser.add_argument("--reduction-controls", action="store_true")
    parser.add_argument("--accumulation-controls", action="store_true")
    parser.add_argument("--raw-fp32-control", action="store_true")
    args = parser.parse_args()
    if args.arm == "native" and (args.msa_reference is None or args.upstream is None):
        parser.error("native requires --msa-reference and --upstream")
    if args.arm == "foldjax" and (
        args.msa_reference is not None or args.upstream is not None
    ):
        parser.error("JAX reads the completed native averaging reference only")
    if args.arm == "native" and args.accumulation_controls:
        parser.error("--accumulation-controls applies to JAX only")
    if args.arm == "foldjax" and args.reduction_controls:
        parser.error("--reduction-controls applies to native only")
    if args.arm == "foldjax" and args.raw_fp32_control:
        parser.error("--raw-fp32-control applies to native only")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.source_root = args.source_root.resolve()
    if args.raw_fp32_control and args.out.resolve().is_relative_to(args.source_root):
        parser.error("private execution artifacts must be outside the source snapshot")
    args.source_binding = source_binding(args.source_root)
    (native if args.arm == "native" else foldjax)(args)
    if source_binding(args.source_root) != args.source_binding:
        raise ValueError("benchmark source changed during run")


if __name__ == "__main__":
    main()
