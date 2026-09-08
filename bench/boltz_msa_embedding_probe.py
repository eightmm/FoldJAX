"""Actual Boltz s_proj replay on the full captured [1,437,384] input.

This isolates the small broadcast projection before PWA. It does not replace
the native GEMM by an exact dot or change the installed model. Reconstruction
uses a separately labelled CPU sparse-projection formula, not native GEMM
evidence. Both frameworks retain their original BF16 rounding policy.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_probe import (
    FEATURES,
    profile_from_hparams,
    source_hashes,
    verify_bound_file,
)
from bench.boltz_pwa_averaging_probe import (
    bitwise_comparison,
    loaded_cublas_libraries,
    lowering_evidence,
    profiled_call,
    save_array,
    tensor_layout,
)
from bench.boltz_relpos_probe import bf16_round, comparison, torch_policy

SUSPECTED = ((18, 53), (217, 2), (258, 54), (261, 3))
SHAPE = (1, 4436, 437, 64)
PROFILES = ("baseline", "split-k-1", "no-triton")


def validate_projection(emb, weight, output=None):
    for name, value, shape in (
        ("emb", emb, (1, 437, 384)),
        ("weight", weight, (64, 384)),
    ):
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError(f"{name} must retain full native FP32 shape {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite {name}")
        if not np.isfinite(bf16_round(value)).all():
            raise ValueError(f"{name} overflows native BF16 conversion")
    if output is not None and (
        output.shape != (1, 437, 64)
        or output.dtype != np.float32
        or not np.isfinite(output).all()
        or not np.array_equal(bf16_round(output), output)
    ):
        raise ValueError("output must retain all 27968 BF16 values in FP32 storage")


def harness_hashes():
    # Include transitive local evidence helpers, not just this CLI's bytes.
    directory = Path(__file__).resolve().parent
    return {p.name: sha(p) for p in sorted(directory.glob("*.py"))}


def reference_files(root, report):
    stage = report["stages"]["layers/00/input_m"]
    return {
        "report.json": sha(root / "report.json"),
        "operands.npz": report["artifacts"]["operands.npz"],
        "native-weights.npz": report["artifacts"]["native-weights.npz"],
        "layers/00/input_m.npz": stage["arrays_sha256"],
        "layers/00/input_m.tree.json": stage["tree_sha256"],
    }


def verify_files(root, bindings):
    for name, digest in bindings.items():
        verify_bound_file(root / name, digest)


def load_reference(root):
    report = json.loads((root / "report.json").read_text())
    runtime = report.get("runtime", {})
    if (
        report.get("arm") != "native"
        or report.get("passed") is not True
        or runtime.get("float32_matmul_precision") != "highest"
        or runtime.get("cuda_allow_tf32") is not False
        or runtime.get("bf16_reduced_precision_reduction") is not True
    ):
        raise ValueError("reference must be reproduced native BF16/highest MSA")
    bindings = reference_files(root, report)
    verify_files(root, bindings)
    tree = json.loads((root / "layers/00/input_m.tree.json").read_text())
    if tree != {
        "": {
            "native_dtype": "torch.bfloat16",
            "storage_dtype": "float32",
            "shape": list(SHAPE),
        }
    }:
        raise ValueError("reference must retain the original complete S4436 MSA")
    with np.load(root / "operands.npz", allow_pickle=False) as archive:
        values = {k: archive[k] for k in (*FEATURES, "emb")}
    with np.load(root / "native-weights.npz", allow_pickle=False) as archive:
        weight = archive["msa_module.s_proj.weight"]
        msa_weight = archive["msa_module.msa_proj.weight"]
    validate_projection(values["emb"], weight)
    validate_reconstruction(values, msa_weight, SHAPE)
    return report, bindings, values, weight, msa_weight


def validate_reconstruction(values, weight, shape):
    if len(shape) != 4 or min(shape) < 1 or shape[-1] != 64:
        raise ValueError("reconstruction requires nonempty batch/MSA/token and C64")
    if weight.shape != (64, 36) or weight.dtype != np.float32:
        raise ValueError("original msa_proj FP32 weight must be [64,36]")
    if not np.isfinite(weight).all() or not np.isfinite(bf16_round(weight)).all():
        raise ValueError("nonfinite msa_proj weight")
    for key in ("msa", "has_deletion", "deletion_value", "msa_paired"):
        x = values[key]
        if x.shape != shape[:3] or x.dtype.kind not in "bifu":
            raise ValueError(f"invalid reconstruction feature {key}")
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite reconstruction feature {key}")
        if key != "msa" and not np.isfinite(bf16_round(x)).all():
            raise ValueError(f"feature overflows native BF16 conversion: {key}")
    msa = values["msa"]
    if msa.dtype.kind not in "iu" or msa.min() < 0 or msa.max() >= 33:
        raise ValueError("MSA IDs must remain integer tokens in [0,33)")
    for key in ("has_deletion", "msa_paired"):
        if not np.isin(values[key], [0, 1]).all():
            raise ValueError(f"nonbinary feature {key}")


def reconstruct_embedding(values, weight, projection, expected, row_chunk=64):
    """Compare a CPU four-term formula, preserving both BF16 storage casts.

    FP64 sums here are a counterfactual, not an emulation of cuBLAS's carried
    accumulator. A failure remains recorded and cannot invalidate or repair a
    separately observed native s_proj output.
    """
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    validate_reconstruction(values, weight, expected.shape)
    if expected.shape[-1] != 64 or projection.shape != (
        expected.shape[0],
        expected.shape[2],
        64,
    ):
        raise ValueError("projection/reconstruction shape mismatch")
    if expected.dtype != np.float32 or not np.isfinite(expected).all():
        raise ValueError("reconstruction reference must be finite FP32 storage")
    if not np.isfinite(projection).all() or not np.array_equal(
        bf16_round(projection), projection
    ):
        raise ValueError("projection must be finite BF16-representable")
    w = bf16_round(weight).astype(np.float64)
    result = {"max_abs": 0.0, "rmse": 0.0, "unequal": 0, "values_equal": True}
    sum_squares, count = 0.0, 0
    for start in range(0, expected.shape[1], row_chunk):
        section = (slice(None), slice(start, start + row_chunk))
        sparse = w[:, values["msa"][section]].transpose(1, 2, 3, 0).copy()
        for index, key in enumerate(("has_deletion", "deletion_value", "msa_paired")):
            extra = bf16_round(values[key][section]).astype(np.float64)
            sparse += extra[..., None] * w[:, 33 + index]
        m = bf16_round(sparse.astype(np.float32))
        actual = bf16_round(m + projection[:, None])
        target = expected[section]
        metrics = comparison(actual, target)
        result["max_abs"] = max(result["max_abs"], metrics["max_abs"])
        result["unequal"] += metrics["unequal"]
        result["values_equal"] &= metrics["values_equal"]
        sum_squares += metrics["rmse"] ** 2 * target.size
        count += target.size
    result["rmse"] = float(np.sqrt(sum_squares / count))
    return {"formula_only": True, "elements": count, "comparison": result}


def suspected_scalars(emb, weight, actual):
    exact = bf16_round(emb).astype(np.float64) @ bf16_round(weight).astype(np.float64).T
    rounded = bf16_round(exact.astype(np.float32))
    return [
        {
            "token": t,
            "channel": c,
            "observed": float(actual[0, t, c]),
            "cpu_exact_bf16_operand_dot": float(exact[0, t, c]),
            "cpu_exact_dot_fp32_then_bf16": float(rounded[0, t, c]),
            "cpu_value_is_counterfactual_not_native_capture": True,
        }
        for t, c in SUSPECTED
    ]


def candidate_projection(emb, kernel):
    from foldjax.models.boltz2.models.primitives._common import linear

    return linear(emb, kernel)


def candidate_compile_options(profile):
    from foldjax.models.boltz2.compile_policy import compiler_options

    options = compiler_options("bfloat16")
    if options != {"xla_allow_excess_precision": False}:
        raise ValueError("production BF16 rounding policy changed")
    if profile == "split-k-1":
        # JAX 0.11.1's installed XLA flag help: zero uses the heuristic;
        # positive values force that split count. One does not disable Triton.
        options["xla_gpu_experimental_force_split_k"] = 1
    elif profile == "no-triton":
        options["xla_gpu_enable_triton_gemm"] = False
    elif profile != "baseline":
        raise ValueError(f"unknown candidate profile: {profile}")
    return options


def projection_lowering_evidence(hlo, profile):
    """Check this fixed projection's emitted split graph, not the flag name.

    A batched [factor,437,64] partial dot followed by reduction is split-K for
    this originally unbatched GEMM. A custom cuBLAS call hides its internal
    accumulation policy; absence of visible split-K is not proof about cuBLAS.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown candidate profile: {profile}")
    evidence = lowering_evidence(hlo)
    factors, unclassified = [], []
    for line in evidence["dot_instructions"]:
        match = re.search(r"= (?:bf16|f32)\[([\d,]+)\]", line)
        shape = tuple(map(int, match[1].split(","))) if match else ()
        if "batch_dims=" in line:
            if len(shape) == 3 and shape[1:] in ((437, 64), (64, 437)):
                factors.append(shape[0])
            else:
                unclassified.append(line)
        elif shape not in ((437, 64), (64, 437), (1, 437, 64)):
            unclassified.append(line)
    reductions = [line.strip() for line in hlo.splitlines() if " reduce(" in line]
    external_blas = any(
        "blas" in name.lower() for name in evidence["custom_call_targets"]
    )
    verified = True
    if profile == "split-k-1":
        verified = (
            evidence["contains_triton_gemm"]
            and bool(evidence["dot_instructions"])
            and not any(factor > 1 for factor in factors)
            and not unclassified
            and not reductions
        )
    elif profile == "no-triton":
        verified = not evidence["contains_triton_gemm"] and external_blas
    return {
        **evidence,
        "requested_profile": profile,
        "visible_split_k_factors": sorted(set(factors)),
        "reduction_instructions": reductions,
        "unclassified_dot_instructions": unclassified,
        "contains_external_blas": external_blas,
        "external_blas_internal_split_k_observable": False,
        "requested_profile_verified": bool(verified),
    }


def native(args, reference, values, weight):
    import torch

    source = args.source_root.resolve()
    before = source_hashes(source, "src")
    if before != reference["native_source"]:
        raise ValueError("native source differs from original reproduced MSA")
    sys.path.insert(0, str(source / "src"))
    from boltz.model.modules.trunkv2 import MSAModule

    if Path(inspect.getfile(MSAModule)).resolve() != (
        source / "src/boltz/model/modules/trunkv2.py"
    ):
        raise ValueError("another native MSAModule was imported")
    profile = reference["profile"]
    # Reuse the pinned profile validator; no changed MSA configuration is admitted.
    profile_from_hparams(
        {
            "token_s": profile["token_s"],
            "token_z": profile["token_z"],
            "msa_args": {
                k: v for k, v in profile.items() if k not in {"token_s", "token_z"}
            },
        }
    )
    model = MSAModule(**profile).eval()
    projection = model.s_proj
    projection.load_state_dict({"weight": torch.from_numpy(weight.copy())}, strict=True)
    if projection.bias is not None or projection.weight.dtype != torch.float32:
        raise ValueError(
            "native s_proj must remain bias-free with original FP32 weight"
        )
    projection = projection.cuda()
    emb = torch.from_numpy(values["emb"].copy()).cuda()
    runtime = reference["runtime"]
    if (
        torch.__version__ != runtime["torch"]
        or torch.version.cuda != runtime["cuda"]
        or torch.cuda.get_device_name() != runtime["device"]
    ):
        raise ValueError("native Torch/CUDA/device differs from original capture")
    outputs, profiles = [], []
    reduction = not args.disable_bf16_reduced_precision
    with (
        torch_policy(torch, reduction=reduction, precision="highest"),
        torch.inference_mode(),
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(2):
                result, events = profiled_call(torch, lambda: projection(emb))
                if result.dtype != torch.bfloat16:
                    raise ValueError("native autocast s_proj output is not BF16")
                outputs.append(result.detach().float().cpu().numpy())
                profiles.append(events)
            metadata = {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "bf16_reduced_precision_reduction": (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                ),
                "autocast_enabled": torch.is_autocast_enabled("cuda"),
                "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                "input": tensor_layout(emb),
                "weight": tensor_layout(projection.weight),
                "output": tensor_layout(result),
                "profiles": profiles,
                "control": "bf16_reduction_disabled" if not reduction else "none",
                "torch_source_git": torch.version.git_version,
                "cublas_libraries": loaded_cublas_libraries(),
            }
    if source_hashes(source, "src") != before:
        raise ValueError("native source changed during projection")
    return outputs, metadata, before


def candidate(args, reference, values, weight):
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives import _common

    source = args.source_root.resolve()
    if Path(inspect.getfile(_common)).resolve() != (
        source / "src/foldjax/models/boltz2/models/primitives/_common.py"
    ):
        raise ValueError("another FoldJAX linear implementation was imported")
    before = source_hashes(source, "src")
    if jax.devices()[0].platform != "gpu":
        raise ValueError("candidate evidence requires the queued GPU route")
    expected_device = reference["runtime"]["device"]
    if jax.devices()[0].device_kind != expected_device:
        raise ValueError("candidate device differs from native capture")
    options = candidate_compile_options(args.candidate_profile)
    emb = jnp.asarray(values["emb"])
    kernel = jnp.asarray(weight.T.copy(), dtype=jnp.bfloat16)
    with jax.default_matmul_precision("highest"):
        executable = (
            jax.jit(candidate_projection, compiler_options=options)
            .lower(emb, kernel)
            .compile()
        )
        hlo = executable.as_text()
        outputs = [
            np.asarray(executable(emb, kernel).block_until_ready(), dtype=np.float32)
            for _ in range(2)
        ]
    hlo_path = args.out / "candidate.hlo.txt"
    with hlo_path.open("x") as stream:
        stream.write(hlo)
    if source_hashes(source, "src") != before:
        raise ValueError("FoldJAX source changed during projection")
    return (
        outputs,
        {
            "jax": jax.__version__,
            "device": jax.devices()[0].device_kind,
            "compiler_options": options,
            "float32_matmul_precision": "highest",
            "control": (
                "none"
                if args.candidate_profile == "baseline"
                else args.candidate_profile
            ),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "input_dtype": str(emb.dtype),
            "kernel_dtype": str(kernel.dtype),
            "input_shape": list(emb.shape),
            "kernel_shape": list(kernel.shape),
            "hlo_sha256": sha(hlo_path),
            "lowering": projection_lowering_evidence(hlo, args.candidate_profile),
        },
        before,
    )


def load_native_projection(root, reference_binding, emb, weight):
    report = json.loads((root / "report.json").read_text())
    if (
        report.get("arm") != "native"
        or report.get("passed") is not True
        or report.get("runtime", {}).get("control") != "none"
        or report.get("reference_files") != reference_binding
    ):
        raise ValueError("candidate requires the unchanged native baseline projection")
    bindings = {
        "report.json": sha(root / "report.json"),
        "operands.npz": report["artifacts"]["operands.npz"],
        "outputs.npz": report["artifacts"]["outputs.npz"],
    }
    verify_files(root, bindings)
    with np.load(root / "operands.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"emb", "weight"} or any(
            archive[key].dtype != value.dtype or not np.array_equal(archive[key], value)
            for key, value in (("emb", emb), ("weight", weight))
        ):
            raise ValueError("native projection operands changed")
    with np.load(root / "outputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"first", "repeat"}:
            raise ValueError("native projection repeat archive is incomplete")
        first, repeat = archive["first"], archive["repeat"]
    for value in (first, repeat):
        validate_projection(emb, weight, value)
    if not bitwise_comparison(first, repeat)["bitwise_equal"]:
        raise ValueError("native projection repeat was not exact")
    return first, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "candidate"), required=True)
    parser.add_argument("--msa-reference", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable-bf16-reduced-precision", action="store_true")
    parser.add_argument("--candidate-profile", choices=PROFILES, default="baseline")
    args = parser.parse_args()
    if args.mode == "candidate" and (
        args.reference is None or args.disable_bf16_reduced_precision
    ):
        parser.error("candidate requires --reference and unchanged reduction policy")
    if args.mode == "native" and args.reference is not None:
        parser.error("native projection uses --msa-reference, not --reference")
    if args.mode == "native" and args.candidate_profile != "baseline":
        parser.error("candidate compiler controls cannot change native projection")
    harness = harness_hashes()
    root = args.msa_reference.resolve()
    report, bindings, values, weight, msa_weight = load_reference(root)
    native_output, native_bindings = None, None
    if args.mode == "candidate":
        native_output, native_bindings = load_native_projection(
            args.reference, bindings, values["emb"], weight
        )
    args.out.mkdir(parents=True, exist_ok=False)
    operand_sha = save_array(
        args.out / "operands.npz", emb=values["emb"], weight=weight
    )
    outputs, runtime, source = (native if args.mode == "native" else candidate)(
        args, report, values, weight
    )
    for output in outputs:
        validate_projection(values["emb"], weight, output)
    output_sha = save_array(
        args.out / "outputs.npz", first=outputs[0], repeat=outputs[1]
    )
    with np.load(root / "layers/00/input_m.npz", allow_pickle=False) as archive:
        expected_m = archive[""]
    reconstruction = reconstruct_embedding(values, msa_weight, outputs[0], expected_m)
    repeat = bitwise_comparison(*outputs)
    verify_files(root, bindings)
    if native_bindings is not None:
        verify_files(args.reference, native_bindings)
    if harness_hashes() != harness:
        raise ValueError("probe/evidence helper source changed during execution")
    if source_hashes(args.source_root.resolve(), "src") != source:
        raise ValueError("runtime source changed during the complete diagnostic")
    parity = (
        bitwise_comparison(outputs[0], native_output)
        if native_output is not None
        else None
    )
    result = {
        "arm": args.mode,
        "capture_complete": True,
        "passed": repeat["bitwise_equal"]
        and (parity is None or parity["bitwise_equal"])
        and runtime.get("lowering", {}).get("requested_profile_verified", True),
        "scope": "same-operand s_proj only; no model/performance admission",
        "native_python_source_of_reference": report["native_source"],
        "reference_files": bindings,
        "native_projection_files": native_bindings,
        "runtime": runtime,
        "source": source,
        "harness_source": harness,
        "original_checkpoint_sha256": report["checkpoint_sha256"],
        "artifacts": {"operands.npz": operand_sha, "outputs.npz": output_sha},
        "repeat": repeat,
        "candidate_vs_native": parity,
        "suspected_scalars": suspected_scalars(values["emb"], weight, outputs[0]),
        "embedding_reconstruction": reconstruction,
    }
    save_new(args.out / "report.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "passed",
                    "repeat",
                    "candidate_vs_native",
                    "suspected_scalars",
                    "embedding_reconstruction",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
