"""Native reduction-policy counterfactual on the actual complete first w3 input."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from bench.boltz_amp_report import compare_arrays
from bench.esmfold2_lm_encoder_candidate import (
    compare_boundaries,
    validate_capture,
    validate_transition_projection,
)
from bench.esmfold2_tape import _save_npz, _sha256


def kernel_launches(trace):
    return [
        {
            "name": event["name"],
            "launch": {
                key: event.get("args", {}).get(key)
                for key in ("grid", "block", "registers per thread", "shared memory")
            },
        }
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-kernels", action="store_true")
    for name in ("native", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report, arrays = validate_capture(args.native, args.weights)
    validate_transition_projection(report, arrays)
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
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one native CUDA device is required")
    if (
        str(torch.__version__) != report["torch"]
        or torch.version.git_version != report["torch_git"]
    ):
        raise ValueError("native Torch runtime differs from capture")
    torch.set_float32_matmul_precision(report["matmul"])
    torch.backends.cuda.matmul.allow_tf32 = report["allow_tf32"]
    original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    if (
        original != report["allow_bf16_reduced_precision_reduction"]
        or report["precision_control"]
    ):
        raise ValueError("reference must retain native reduction default")
    prefix = "lm_encoder.blocks.0.pair_transition.ffn.w3"
    with safe_open(args.weights / "model.safetensors", framework="numpy") as handle:
        weight = torch.from_numpy(handle.get_tensor(prefix + ".weight")).cuda()
        bias = (
            torch.from_numpy(handle.get_tensor(prefix + ".bias")).cuda()
            if prefix + ".bias" in handle.keys()
            else None
        )
    value = (
        torch.from_numpy(arrays["transition_projection.input"])
        .cuda()
        .to(torch.bfloat16)
    )
    schema = report["boundaries"]["transition_projection.input"]
    if (
        list(value.stride()) != schema["native_stride"]
        or value.storage_offset() != schema["native_storage_offset"]
    ):
        raise ValueError(
            "native projection replay must preserve actual input storage layout"
        )
    expected = arrays["transition_projection.output"]
    del arrays
    outputs, profiles = {}, {}
    try:
        for label, reduction in (
            ("native_default", original),
            ("reduction_false", False),
        ):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                reduction
            )
            for repeat in range(2):
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    result = functional.linear(value, weight, bias)
                if result.dtype != torch.bfloat16:
                    raise ValueError("native projection output must be BF16")
                outputs[f"{label}.{repeat}"] = result.float().cpu().numpy().copy()
            if args.profile_kernels:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        profiled = functional.linear(value, weight, bias)
                    torch.cuda.synchronize()
                unchanged = (
                    profiled.float().cpu().numpy().tobytes()
                    == outputs[label + ".0"].tobytes()
                )
                profiles[label] = {
                    "output_bytes_equal": unchanged,
                    "cuda_kernels": sorted(
                        {
                            event.name
                            for event in profile.events()
                            if "CUDA" in str(event.device_type)
                        }
                    ),
                    "performance_evidence": False,
                }
                if not unchanged:
                    raise ValueError("profiling changed the native projection output")
                trace_path = args.output / f"{label}.trace.json"
                profile.export_chrome_trace(str(trace_path))
                profiles[label]["launches"] = kernel_launches(
                    json.loads(trace_path.read_text())
                )
                profiles[label]["trace_sha256"] = _sha256(trace_path)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original
    comparison = compare_boundaries(
        {key: expected for key in outputs},
        outputs,
        {key: {"original_dtype": "torch.bfloat16"} for key in outputs},
        {key: {"dtype": "torch.bfloat16"} for key in outputs},
    )
    if before != {str(p.resolve()): _sha256(p) for p in paths}:
        raise ValueError("reference/checkpoint/runner changed during replay")
    _save_npz(args.output / "outputs.npz", outputs)
    result = {
        "scope": "same_complete_native_first_transition_w3_input",
        "model_admission": None,
        "bindings": before,
        "native_default_reduction": original,
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "comparison_to_native_block": comparison,
        "kernel_profiles": profiles,
        "repeat_bytes_equal": {
            label: outputs[label + ".0"].tobytes() == outputs[label + ".1"].tobytes()
            for label in ("native_default", "reduction_false")
        },
        "outputs_sha256": _sha256(args.output / "outputs.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
