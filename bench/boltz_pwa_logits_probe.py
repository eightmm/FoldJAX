"""Same-input full PWA logits replay; no model admission or default changes.

Keep the native normalized FP32 input and original FP32 weight, including
head-wise one-column projection. Compiler alternatives are isolated controls.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_embedding_probe import candidate_compile_options
from bench.boltz_msa_probe import source_hashes, verify_bound_file
from bench.boltz_pwa_averaging_probe import bitwise_comparison, profiled_call
from bench.boltz_pwa_production_probe import select_arrays, validate_fp32
from bench.boltz_relpos_probe import bf16_round, torch_policy


def validate_arrays(x, weight, target):
    for value, shape, bf16 in (
        (x, (1, 437, 437, 128), False),
        (weight, (8, 128), False),
        (target, (1, 437, 437, 8), True),
    ):
        if value.shape != shape:
            raise ValueError("requires the full native eight-head logits shape")
        validate_fp32(value, bf16=bf16)
        if not np.isfinite(bf16_round(value)).all():
            raise ValueError("operand overflows native BF16 conversion")


def load_reference(root):
    report_path = root / "report.json"
    digest = sha(report_path)
    report = json.loads(report_path.read_text())
    if (
        report.get("arm") != "native"
        or report.get("passed") is not True
        or report.get("rows") != 4436
        or report.get("native_decomposition", {}).get("values_equal") is not True
        or report.get("row_slice_vs_full", {}).get("values_equal") is not True
    ):
        raise ValueError("requires reproduced full native PWA")
    bindings = {report_path: digest}
    for name in ("stages.npz", "weights.npz"):
        bindings[root / name] = report["artifacts"][name]
    verify_bindings(bindings)
    names = ["norm_z", *(f"head{h}/logits" for h in range(8))]
    values = select_arrays(root / "stages.npz", names)
    x = values.pop("norm_z")
    target = np.concatenate([values[k] for k in names[1:]], axis=-1)
    weight = select_arrays(root / "weights.npz", ["proj_z.weight"])["proj_z.weight"]
    validate_arrays(x, weight, target)
    verify_bindings(bindings)
    return report, x, weight, target, bindings


def verify_bindings(bindings):
    for path, digest in bindings.items():
        verify_bound_file(path, digest)


def candidate_forward(x, kernel):
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives._common import linear

    return jnp.concatenate(
        [linear(x, kernel[:, h : h + 1]) for h in range(8)], axis=-1
    )


def native(args, reference, x, weight):
    import torch

    if (
        torch.__version__ != reference["torch"]
        or torch.cuda.get_device_name() != reference["device"]
    ):
        raise ValueError("native Torch/device differs from original PWA capture")
    if source_hashes(args.source_root, "src") != reference["native_source"]:
        raise ValueError("native source differs from original PWA capture")
    xx, ww = (torch.from_numpy(v.copy()).cuda() for v in (x, weight))

    def forward():
        return torch.cat([xx @ ww[h : h + 1].T for h in range(8)], dim=-1)

    with torch_policy(torch, precision="highest"), torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            forward()
            value, profile = profiled_call(torch, forward)
            repeat = forward()
            outputs = [v.float().cpu().numpy() for v in (value, repeat)]
    if value.dtype != torch.bfloat16 or not profile["kernel_capture_complete"]:
        raise ValueError("native BF16 output and kernel capture are required")
    return outputs, {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "profile": profile,
        "bf16_reduced_precision_reduction": True,
        "float32_matmul_precision": "highest",
    }


def candidate(args, reference, x, weight):
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives import _common

    if Path(inspect.getfile(_common)).resolve() != (
        args.source_root / "src/foldjax/models/boltz2/models/primitives/_common.py"
    ):
        raise ValueError("candidate imported outside selected source")
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("requires one queued GPU")
    if jax.devices()[0].device_kind != reference["device"]:
        raise ValueError("candidate device differs from native capture")
    xx, ww = jnp.asarray(x), jnp.asarray(weight.T.copy(), jnp.bfloat16)
    options = candidate_compile_options(args.profile)
    with jax.default_matmul_precision("highest"):
        executable = (
            jax.jit(candidate_forward, compiler_options=options)
            .lower(xx, ww)
            .compile()
        )
        outputs = [
            np.asarray(executable(xx, ww).block_until_ready(), dtype=np.float32)
            for _ in range(2)
        ]
    hlo = args.out / "compiled.hlo.txt"
    with hlo.open("x") as stream:
        stream.write(executable.as_text())
    return outputs, {
        "jax": jax.__version__,
        "device": jax.devices()[0].device_kind,
        "compiler_options": options,
        "hlo_sha256": sha(hlo),
        "requested_profile": args.profile,
        "profile_is_control_not_runtime_default": args.profile != "baseline",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "candidate"))
    for name in ("reference", "source-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("baseline", "split-k-1", "no-triton"), default="baseline"
    )
    args = parser.parse_args()
    if args.arm == "native" and args.profile != "baseline":
        parser.error("compiler profiles apply only to candidate")
    args.source_root = args.source_root.resolve()
    args.reference = args.reference.resolve()
    args.out = args.out.resolve()
    if any(args.out.is_relative_to(p) for p in (args.source_root, args.reference)):
        parser.error("output must be outside immutable source and reference")
    source_before = source_hashes(args.source_root, "src")
    harness = Path(__file__).resolve().parent
    bindings = {p: sha(p) for p in harness.glob("*.py")}
    reference, x, weight, target, captured = load_reference(args.reference)
    bindings.update(captured)
    args.out.mkdir(parents=True, exist_ok=False)
    outputs, runtime = (native if args.arm == "native" else candidate)(
        args, reference, x, weight
    )
    for value in outputs:
        validate_arrays(x, weight, value)
    measured = bitwise_comparison(outputs[0], target)
    repeated = bitwise_comparison(outputs[0], outputs[1])
    verify_bindings(bindings)
    if source_before != source_hashes(args.source_root, "src"):
        raise ValueError("source changed during replay")
    archive = args.out / "outputs.npz"
    with archive.open("xb") as stream:
        np.savez(stream, actual=outputs[0], repeat=outputs[1])
    result = {
        "arm": args.arm,
        "scope": "same-input full PWA head-wise logits only",
        "model_admission": False,
        "comparison": measured,
        "repeat": repeated,
        "runtime": runtime,
        "source": source_before,
        "bindings": {str(p): digest for p, digest in bindings.items()},
        "outputs_sha256": sha(archive),
        "native_reproduction_passed": (
            measured["bitwise_equal"] and repeated["bitwise_equal"]
            if args.arm == "native"
            else None
        ),
    }
    save_new(args.out / "report.json", result)
    print(json.dumps({"comparison": measured, "repeat": repeated}))
    if args.arm == "native" and not result["native_reproduction_passed"]:
        raise RuntimeError("native standalone logits did not reproduce original PWA")


if __name__ == "__main__":
    main()
