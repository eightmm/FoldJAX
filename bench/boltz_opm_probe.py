"""Matched native first OPM input; isolate native BF16 reduction policy."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_probe import source_hashes, verify_bound_file
from bench.boltz_relpos_probe import arrays, comparison, torch_policy


def execution_profiles(backend_controls=False, native_norm_control=False):
    profiles = [("highest", "highest", {}), ("default", "default", {})]
    if backend_controls:
        no_triton = {"xla_gpu_enable_triton_gemm": False}
        no_lt = {**no_triton, "xla_gpu_enable_cublaslt": False}
        profiles += [
            ("cublaslt", "highest", no_triton),
            ("cublas", "highest", no_lt),
            ("cublas_no_autotune", "highest", {**no_lt, "xla_gpu_autotune_level": 0}),
            ("native_shape", "highest", {}),
            ("native_shape_no_triton", "highest", no_triton),
            (
                "native_shape_no_autotune",
                "highest",
                {**no_lt, "xla_gpu_autotune_level": 0},
            ),
        ]
    if native_norm_control:
        if not backend_controls:
            profiles.append(("native_shape", "highest", {}))
        profiles += [
            ("highest_native_norm", "highest", {}),
            ("native_shape_native_norm", "highest", {}),
            ("production_policy", "highest", {}),
        ]
    return profiles


def load_inputs(root):
    report = json.loads((root / "report.json").read_text())
    if report.get("arm") != "native" or report.get("passed") is not True:
        raise ValueError("native full MSA reproduction must pass first")
    for file in ("operands.npz", "native-weights.npz"):
        verify_bound_file(root / file, report["artifacts"][file])
    for label in ("layer_output", "opm"):
        name = f"layers/00/{label}"
        verify_bound_file(root / f"{name}.npz", report["stages"][name]["arrays_sha256"])
        verify_bound_file(
            root / f"{name}.tree.json", report["stages"][name]["tree_sha256"]
        )
    with np.load(root / "layers/00/layer_output.npz") as archive:
        m = archive["m"]
    with np.load(root / "operands.npz") as archive:
        mask = archive["msa_mask"]
    expected = arrays(root / "layers/00/opm.npz")[""]
    prefix = "msa_module.layers.0.outer_product_mean."
    weights = {
        k.removeprefix(prefix): v
        for k, v in arrays(root / "native-weights.npz").items()
        if k.startswith(prefix)
    }
    if m.dtype != np.float32 or m.shape[2] <= 384:
        raise ValueError("requires native FP32 MSA residual and chunked OPM")
    return report, m, mask, weights, expected


def native(args):
    import torch

    root, upstream = args.reference.resolve(), args.upstream.resolve()
    report, m, mask, weights, expected = load_inputs(root)
    if source_hashes(upstream, "src") != report["native_source"]:
        raise ValueError("upstream source differs from capture")
    sys.path.insert(0, str(upstream / "src"))
    from boltz.model.layers.outer_product_mean import OuterProductMean

    module = OuterProductMean(64, 32, 128).eval()
    module.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True
    )
    module.cuda()
    m, mask = torch.from_numpy(m.copy()).cuda(), torch.from_numpy(mask.copy()).cuda()
    args.out.mkdir(parents=True, exist_ok=False)
    arms = {}
    for reduction in (True, False):
        with (
            torch_policy(torch, reduction=reduction, precision="highest"),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            result = module(m, mask, chunk_size=4)
            torch.cuda.synchronize()
        stored = result.float().cpu().numpy()
        label = "reduction_enabled" if reduction else "reduction_disabled"
        with (args.out / f"{label}.npz").open("xb") as stream:
            np.savez(stream, output=stored)
        arms[label] = {
            "native_original": comparison(stored, expected),
            "output_sha256": sha(args.out / f"{label}.npz"),
        }
    passed = arms["reduction_enabled"]["native_original"]["values_equal"]
    save_new(
        args.out / "report.json",
        {
            "arm": "native",
            "passed": passed,
            "arms": arms,
            "reference_sha256": sha(root / "report.json"),
            "source": report["native_source"],
            "wrapper_sha256": sha(Path(__file__)),
            "torch": torch.__version__,
        },
    )
    print(json.dumps(arms))
    if not passed:
        raise RuntimeError("native default OPM did not reproduce original output")


def foldjax(args):
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.compile_policy import compiler_options
    from foldjax.models.boltz2.models.primitives.native_amp_norm import amp_layer_norm
    from foldjax.models.boltz2.models.trunk_blocks.msa import outer_product_mean_forward

    root = args.reference.resolve()
    _, m, mask, weights, expected = load_inputs(root)
    native_report = json.loads((args.native_control / "report.json").read_text())
    if native_report.get("passed") is not True or native_report[
        "reference_sha256"
    ] != sha(root / "report.json"):
        raise ValueError("matched native OPM control missing")
    targets = {"original": expected}
    for label, arm in native_report["arms"].items():
        verify_bound_file(args.native_control / f"{label}.npz", arm["output_sha256"])
        targets[label] = arrays(args.native_control / f"{label}.npz")["output"]
    params = {}
    for name, value in weights.items():
        module, leaf = name.split(".")
        key = "bias" if leaf == "bias" else "scale" if module == "norm" else "kernel"
        params.setdefault(module, {})[key] = jnp.asarray(
            value.T if key == "kernel" else value,
            dtype=jnp.bfloat16 if key == "kernel" else jnp.float32,
        )
    m, mask = jnp.asarray(m), jnp.asarray(mask)
    args.out.mkdir(parents=True, exist_ok=False)
    arms = {}
    for label, precision, extra_options in execution_profiles(
        args.backend_controls, args.native_norm_control
    ):
        token_chunk = m.shape[2] if label.startswith("native_shape") else 128

        def run(p, m, mask):
            return outer_product_mean_forward(
                p,
                m,
                mask,
                chunk_size=token_chunk,
                preserve_native_amp_shape=label == "production_policy",
            )

        norm_context = (
            patch(
                "foldjax.models.boltz2.models.trunk_blocks.msa._layer_norm",
                amp_layer_norm,
            )
            if label.endswith("_native_norm")
            else nullcontext()
        )
        with jax.default_matmul_precision(precision), norm_context:
            options = {**compiler_options("bfloat16"), **extra_options}
            executable = (
                jax.jit(run, compiler_options=options).lower(params, m, mask).compile()
            )
            result = executable(params, m, mask)
            result.block_until_ready()
        stored = np.asarray(result)
        with (args.out / f"{label}.npz").open("xb") as stream:
            np.savez(stream, output=stored)
        with (args.out / f"{label}.hlo.txt").open("x") as stream:
            stream.write(executable.as_text())
        arms[label] = {
            "comparisons": {k: comparison(stored, v) for k, v in targets.items()},
            "output_sha256": sha(args.out / f"{label}.npz"),
            "hlo_sha256": sha(args.out / f"{label}.hlo.txt"),
            "compiler_options": options,
            "token_chunk_size": token_chunk,
            "native_norm_control": label.endswith("_native_norm"),
            "preserve_native_amp_shape": label == "production_policy",
        }
    save_new(
        args.out / "report.json",
        {
            "arm": "foldjax",
            "capture_complete": True,
            "not_model_parity_admission": True,
            "arms": arms,
            "reference_sha256": sha(root / "report.json"),
            "native_report_sha256": sha(args.native_control / "report.json"),
            "source": source_hashes(Path(__file__).resolve().parents[1], "src"),
            "wrapper_sha256": sha(Path(__file__)),
            "compiler_options": compiler_options("bfloat16"),
        },
    )
    print(json.dumps(arms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--native-control", type=Path)
    parser.add_argument("--backend-controls", action="store_true")
    parser.add_argument("--native-norm-control", action="store_true")
    args = parser.parse_args()
    if args.arm == "native" and args.upstream is None:
        parser.error("native arm requires --upstream")
    if args.arm == "native" and args.native_norm_control:
        parser.error("native norm control applies to FoldJAX only")
    if args.arm == "foldjax" and args.native_control is None:
        parser.error("FoldJAX arm requires --native-control")
    (native if args.arm == "native" else foldjax)(args)


if __name__ == "__main__":
    main()
