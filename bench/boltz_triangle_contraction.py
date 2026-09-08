"""Same-operand triangle contraction diagnostic, not model admission."""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_probe import verify_bound_file
from bench.boltz_pwa_averaging_probe import bitwise_comparison, profiled_call
from bench.boltz_pwa_production_probe import select_arrays
from bench.boltz_relpos_probe import bf16_round, torch_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "candidate"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-triton", action="store_true")
    args = parser.parse_args()
    if args.no_triton and args.arm != "candidate":
        parser.error("compiler control applies only to candidate")
    report_path = args.reference / "report.json"
    digest = sha(report_path)
    report = json.loads(report_path.read_text())
    if not report["comparison"]["bitwise_equal"]:
        raise ValueError("requires reproduced native triangle")
    archive = args.reference / "stages.npz"
    verify_bound_file(archive, report["stages_sha256"])
    values = select_arrays(archive, ["fused_sigmoid_gated_dual_gemm/0", "contraction"])
    ab, expected = values["fused_sigmoid_gated_dual_gemm/0"], values["contraction"]
    if ab.shape != (256, 1, 437, 437) or expected.shape != (128, 1, 437, 437):
        raise ValueError("requires full 437-token outgoing contraction")
    if not all(
        np.isfinite(v).all() and np.array_equal(v, bf16_round(v))
        for v in (ab, expected)
    ):
        raise ValueError("requires finite BF16 operands and target")
    args.out.mkdir(parents=True, exist_ok=False)
    if args.arm == "native":
        import torch

        x = torch.from_numpy(ab).cuda().bfloat16()
        a, b = x.chunk(2, dim=0)
        with torch_policy(torch, precision="highest"), torch.inference_mode():
            def fn():
                return torch.einsum("dbik,dbjk->dbij", a, b)
            fn()
            actual, profile = profiled_call(torch, fn)
            repeat = fn()
        actual, repeat = [v.float().cpu().numpy() for v in (actual, repeat)]
        runtime = {
            "torch": torch.__version__,
            "profile": profile,
            "device": torch.cuda.get_device_name(),
        }
    else:
        import jax
        import jax.numpy as jnp

        if jax.default_backend() != "gpu":
            raise ValueError("requires queued GPU")
        a, b = jnp.split(jnp.asarray(ab, jnp.bfloat16), 2, axis=0)
        with jax.default_matmul_precision("highest"):
            executable = (
                jax.jit(
                    lambda a, b: jnp.einsum("dbik,dbjk->dbij", a, b),
                    compiler_options={"xla_gpu_enable_triton_gemm": False}
                    if args.no_triton else None,
                )
                .lower(a, b)
                .compile()
            )
            actual, repeat = [
                np.asarray(executable(a, b), np.float32) for _ in range(2)
            ]
        with (args.out / "compiled.hlo.txt").open("x") as stream:
            stream.write(executable.as_text())
        runtime = {
            "jax": jax.__version__,
            "device": jax.devices()[0].device_kind,
            "hlo_sha256": sha(args.out / "compiled.hlo.txt"),
        }
    verify_bound_file(archive, report["stages_sha256"])
    verify_bound_file(report_path, digest)
    save_new(
        args.out / "report.json",
        {
            "comparison": bitwise_comparison(actual, expected),
            "repeat": bitwise_comparison(actual, repeat),
            "runtime": runtime,
            "reference_sha256": digest,
            "source_sha256": sha(Path(__file__)),
            "not_model_parity_admission": True,
            "no_triton_control": args.no_triton,
        },
    )


if __name__ == "__main__":
    main()
