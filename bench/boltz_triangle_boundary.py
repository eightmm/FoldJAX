"""Reproduce first outgoing triangle and capture native intermediate boundaries."""

import argparse
import importlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_probe import verify_bound_file
from bench.boltz_pwa_averaging_probe import bitwise_comparison
from bench.boltz_pwa_production_probe import select_arrays
from bench.boltz_relpos_probe import torch_policy


def native_vector4_mean(x):
    """Observed FP32 PTX sum order; no buffer-alignment assumptions."""
    if x.dtype != np.float32 or x.shape[-1] != 128:
        raise ValueError("requires FP32 width 128")
    values = (x[..., :64] + x[..., 64:]).reshape(*x.shape[:-1], 16, 4)
    total = ((values[..., 0] + values[..., 1]) + values[..., 2]) + values[..., 3]
    for offset in (8, 4, 2, 1):
        total = total[..., :offset] + total[..., offset:]
    return total[..., 0] / np.float32(128)


def candidate(args, x, mask, weights, prefix, target, bindings):
    import inspect

    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.triangle import triangle_cueq as module

    if jax.default_backend() != "gpu":
        raise ValueError("requires queued CUDA execution")
    native_report_path = args.candidate_reference / "report.json"
    native_report = json.loads(native_report_path.read_text())
    if not native_report["comparison"]["bitwise_equal"]:
        raise ValueError("native standalone reproduction must pass")
    native_stages = args.candidate_reference / "stages.npz"
    bindings[native_report_path] = sha(native_report_path)
    bindings[native_stages] = native_report["stages_sha256"]
    bindings[Path(inspect.getfile(module))] = sha(Path(inspect.getfile(module)))
    for path, digest in bindings.items():
        verify_bound_file(path, digest)
    params = {}
    for name, value in weights.items():
        part, field = name.removeprefix(prefix).split(".")
        key = "scale" if part.startswith("norm") and field == "weight" else field
        if not part.startswith("norm"):
            key, value = "kernel", value.T
        params.setdefault(part, {})[key] = jnp.asarray(
            value, jnp.float32 if part.startswith("norm") else jnp.bfloat16
        )
    cuex = module._load_cueq()
    norm, gemm, dual = module._load_cueq_amp_primitives()
    if args.explicit_input_norm:
        from foldjax.models.boltz2.models.primitives.native_cueq_norm import (
            native_cueq_norm,
        )

        original_norm = norm

        def norm(x, s, b, **kw):
            if kw.get("layout") == "bijd->bijd" and x.dtype == jnp.float32:
                return native_cueq_norm(x, s, b, kw.get("eps", 1e-5))[0]
            return original_norm(x, s, b, **kw)

    def forward(x, mask, p):
        if x.shape == (1, 437, 437, 128):
            from foldjax.models.boltz2.models.primitives.native_cueq_norm import (
                native_cueq_norm as runtime_norm,
            )

            norm_in = runtime_norm(x, p["norm_in"]["scale"], p["norm_in"]["bias"])[0]
        else:
            norm_in = norm(
                x,
                p["norm_in"]["scale"],
                p["norm_in"]["bias"],
                eps=1e-5,
                layout="bijd->bijd",
                fallback=False,
            )
        xi = norm_in.astype(jnp.bfloat16)
        precision = module.triangle_multiplication_precision(cuex, dtype=xi.dtype)
        ab = gemm(
            xi,
            p["g_in"]["kernel"].T,
            p["p_in"]["kernel"].T,
            mask=mask,
            transpose_out=True,
            precision=precision,
            fallback=False,
        )
        a, b = jnp.split(ab, 2, axis=0)
        contracted = jnp.einsum("dbik,dbjk->dbij", a, b)
        from foldjax.models.boltz2.models.primitives.native_cueq_norm import (
            native_cueq_output_norm,
        )

        norm_out = native_cueq_output_norm(
            contracted, p["norm_out"]["scale"], p["norm_out"]["bias"]
        )[0]
        out = dual(
            xi,
            norm_out.astype(jnp.bfloat16),
            p["g_out"]["kernel"].T,
            p["p_out"]["kernel"].T,
            precision=precision,
            fallback=False,
        )
        return {
            "layer_norm_transpose/0": norm_in,
            "layer_norm_transpose/1": norm_out,
            "fused_sigmoid_gated_dual_gemm/0": ab,
            "contraction": contracted,
            "fused_sigmoid_gated_dual_gemm_dual_x/0": out,
            "output": out,
        }

    args.out.mkdir(parents=True, exist_ok=False)
    xx, mm = jnp.asarray(x), jnp.asarray(mask)
    with jax.default_matmul_precision("highest"), ExitStack() as stack:
        if args.explicit_input_norm:
            stack.enter_context(
                patch.object(
                    module, "_load_cueq_amp_primitives", lambda: (norm, gemm, dual)
                )
            )
        if args.norm_alignment_control:
            from cuequivariance_jax.triangle import triton_utils

            original_source = triton_utils.tc.ASTSource

            class AlignedSource(original_source):
                def __init__(self, fn, signature, **kw):
                    if fn.__name__ == "layer_norm_transpose_forward_kernel":
                        attrs = dict(kw.pop("attrs", {}) or {})
                        for name, dtype in signature.items():
                            if dtype.startswith("*"):
                                attrs[(fn.arg_names.index(name),)] = [
                                    ["tt.divisibility", 16]
                                ]
                        kw["attrs"] = attrs
                    super().__init__(fn, signature, **kw)

            stack.enter_context(
                patch.object(triton_utils.tc, "ASTSource", AlignedSource)
            )
        if args.norm_fp_fusion:
            norm_module = importlib.import_module(
                "cuequivariance_jax.triangle._layer_norm_transpose"
            )
            original = norm_module.triton_call

            def fused_call(*a, **kw):
                return original(*a, **{**kw, "enable_fp_fusion": True})

            stack.enter_context(patch.object(norm_module, "triton_call", fused_call))
        stages = jax.jit(forward)(xx, mm, params)
        full = jax.jit(
            lambda x, m, p: module.cueq_triangle_multiplication_forward(
                p, x, m, "outgoing"
            )
        )(xx, mm, params)
        if args.norm_statistics:
            norm_module = importlib.import_module(
                "cuequivariance_jax.triangle._layer_norm_transpose"
            )

            def statistics(x, p):
                if args.explicit_input_norm or x.shape == (1, 437, 437, 128):
                    from foldjax.models.boltz2.models.primitives import (
                        native_cueq_norm as selected_norm_module,
                    )

                    y, mean, rstd = selected_norm_module.native_cueq_norm(
                        x, p["norm_in"]["scale"], p["norm_in"]["bias"]
                    )
                    return (
                        y.reshape(1, -1, 128),
                        mean.reshape(1, -1),
                        rstd.reshape(1, -1),
                    )
                return norm_module.layer_norm_fwd_p.bind(
                    x.reshape(1, -1, 128),
                    p["norm_in"]["scale"],
                    p["norm_in"]["bias"],
                    eps=1e-5,
                    elementwise_affine=True,
                    layout=norm_module.Layout.BND_BND,
                    fallback=False,
                )

            norm_output, mean, rstd = jax.jit(statistics)(xx, params)
            stages.update(
                norm_stats_output=norm_output.reshape(xx.shape),
                norm_mean=mean,
                norm_rstd=rstd,
            )
    stages = {k: np.asarray(v, np.float32) for k, v in stages.items()}
    expected = select_arrays(native_stages, list(stages))
    comparisons = {k: bitwise_comparison(v, expected[k]) for k, v in stages.items()}
    full = np.asarray(full, np.float32)
    with (args.out / "stages.npz").open("xb") as stream:
        np.savez(stream, **stages)
    for path, digest in bindings.items():
        verify_bound_file(path, digest)
    save_new(
        args.out / "report.json",
        {
            "stage_comparisons": comparisons,
            "production_comparison": bitwise_comparison(full, target),
            "capture_bridge": bitwise_comparison(stages["output"], full),
            "stages_sha256": sha(args.out / "stages.npz"),
            "bindings": {str(p): d for p, d in bindings.items()},
            "not_model_parity_admission": True,
            "norm_fp_fusion_control": args.norm_fp_fusion,
            "norm_alignment_control": args.norm_alignment_control,
            "explicit_input_norm": args.explicit_input_norm,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-reference", type=Path)
    parser.add_argument("--norm-fp-fusion", action="store_true")
    parser.add_argument("--norm-statistics", action="store_true")
    parser.add_argument("--norm-alignment-control", action="store_true")
    parser.add_argument("--explicit-input-norm", action="store_true")
    args = parser.parse_args()
    if args.norm_fp_fusion and not args.candidate_reference:
        parser.error("norm fusion control requires the JAX arm")
    if args.norm_alignment_control and not args.candidate_reference:
        parser.error("alignment control requires the JAX arm")
    root = args.reference
    report_path = root / "report.json"
    bindings = {report_path: sha(report_path), Path(__file__): sha(Path(__file__))}
    report = json.loads(report_path.read_text())
    if report.get("passed") is not True or report.get("arm") != "native":
        raise ValueError("requires reproduced native MSA")
    for name in ("operands.npz", "native-weights.npz"):
        bindings[root / name] = report["artifacts"][name]
    for stage in ("opm", "tri_mul_out"):
        name = f"layers/00/{stage}"
        bindings[root / f"{name}.npz"] = report["stages"][name]["arrays_sha256"]
    for path, digest in bindings.items():
        verify_bound_file(path, digest)
    values = select_arrays(root / "operands.npz", ["input_z", "token_pad_mask"])
    opm = select_arrays(root / "layers/00/opm.npz", [""])[""]
    target = select_arrays(root / "layers/00/tri_mul_out.npz", [""])[""]
    prefix = "msa_module.layers.0.pairformer_layer.tri_mul_out."
    names = [
        f"{n}.{p}"
        for n, ps in (
            ("norm_in", ("weight", "bias")),
            ("norm_out", ("weight", "bias")),
            ("p_in", ("weight",)),
            ("g_in", ("weight",)),
            ("p_out", ("weight",)),
            ("g_out", ("weight",)),
        )
        for p in ps
    ]
    weights = select_arrays(root / "native-weights.npz", [prefix + n for n in names])
    if args.candidate_reference:
        mask = values["token_pad_mask"]
        candidate(
            args,
            values["input_z"] + opm,
            mask[:, :, None] * mask[:, None, :],
            weights,
            prefix,
            target,
            bindings,
        )
        return
    import torch

    if torch.__version__ != report["runtime"]["torch"]:
        raise ValueError("native Torch version differs")
    if torch.cuda.get_device_name() != report["runtime"]["device"]:
        raise ValueError("native device differs")
    module = importlib.import_module(
        "cuequivariance_ops_torch.triangle_multiplicative_update"
    )
    bindings[Path(module.__file__)] = sha(Path(module.__file__))
    args.out.mkdir(parents=True, exist_ok=False)
    x = torch.from_numpy((values["input_z"] + opm).copy()).cuda()
    mask = torch.from_numpy(values["token_pad_mask"].copy()).cuda()
    mask = mask[:, :, None] * mask[:, None, :]
    kwargs = {
        n.replace(".", "_"): torch.from_numpy(weights[prefix + n]).cuda() for n in names
    }
    stages = {}
    counts = {}

    def observe(name, original):
        def wrapped(*a, **kw):
            result = original(*a, **kw)
            index = counts.get(name, 0)
            counts[name] = index + 1
            stages[f"{name}/{index}"] = result.detach().float().cpu().numpy()
            if name == "layer_norm_transpose" and index == 1:
                stages["contraction"] = a[0].detach().float().cpu().numpy()
            return result

        return wrapped

    with (
        torch_policy(torch, precision="highest"),
        torch.inference_mode(),
        ExitStack() as stack,
    ):
        for name in (
            "layer_norm_transpose",
            "fused_sigmoid_gated_dual_gemm",
            "fused_sigmoid_gated_dual_gemm_dual_x",
        ):
            stack.enter_context(
                patch.object(module, name, observe(name, getattr(module, name)))
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module.triangle_multiplicative_update(
                x, "outgoing", mask=mask, **kwargs
            )
    actual = output.float().cpu().numpy()
    if args.norm_statistics:
        with torch.inference_mode():
            norm_output, mean, rstd = torch.ops.cuequivariance.layer_norm_transpose(
                x.reshape(1, -1, 128),
                kwargs["norm_in_weight"],
                kwargs["norm_in_bias"],
                1e-5,
                True,
                0,
            )
        stages.update(
            norm_stats_output=norm_output.reshape(x.shape).float().cpu().numpy(),
            norm_mean=mean.float().cpu().numpy(),
            norm_rstd=rstd.float().cpu().numpy(),
        )
        if not np.array_equal(
            stages["norm_stats_output"], stages["layer_norm_transpose/0"]
        ):
            raise ValueError("statistics capture does not reproduce native input norm")
        with torch.inference_mode():
            contracted = torch.from_numpy(stages["contraction"]).cuda().bfloat16()
            y, mean, rstd = torch.ops.cuequivariance.layer_norm_transpose(
                contracted.reshape(128, 1, -1),
                kwargs["norm_out_weight"],
                kwargs["norm_out_bias"],
                1e-5,
                True,
                3,
            )
        stages.update(
            output_norm_stats=y.reshape(x.shape).float().cpu().numpy(),
            output_norm_mean=mean.float().cpu().numpy(),
            output_norm_rstd=rstd.float().cpu().numpy(),
        )
        if not np.array_equal(
            stages["output_norm_stats"], stages["layer_norm_transpose/1"]
        ):
            raise ValueError("statistics capture does not reproduce native output norm")
    comparison = bitwise_comparison(actual, target)
    if counts != {
        "layer_norm_transpose": 2,
        "fused_sigmoid_gated_dual_gemm": 1,
        "fused_sigmoid_gated_dual_gemm_dual_x": 1,
    }:
        raise ValueError("native kernel route differs")
    stages.update(input=x.cpu().numpy(), mask=mask.cpu().numpy(), output=actual)
    with (args.out / "stages.npz").open("xb") as stream:
        np.savez(stream, **stages)
    for path, digest in bindings.items():
        verify_bound_file(path, digest)
    save_new(
        args.out / "report.json",
        {
            "comparison": comparison,
            "counts": counts,
            "bindings": {str(p): d for p, d in bindings.items()},
            "stages_sha256": sha(args.out / "stages.npz"),
            "not_model_parity_admission": True,
        },
    )
    if not comparison["bitwise_equal"]:
        raise ValueError("standalone triangle did not reproduce native capture")


if __name__ == "__main__":
    main()
