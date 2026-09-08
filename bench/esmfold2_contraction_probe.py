"""First real triangle chunk: same operands, native reduction control and JAX.

Development operator evidence only. Requires an opt-in full-operand capture
from esmfold2_lm_encoder_probe; no synthetic inputs or reduced GEMM shapes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from bench.esmfold2_lm_encoder_candidate import (
    compare_boundaries,
    validate_capture,
    validate_incoming_boundary,
)
from bench.esmfold2_tape import _save_npz, _sha256

EQUATION = "bikd,bjkd->bijd"
EQUATIONS = {"outgoing": EQUATION, "incoming": "bkid,bkjd->bijd"}


def validate_operands(report, arrays):
    if not report.get("capture_first_contraction"):
        raise ValueError("native capture omitted complete first contraction")
    if report["precision_control"]:
        raise ValueError("reference must retain original native reduction policy")
    shape = report["boundaries"]["block.input"]["shape"]
    batch, rows, columns, channels = shape
    if rows != columns:
        raise ValueError("native pair grid must be square")
    expected = {
        "lhs": [batch, min(64, rows), rows, channels],
        "rhs": [batch, rows, rows, channels],
        "output": [batch, min(64, rows), rows, channels],
    }
    for name, target in expected.items():
        key = "first_contraction." + name
        value, schema = arrays[key], report["boundaries"][key]
        dtype = "torch.bfloat16" if name == "output" else "torch.float32"
        if (
            list(value.shape) != target
            or schema["shape"] != target
            or schema["stored_shape"] != target
            or schema["scope"] != "full"
            or schema["original_dtype"] != dtype
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("invalid complete native contraction operand/schema")
        strides, offset = schema["native_stride"], schema["native_storage_offset"]
        if (
            len(strides) != len(target)
            or any(type(v) is not int or v <= 0 for v in strides)
            or type(offset) is not int
            or offset < 0
            or offset + sum((n - 1) * s for n, s in zip(target, strides))
            >= 4 * int(np.prod(shape))
        ):
            raise ValueError("invalid or unexpectedly large native operand strides")
    return {name: arrays["first_contraction." + name] for name in expected}


def select_contraction(report, arrays, direction):
    if direction == "outgoing":
        return validate_operands(report, arrays), {
            name: report["boundaries"]["first_contraction." + name]
            for name in ("lhs", "rhs", "output")
        }
    if direction != "incoming":
        raise ValueError("unknown contraction direction")
    validate_incoming_boundary(report, arrays)
    if report["precision_control"]:
        raise ValueError("reference must retain original native reduction policy")
    shape = report["boundaries"]["block.input"]["shape"]
    if len(shape) != 4 or shape[1] != shape[2]:
        raise ValueError("native pair grid must be square")
    size = min(64, shape[1])
    operands = {
        "lhs": arrays["incoming.left"][:, :, :size],
        "rhs": arrays["incoming.right"],
        "output": arrays["incoming.output"][:, :size],
    }
    schemas = {}
    for name, leaf in (("lhs", "left"), ("rhs", "right"), ("output", "output")):
        schema = report["boundaries"]["incoming." + leaf]
        strides, offset = schema["native_stride"], schema["native_storage_offset"]
        target = operands[name].shape
        if (
            len(strides) != 4
            or any(type(v) is not int or v <= 0 for v in strides)
            or type(offset) is not int
            or offset < 0
            or offset + sum((n - 1) * s for n, s in zip(target, strides))
            >= 4 * int(np.prod(shape))
        ):
            raise ValueError("invalid native incoming operand layout")
        # Slicing the native output-i dimension retains the original strides
        # and offset, including the right half of the interleaved routed pair.
        schemas[name] = {
            **schema,
            "shape": list(target),
            "stored_shape": list(target),
            "source_view": "first_output_i_chunk_of_complete_incoming_capture",
        }
    return operands, schemas


def outgoing_layout_operands(operands, transposed):
    left, right = operands["lhs"], operands["rhs"]
    if left.ndim != 4 or right.ndim != 4 or left.shape[0] != 1 or right.shape[0] != 1:
        raise ValueError("layout control requires batch-one rank-four operands")
    if left.shape[2:] != right.shape[2:]:
        raise ValueError("layout control requires equal contraction/channel dimensions")
    return (
        np.ascontiguousarray(left[0].transpose(2, 1, 0)),
        np.ascontiguousarray(
            right[0].transpose((2, 1, 0) if transposed else (2, 0, 1))
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("native", "candidate"))
    parser.add_argument("--direction", choices=tuple(EQUATIONS), default="outgoing")
    parser.add_argument("--layout-control", action="store_true")
    for name in ("native", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.layout_control and (
        args.backend != "candidate" or args.direction != "outgoing"
    ):
        parser.error("layout control requires the candidate outgoing contraction")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        args.native / "report.json",
        args.native / "native.npz",
        args.weights / "model.safetensors",
        Path(__file__),
        Path(inspect.getfile(compare_boundaries)),
        Path(inspect.getfile(_sha256)),
    ]
    before = {str(p.resolve()): _sha256(p) for p in paths}
    report, arrays = validate_capture(args.native, args.weights)
    operands, schemas = select_contraction(report, arrays, args.direction)
    equation = EQUATIONS[args.direction]
    del arrays
    outputs, policies, dtypes = {}, {}, {}
    if args.backend == "native":
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("native probe requires one CUDA GPU")
        if (
            str(torch.__version__) != report["torch"]
            or torch.version.git_version != report["torch_git"]
        ):
            raise ValueError("native Torch runtime differs from captured block")
        torch.set_float32_matmul_precision(report["matmul"])
        torch.backends.cuda.matmul.allow_tf32 = report["allow_tf32"]
        prior = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if prior != report["allow_bf16_reduced_precision_reduction"]:
            raise ValueError("native inherited reduction default differs from capture")

        def restore(name):
            array = operands[name]
            schema = schemas[name]
            stride, offset = schema["native_stride"], schema["native_storage_offset"]
            size = offset + 1 + sum((n - 1) * s for n, s in zip(array.shape, stride))
            storage = torch.empty(size, dtype=torch.float32, device="cuda")
            view = torch.as_strided(storage, array.shape, stride, offset)
            view.copy_(torch.from_numpy(array).cuda())
            return view

        lhs, rhs = (restore(n) for n in ("lhs", "rhs"))
        try:
            for name, reduction in (
                ("native_default", prior),
                ("native_reduction_false", False),
            ):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                    reduction
                )
                policies[name] = {"allow_bf16_reduced_precision_reduction": reduction}
                for repeat in range(2):
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        value = torch.einsum(equation, lhs, rhs)
                    key = f"{name}.{repeat}"
                    dtypes[key] = {"dtype": str(value.dtype)}
                    outputs[key] = value.float().cpu().numpy().copy()
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prior
        versions = {
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        }
    else:
        import jax
        import jax.numpy as jnp

        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise RuntimeError("candidate probe requires one GPU")
        jax.config.update("jax_default_matmul_precision", "highest")
        lhs, rhs = (jnp.asarray(operands[n]) for n in ("lhs", "rhs"))

        def contract(left, right):
            return jnp.einsum(
                equation,
                left.astype(jnp.bfloat16),
                right.astype(jnp.bfloat16),
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)

        lowered = jax.jit(contract).lower(lhs, rhs)
        compiled = lowered.compile()
        for repeat in range(2):
            value = compiled(lhs, rhs)
            value.block_until_ready()
            key = f"candidate.{repeat}"
            dtypes[key] = {"dtype": str(value.dtype)}
            outputs[key] = np.asarray(value.astype(jnp.float32))
        (args.output / "compiled.hlo.txt").write_text(compiled.as_text())
        policies["candidate"] = {
            "matmul": "highest",
            "reduction": "FP32 preferred accumulation; BF16 output",
        }
        if args.layout_control:
            # Same logical operands, two explicit channel-major storage orders.
            # Disable the previously isolated GEMM/padding rewrites in both arms.
            options = {
                "xla_gpu_enable_triton_gemm": False,
                "xla_disable_hlo_passes": "cublas-pad-for-gemms",
            }
            for name, transposed in (("rhs_jk", False), ("rhs_kj", True)):
                left, right = outgoing_layout_operands(operands, transposed)
                left, right = (
                    jnp.asarray(v).astype(jnp.bfloat16) for v in (left, right)
                )
                layout_equation = "dkj,dki->dji" if transposed else "djk,dki->dji"

                def layout_contract(a, b):
                    return jnp.einsum(
                        layout_equation, b, a, preferred_element_type=jnp.float32
                    ).astype(jnp.bfloat16)

                executable = (
                    jax.jit(layout_contract, compiler_options=options)
                    .lower(left, right)
                    .compile()
                )
                (args.output / f"{name}.hlo.txt").write_text(executable.as_text())
                for repeat in range(2):
                    value = executable(left, right)
                    value.block_until_ready()
                    key = f"{name}.{repeat}"
                    dtypes[key] = {"dtype": str(value.dtype)}
                    outputs[key] = np.asarray(value.astype(jnp.float32)).transpose(
                        2, 1, 0
                    )[None]
                policies[name] = {
                    "compiler_options": options,
                    "equation": layout_equation,
                    "hlo_sha256": _sha256(args.output / f"{name}.hlo.txt"),
                    "runtime_default_change": False,
                }
        versions = {"jax": jax.__version__, "device": str(jax.devices()[0])}
    reference = {key: operands["output"] for key in outputs}
    comparison = compare_boundaries(
        reference,
        outputs,
        {key: {"original_dtype": "torch.bfloat16"} for key in outputs},
        dtypes,
    )
    repeats = {
        name: compare_boundaries(
            {"output": outputs[name + ".0"]},
            {"output": outputs[name + ".1"]},
            {"output": {"original_dtype": dtypes[name + ".0"]["dtype"]}},
            {"output": dtypes[name + ".1"]},
        )
        for name in policies
    }
    _save_npz(args.output / "outputs.npz", outputs)
    validate_capture(args.native, args.weights)
    if before != {str(p.resolve()): _sha256(p) for p in paths}:
        raise ValueError("bound capture/checkpoint/runner changed during replay")
    result = {
        "scope": f"same_complete_native_first_{args.direction}_chunk_operator_only",
        "model_admission": None,
        "equation": equation,
        "operand_schemas": schemas,
        "bindings": before,
        "policies": policies,
        "versions": versions,
        "original_dtypes": dtypes,
        "comparison_to_original_block": comparison,
        "same_process_repeats": repeats,
        "outputs_sha256": _sha256(args.output / "outputs.npz"),
        "hlo_sha256": _sha256(args.output / "compiled.hlo.txt")
        if args.backend == "candidate"
        else None,
        "python_executable": sys.executable,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
