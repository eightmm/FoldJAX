"""Teacher-forced first MSA pair block, with native reproduction prerequisite.

Every operator receives the original native FP32 residual, not the previous
FoldJAX result. This isolates local error from propagated MSA/trunk error.
Neither this probe nor its normalization control is model parity admission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_probe import source_hashes, verify_bound_file
from bench.boltz_relpos_probe import arrays, comparison, torch_policy

STAGES = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end", "transition_z")
PREFIX = "msa_module.layers.0.pairformer_layer"


def reconstruct_inputs(initial, opm, updates, final):
    """Native eval dropout promotes each update before its FP32 residual add."""
    if set(updates) != set(STAGES):
        raise ValueError("all five native pair updates are required")
    if any(x.dtype != np.float32 for x in (initial, opm, final, *updates.values())):
        raise ValueError("captures must retain FP32 residuals and lossless updates")
    if any(x.shape != initial.shape for x in (opm, final, *updates.values())):
        raise ValueError("native pair shapes differ")
    result = initial + opm
    inputs = {}
    for name in STAGES:
        inputs[name] = result
        result = result + updates[name]
    if not np.array_equal(result, final):
        raise ValueError("reconstructed residual does not equal captured pair output")
    return inputs


def load_reference(root):
    report = json.loads((root / "report.json").read_text())
    if report.get("arm") != "native" or report.get("passed") is not True:
        raise ValueError("full native MSA reproduction must pass first")
    for name in ("operands.npz", "native-weights.npz"):
        verify_bound_file(root / name, report["artifacts"][name])
    operands = arrays(root / "operands.npz")
    captured = {}
    for name in (*STAGES, "opm", "pair_output"):
        key = f"layers/00/{name}"
        verify_bound_file(root / f"{key}.npz", report["stages"][key]["arrays_sha256"])
        captured[name] = arrays(root / f"{key}.npz")[""]
    inputs = reconstruct_inputs(
        operands["input_z"],
        captured["opm"],
        {key: captured[key] for key in STAGES},
        captured["pair_output"],
    )
    mask = operands["token_pad_mask"].astype(np.float32)
    mask = mask[:, :, None] * mask[:, None, :]
    weights = {
        k: v
        for k, v in arrays(root / "native-weights.npz").items()
        if k.startswith(PREFIX + ".")
    }
    return report, inputs, captured, mask, weights


def native(args):
    import torch

    report, inputs, targets, mask, weights = load_reference(args.reference)
    if source_hashes(args.upstream, "src") != report["native_source"]:
        raise ValueError("upstream source differs from original native capture")
    sys.path.insert(0, str(args.upstream / "src"))
    from boltz.model.layers.pairformer import PairformerNoSeqLayer

    module = PairformerNoSeqLayer(128, pairwise_head_width=32, pairwise_num_heads=4)
    module.load_state_dict(
        {
            k.removeprefix(PREFIX + "."): torch.from_numpy(v.copy())
            for k, v in weights.items()
        },
        strict=True,
    )
    module.eval().cuda()
    args.out.mkdir(parents=True, exist_ok=False)
    checks, norms = {}, {}
    with ExitStack() as hooks:
        for name in STAGES:

            def before(_module, operands, name=name):
                checks[name + "/input"] = comparison(
                    operands[0].float().cpu().numpy(), inputs[name]
                )

            def after(_module, _args, output, name=name):
                checks[name + "/output"] = comparison(
                    output.float().cpu().numpy(), targets[name]
                )

            child = getattr(module, name)
            hooks.callback(child.register_forward_pre_hook(before).remove)
            hooks.callback(child.register_forward_hook(after).remove)
            if name.startswith("tri_att") or name == "transition_z":

                def norm_hook(_module, operands, output, name=name):
                    path = args.out / f"{name}-norm.npz"
                    with path.open("xb") as stream:
                        np.savez(
                            stream,
                            input=operands[0].float().cpu().numpy(),
                            output=output.float().cpu().numpy(),
                        )
                    norms[name] = sha(path)

                norm = child.norm if name == "transition_z" else child.layer_norm
                hooks.callback(norm.register_forward_hook(norm_hook).remove)
        with (
            torch_policy(torch, reduction=True, precision="highest"),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            result = module(
                torch.from_numpy(inputs[STAGES[0]].copy()).cuda(),
                torch.from_numpy(mask.copy()).cuda(),
                chunk_size_tri_attn=128,
                use_kernels=True,
            )
            torch.cuda.synchronize()
    checks["pair_output"] = comparison(
        result.float().cpu().numpy(), targets["pair_output"]
    )
    passed = all(value["values_equal"] for value in checks.values())
    save_new(
        args.out / "report.json",
        {
            "arm": "native",
            "passed": passed,
            "checks": checks,
            "norms": norms,
            "reference_sha256": sha(args.reference / "report.json"),
            "wrapper_sha256": sha(Path(__file__)),
            "torch": torch.__version__,
            "source": report["native_source"],
        },
    )
    print(json.dumps({"passed": passed, "checks": checks}), flush=True)
    if not passed:
        raise ValueError("native pair replay differs; candidate interpretation blocked")


def foldjax(args):
    import jax
    import jax.numpy as jnp

    from bench.boltz_native_layer_norm import native_layer_norm
    from foldjax.models.boltz2.bridge.torch_mapping import (
        map_pairformer_no_seq_layer_state_dict,
    )
    from foldjax.models.boltz2.compile_policy import compiler_options
    from foldjax.models.boltz2.models.primitives import transition
    from foldjax.models.boltz2.models.triangle import triangle_attention
    from foldjax.models.boltz2.models.trunk_blocks import msa, trunk

    _, inputs, targets, mask, weights = load_reference(args.reference)
    native_report = json.loads((args.native_control / "report.json").read_text())
    if native_report.get("passed") is not True or native_report[
        "reference_sha256"
    ] != sha(args.reference / "report.json"):
        raise ValueError("matching native pair reproduction is required")
    params = trunk._cast_trunk_params(
        map_pairformer_no_seq_layer_state_dict(weights, PREFIX), jnp.bfloat16
    )
    source = Path(__file__).resolve().parents[1]
    before = source_hashes(source, "src")
    options = compiler_options("bfloat16")
    args.out.mkdir(parents=True, exist_ok=False)
    arms = {}

    def corrected_norm(x, scale, bias, eps):
        if x.dtype != jnp.float32:
            raise ValueError("teacher-forced native pair norms must receive FP32")
        return native_layer_norm(x, scale, bias, eps)[0]

    for arm in ("production", "native_norm"):
        values = {}
        with (
            ExitStack() as controls,
            jax.default_matmul_precision("highest"),
            patch.dict(
                os.environ, {"BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND": "cueq"}
            ),
        ):
            if arm == "native_norm":
                controls.enter_context(
                    patch.object(transition, "amp_layer_norm", corrected_norm)
                )
                controls.enter_context(
                    patch.object(triangle_attention, "amp_layer_norm", corrected_norm)
                )
            for name in STAGES:

                def run(p, x, mask):
                    if name.startswith("tri_mul"):
                        return msa.triangle_multiplication_forward(
                            p,
                            x,
                            mask,
                            "outgoing" if name.endswith("out") else "incoming",
                            chunk_size=128,
                        )
                    if name.startswith("tri_att"):
                        return msa.triangle_attention_forward(
                            p,
                            x,
                            mask,
                            starting=name.endswith("start"),
                            chunk_size=128,
                            triangle_backend="cueq",
                            matmul_precision="highest",
                        )
                    return transition.transition_forward(
                        p,
                        x,
                        row_chunk_size=128,
                        native_amp_norm=p["fc1"]["kernel"].dtype == jnp.bfloat16,
                    )

                output = jax.jit(run, compiler_options=options)(
                    params[name], jnp.asarray(inputs[name]), jnp.asarray(mask)
                )
                output.block_until_ready()
                values[name] = comparison(
                    np.asarray(output.astype(jnp.float32)), targets[name]
                )
                if name in native_report["norms"]:
                    path = args.native_control / f"{name}-norm.npz"
                    verify_bound_file(path, native_report["norms"][name])
                    captured = arrays(path)
                    norm_params = params[name][
                        "norm" if name == "transition_z" else "layer_norm"
                    ]
                    function = (
                        transition.amp_layer_norm
                        if name == "transition_z"
                        else triangle_attention.amp_layer_norm
                    )
                    normalized = jax.jit(
                        function, static_argnums=(3,), compiler_options=options
                    )(
                        jnp.asarray(captured["input"]),
                        norm_params["scale"],
                        norm_params["bias"],
                        1e-5,
                    )
                    values[name + "/norm"] = comparison(
                        np.asarray(normalized), captured["output"]
                    )
                print(
                    json.dumps({"arm": arm, "stage": name, "comparison": values[name]}),
                    flush=True,
                )
        arms[arm] = values
    if source_hashes(source, "src") != before:
        raise ValueError("source changed during probe")
    save_new(
        args.out / "report.json",
        {
            "arm": "foldjax",
            "capture_complete": True,
            "not_model_parity_admission": True,
            "scope": (
                "first MSA pair block; each operator reads native residual; "
                "mapped original weights"
            ),
            "arms": arms,
            "source": before,
            "compiler_options": options,
            "reference_sha256": sha(args.reference / "report.json"),
            "native_report_sha256": sha(args.native_control / "report.json"),
            "wrapper_sha256": sha(Path(__file__)),
            "diagnostic_norm_sha256": sha(
                Path(__file__).with_name("boltz_native_layer_norm.py")
            ),
            "runtime": {"jax": jax.__version__, "device": jax.devices()[0].device_kind},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--native-control", type=Path)
    args = parser.parse_args()
    if args.arm == "native" and args.upstream is None:
        parser.error("native arm needs --upstream")
    if args.arm == "foldjax" and args.native_control is None:
        parser.error("foldjax arm needs --native-control")
    (native if args.arm == "native" else foldjax)(args)


if __name__ == "__main__":
    main()
