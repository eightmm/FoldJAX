"""First-layer PWA diagnostic on matched native MSA rows, not model parity.

The native decomposition must exactly reproduce its actual module. The row
slice is also compared to the full native capture; it is not assumed bitwise
inert. Framework stages are returned from JIT, so this is instrumented work.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_probe import source_hashes, verify_bound_file
from bench.boltz_relpos_probe import arrays, bf16_round, comparison, torch_policy


def load_reference(root, rows):
    report = json.loads((root / "report.json").read_text())
    if report.get("arm") != "native" or report.get("passed") is not True:
        raise ValueError("native MSA reproduction must have passed")
    for name in ("operands.npz", "native-weights.npz"):
        verify_bound_file(root / name, report["artifacts"][name])
    for stage in ("input_m", "pwa"):
        name = f"layers/00/{stage}"
        identity = report["stages"][name]
        verify_bound_file(root / f"{name}.npz", identity["arrays_sha256"])
        verify_bound_file(root / f"{name}.tree.json", identity["tree_sha256"])
    with np.load(root / "layers/00/input_m.npz", allow_pickle=False) as archive:
        m = archive[""]
    if not 0 < rows <= m.shape[1]:
        raise ValueError("row slice must be nonempty and within the captured MSA")
    with np.load(root / "operands.npz", allow_pickle=False) as archive:
        z, mask = archive["input_z"], archive["token_pad_mask"]
    with np.load(root / "layers/00/pwa.npz", allow_pickle=False) as archive:
        expected = archive[""][:, :rows]
    prefix = "msa_module.layers.0.pair_weighted_averaging."
    weights = {
        k.removeprefix(prefix): v
        for k, v in arrays(root / "native-weights.npz").items()
        if k.startswith(prefix)
    }
    if len(weights) != 8:
        raise ValueError("unreviewed PWA weight schema")
    return report, {"m": m[:, :rows], "z": z, "mask": mask}, weights, expected


def native(args):
    import torch

    root, upstream = args.reference.resolve(), args.upstream.resolve()
    report, inputs, weights, expected = load_reference(root, args.rows)
    if source_hashes(upstream, "src") != report["native_source"]:
        raise ValueError("native source differs from the reproduced MSA")
    sys.path.insert(0, str(upstream / "src"))
    from boltz.model.layers.pair_averaging import PairWeightedAveraging

    if Path(inspect.getfile(PairWeightedAveraging)).resolve() != (
        upstream / "src/boltz/model/layers/pair_averaging.py"
    ):
        raise ValueError("wrong native source import")
    heads = weights["proj_z.weight"].shape[0]
    hidden = weights["proj_m.weight"].shape[0] // heads
    model = PairWeightedAveraging(64, 128, hidden, heads).eval()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})
    model.cuda()
    m = torch.from_numpy(inputs["m"].copy()).cuda().bfloat16()
    z = torch.from_numpy(inputs["z"].copy()).cuda()
    mask = torch.from_numpy(inputs["mask"].copy()).cuda().float()
    mask = mask[:, :, None] * mask[:, None, :]
    stages = {}

    def record(name, value):
        stages[name] = value.detach().float().cpu().numpy()
        return value

    chunked = z.shape[1] > 384
    with torch_policy(torch, precision="highest"), torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            record("actual", model(m, z, mask, chunk_heads=chunked))
            m, z = record("norm_m", model.norm_m(m)), record("norm_z", model.norm_z(z))
            output = None
            group_size = 1 if chunked else heads
            for group, first in enumerate(range(0, heads, group_size)):
                start, stop = first * hidden, (first + group_size) * hidden
                label = f"head{group}"
                v = record(f"{label}/v", m @ model.proj_m.weight[start:stop].T)
                v = v.reshape(*v.shape[:3], group_size, hidden).permute(0, 3, 1, 2, 4)
                b = record(
                    f"{label}/logits",
                    z @ model.proj_z.weight[first : first + group_size].T,
                )
                b = b.permute(0, 3, 1, 2) + (1 - mask[:, None]) * -1e6
                w = record(f"{label}/weights", b.softmax(-1))
                g = record(
                    f"{label}/gate", (m @ model.proj_g.weight[start:stop].T).sigmoid()
                )
                o = torch.einsum("bhij,bhsjd->bhsid", w, v)
                o = record(
                    f"{label}/averaged",
                    o.permute(0, 2, 3, 1, 4).reshape(*m.shape[:3], stop - start),
                )
                partial = record(
                    f"{label}/partial", (g * o) @ model.proj_o.weight[:, start:stop].T
                )
                output = partial if output is None else output + partial
                record(f"{label}/cumulative", output)
        torch.cuda.synchronize()
    record("decomposed", output)
    reproduction = comparison(stages["decomposed"], stages["actual"])
    if not reproduction["values_equal"]:
        raise ValueError(f"native decomposition changed PWA: {reproduction}")
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in {"inputs": inputs, "weights": weights, "stages": stages}.items():
        with (args.out / f"{name}.npz").open("xb") as stream:
            np.savez(stream, **value)
    save_new(
        args.out / "report.json",
        {
            "arm": "native",
            "passed": True,
            "rows": args.rows,
            "native_msa_report_sha256": sha(root / "report.json"),
            "native_decomposition": reproduction,
            "row_slice_vs_full": comparison(stages["actual"], expected),
            "native_source": report["native_source"],
            "wrapper_sha256": sha(Path(__file__)),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "artifacts": {
                f"{name}.npz": sha(args.out / f"{name}.npz")
                for name in ("inputs", "weights", "stages")
            },
        },
    )
    print(
        json.dumps(
            {
                "decomposition": reproduction,
                "row_slice_vs_full": comparison(stages["actual"], expected),
            }
        )
    )


def foldjax(args):
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.compile_policy import compiler_options
    from foldjax.models.boltz2.models.primitives._common import (
        layer_norm,
        linear,
        sigmoid,
    )
    from foldjax.models.boltz2.models.trunk_blocks.msa import (
        pair_weighted_averaging_forward,
    )

    root = args.reference.resolve()
    report = json.loads((root / "report.json").read_text())
    if report.get("arm") != "native" or report.get("passed") is not True:
        raise ValueError("native decomposition has not passed")
    for name, digest in report["artifacts"].items():
        verify_bound_file(root / name, digest)
    inputs, weights, expected = (
        arrays(root / f"{name}.npz") for name in ("inputs", "weights", "stages")
    )
    params = {}
    for name, value in weights.items():
        module, leaf = name.split(".")
        key = (
            "bias"
            if leaf == "bias"
            else "scale"
            if module.startswith("norm")
            else "kernel"
        )
        params.setdefault(module, {})[key] = jnp.asarray(
            value.T if key == "kernel" else value,
            dtype=jnp.bfloat16 if key == "kernel" else jnp.float32,
        )
    operands = {
        k: jnp.asarray(v, dtype=jnp.bfloat16 if k == "m" else jnp.float32)
        for k, v in inputs.items()
    }
    norm_control = getattr(args, "native_norm_control", False)
    diagnostic_norm_sha256 = None
    if norm_control:
        from bench.boltz_native_layer_norm import native_layer_norm

        diagnostic_norm_sha256 = sha(Path(inspect.getfile(native_layer_norm)))

        def layer_norm(x, scale, bias, eps):
            return native_layer_norm(x.astype(jnp.float32), scale, bias, eps)[0]

    def run(params, inputs):
        m, z, mask = (inputs[k] for k in ("m", "z", "mask"))
        mask = mask[:, :, None] * mask[:, None, :]
        stages = {
            "actual": pair_weighted_averaging_forward(
                params, m, z, mask, row_chunk_size=0
            )
        }
        m = stages["norm_m"] = layer_norm(m, **params["norm_m"], eps=1e-5)
        z = stages["norm_z"] = layer_norm(z, **params["norm_z"], eps=1e-5)
        heads = params["proj_z"]["kernel"].shape[-1]
        hidden = params["proj_m"]["kernel"].shape[-1] // heads
        group_size = 1 if z.shape[1] > 384 else heads
        output = None
        for group, first in enumerate(range(0, heads, group_size)):
            start, stop = first * hidden, (first + group_size) * hidden
            label = f"head{group}"
            v = stages[f"{label}/v"] = linear(
                m, params["proj_m"]["kernel"][:, start:stop]
            )
            v = v.reshape(*v.shape[:3], group_size, hidden).transpose(0, 3, 1, 2, 4)
            b = stages[f"{label}/logits"] = linear(
                z, params["proj_z"]["kernel"][:, first : first + group_size]
            )
            b = b.transpose(0, 3, 1, 2).astype(jnp.float32) + (1 - mask[:, None]) * -1e6
            w = stages[f"{label}/weights"] = jax.nn.softmax(b, -1)
            g = stages[f"{label}/gate"] = sigmoid(
                linear(m, params["proj_g"]["kernel"][:, start:stop])
            )
            o = jnp.einsum("bhij,bhsjd->bhsid", w.astype(v.dtype), v)
            o = stages[f"{label}/averaged"] = o.transpose(0, 2, 3, 1, 4).reshape(
                *m.shape[:3], stop - start
            )
            partial = stages[f"{label}/partial"] = linear(
                g * o, params["proj_o"]["kernel"][start:stop]
            )
            output = partial if output is None else output + partial
            stages[f"{label}/cumulative"] = output
        stages["decomposed"] = output
        return stages

    with (
        jax.default_matmul_precision("highest"),
        patch("foldjax.models.boltz2.models.trunk_blocks.msa._layer_norm", layer_norm),
    ):
        stages = jax.jit(run, compiler_options=compiler_options("bfloat16"))(
            params, operands
        )
        jax.block_until_ready(stages)
    actual = {k: np.asarray(v.astype(jnp.float32)) for k, v in stages.items()}
    # A one-column GEMM may lower as a BF16 reduction rather than a tensor-core
    # GEMM. Hold even the normalized native operands fixed to separate this from
    # LayerNorm's FP32 reduction-order differences.
    norm_native = jnp.asarray(expected["norm_z"], jnp.bfloat16)
    kernel = params["proj_z"]["kernel"]

    def project(x, kernel):
        low, wide = [], []
        for head in range(kernel.shape[-1]):
            weight = kernel[:, head : head + 1]
            low.append(jnp.matmul(x, weight))
            wide.append(
                jnp.matmul(x, weight, preferred_element_type=jnp.float32).astype(
                    x.dtype
                )
            )
        return {
            "one_column_default": jnp.concatenate(low, -1),
            "one_column_fp32_accumulator": jnp.concatenate(wide, -1),
            "all_heads_default": jnp.matmul(x, kernel),
        }

    with jax.default_matmul_precision("highest"):
        controls = jax.jit(project, compiler_options=compiler_options("bfloat16"))(
            norm_native, kernel
        )
        jax.block_until_ready(controls)
    target = np.concatenate(
        [expected[f"head{h}/logits"] for h in range(kernel.shape[-1])], -1
    )
    exact = bf16_round(expected["norm_z"]).astype(np.float64) @ bf16_round(
        weights["proj_z.weight"].T
    ).astype(np.float64)
    projection_controls = {
        k: comparison(np.asarray(v.astype(jnp.float32)), target)
        for k, v in controls.items()
    }
    projection_controls["fp64_oracle"] = comparison(bf16_round(exact), target)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "stages.npz").open("xb") as stream:
        np.savez(stream, **actual)
    comparisons = {k: comparison(actual[k], expected[k]) for k in actual}
    save_new(
        args.out / "report.json",
        {
            "arm": "foldjax",
            "capture_complete": True,
            "not_model_parity_admission": True,
            "native_norm_control": norm_control,
            "diagnostic_norm_sha256": diagnostic_norm_sha256,
            "row_slice_scope": report["row_slice_vs_full"],
            "scope": (
                "LayerNorm same-operand diagnostic; sliced PWA contraction may "
                "differ from original native full-row shape and is not admission"
            ),
            "stage_comparisons": comparisons,
            "native_normalized_projection_controls": projection_controls,
            "decomposition_vs_production": comparison(
                actual["decomposed"], actual["actual"]
            ),
            "reference_sha256": sha(root / "report.json"),
            "wrapper_sha256": sha(Path(__file__)),
            "compiler_options": compiler_options("bfloat16"),
            "jax": jax.__version__,
            "stages_sha256": sha(args.out / "stages.npz"),
            "source": source_hashes(
                Path(inspect.getfile(pair_weighted_averaging_forward))
                .resolve()
                .parents[6],
                "src",
            ),
        },
    )
    print(json.dumps(comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--native-norm-control", action="store_true")
    args = parser.parse_args()
    if args.arm == "native" and args.upstream is None:
        parser.error("native arm requires --upstream")
    if args.arm == "native" and args.native_norm_control:
        parser.error("--native-norm-control applies to FoldJAX only")
    (native if args.arm == "native" else foldjax)(args)


if __name__ == "__main__":
    main()
