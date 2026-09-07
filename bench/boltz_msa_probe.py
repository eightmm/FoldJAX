"""Same-input Boltz MSA counterfactual with native reproduction prerequisite.

Both arms observe the real module, not a reimplementation of its layer loop.
The full MSA and first-recycle pair input are publisher captures. This is an
operator diagnostic, not independent preprocessing or end-to-end admission.
Importing this module needs only NumPy; each arm imports its own framework.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections import Counter
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import flatten, sha
from bench.boltz_closure_capture import save_new, save_tree
from bench.boltz_relpos_probe import arrays, comparison, torch_policy

FEATURES = (
    "msa",
    "has_deletion",
    "deletion_value",
    "msa_paired",
    "msa_mask",
    "token_pad_mask",
)
STAGES = (
    "input_m",
    "pwa",
    "msa_transition",
    "opm",
    "tri_mul_out",
    "tri_mul_in",
    "tri_att_start",
    "tri_att_end",
    "transition_z",
    "pair_output",
    "layer_output",
)
MSA_ARGS = {
    "msa_s",
    "msa_blocks",
    "msa_dropout",
    "z_dropout",
    "pairwise_head_width",
    "pairwise_num_heads",
    "activation_checkpointing",
    "use_paired_feature",
    "subsample_msa",
    "num_subsampled_msa",
    "miniformer_blocks",
    "use_trifast",
}


def full_row_control(function):
    """Keep native full batch rows without changing hidden/head reductions."""

    @wraps(function)
    def run(*args, **kwargs):
        return function(*args, **{**kwargs, "row_chunk_size": 0})

    return run


def native_dense_msa_embedding(
    params, emb, msa, has_deletion, deletion_value, msa_paired, *, num_tokens
):
    """Diagnostic native-width GEMM instead of sparse gather plus scalar dot."""
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives._common import linear

    dtype = params["msa_proj"]["kernel"].dtype
    fields = [jax.nn.one_hot(msa, num_tokens, dtype=dtype)]
    fields += [x[..., None].astype(dtype) for x in (has_deletion, deletion_value)]
    if params["msa_proj"]["kernel"].shape[0] == num_tokens + 3:
        fields.append(msa_paired[..., None].astype(dtype))
    m = linear(jnp.concatenate(fields, axis=-1), params["msa_proj"]["kernel"])
    return m + linear(emb, params["s_proj"]["kernel"])[:, None]


def profile_from_hparams(hparams):
    args = dict(hparams["msa_args"])
    if set(args) - MSA_ARGS:
        raise ValueError(f"unreviewed MSA checkpoint options: {set(args) - MSA_ARGS}")
    profile = {
        "token_s": int(hparams["token_s"]),
        "token_z": int(hparams["token_z"]),
        **args,
    }
    if (profile["token_s"], profile["token_z"], args["msa_s"], args["msa_blocks"]) != (
        384,
        128,
        64,
        4,
    ):
        raise ValueError("probe requires the released Boltz-2 MSA dimensions")
    if not args.get("use_paired_feature", True) or args.get("subsample_msa", False):
        raise ValueError("probe requires paired features and full MSA")
    return profile


def validate_operands(values, profile):
    if set(values) != set(FEATURES) | {"input_z", "emb"}:
        raise ValueError("unknown or missing MSA operand")
    msa = values["msa"]
    if msa.ndim != 3 or msa.shape[0] != 1 or min(msa.shape) < 1:
        raise ValueError("MSA must have singleton batch and nonempty row/token axes")
    b, _, n = msa.shape
    expected = {key: msa.shape for key in FEATURES if key != "token_pad_mask"}
    expected.update(
        token_pad_mask=(b, n),
        input_z=(b, n, n, profile["token_z"]),
        emb=(b, n, profile["token_s"]),
    )
    for key, value in values.items():
        if value.shape != expected[key] or value.dtype.kind not in "bifu":
            raise ValueError(f"invalid MSA operand shape/dtype: {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite MSA operand: {key}")
    if msa.dtype.kind not in "iu" or msa.min() < 0 or msa.max() >= 33:
        raise ValueError("MSA token IDs must be integers in [0,33)")
    for key in ("input_z", "emb"):
        if values[key].dtype != np.float32:
            raise ValueError(f"native {key} must retain FP32 residual/input values")
    for key in ("has_deletion", "msa_paired", "msa_mask", "token_pad_mask"):
        if not np.isin(values[key], [0, 1]).all():
            raise ValueError(f"nonbinary native MSA flag: {key}")


def validate_counts(counts, layers=4):
    expected = {stage: layers for stage in STAGES}
    if dict(counts) != expected:
        raise ValueError(
            f"incomplete/extra MSA stage calls: {dict(counts)} != {expected}"
        )


def verify_bound_file(path, expected):
    if sha(path) != expected:
        raise ValueError(f"reference identity changed: {path}")


def source_hashes(root, directory):
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted((root / directory).rglob("*.py"))
    }


def save_arrays(path, values):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


class Observer:
    def __init__(self, out, backend):
        self.out, self.backend = out, backend
        self.counts = Counter()
        self.artifacts = {}

    def record(self, stage, value, index=None):
        if stage not in STAGES:
            raise ValueError(f"unknown MSA observation {stage}")
        current = self.counts[stage]
        if index is not None and index != current:
            raise ValueError(f"out-of-order MSA layer: {stage}/{index} != {current}")
        if current >= 4:
            raise ValueError(f"extra MSA stage call: {stage}")
        self.counts[stage] += 1
        name = f"layers/{current:02d}/{stage}"
        if self.backend == "native":
            identity = save_tree(self.out, name, value)
        else:
            from bench.protenix_foldjax_capture import save_jax_boundary

            path = self.out / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.with_suffix(".tree.json").exists():
                raise FileExistsError(path)
            save_jax_boundary(path, value)
            identity = {
                "arrays_sha256": sha(path),
                "tree_sha256": sha(path.with_suffix(".tree.json")),
            }
        self.artifacts[name] = identity

    def native_hooks(self, model):
        stack = ExitStack()
        names = {
            "pair_weighted_averaging": "pwa",
            "msa_transition": "msa_transition",
            "outer_product_mean": "opm",
            "pairformer_layer": "pair_output",
        }
        names.update(
            {
                f"pairformer_layer.{name}": name
                for name in (
                    "tri_mul_out",
                    "tri_mul_in",
                    "tri_att_start",
                    "tri_att_end",
                    "transition_z",
                )
            }
        )
        for index, layer in enumerate(model.layers):

            def before(_module, args, index=index):
                self.record("input_m", args[1], index)

            def after(_module, _args, output, index=index):
                self.record("layer_output", {"z": output[0], "m": output[1]}, index)

            stack.callback(layer.register_forward_pre_hook(before).remove)
            stack.callback(layer.register_forward_hook(after).remove)
            for path, stage in names.items():

                def hook(_module, _args, output, stage=stage, index=index):
                    self.record(stage, output, index)

                stack.callback(
                    layer.get_submodule(path).register_forward_hook(hook).remove
                )
        return stack

    def jax_hooks(self, module):
        import jax

        stack = ExitStack()
        nesting = Counter()

        def wrap(original, stage_fn):
            def observed(*args, **kwargs):
                stage = stage_fn(args, kwargs)
                outer = nesting[stage] == 0
                nesting[stage] += 1
                try:
                    if stage == "layer_output" and outer:
                        jax.debug.callback(
                            lambda x: self.record("input_m", x), args[2], ordered=True
                        )
                    result = original(*args, **kwargs)
                finally:
                    nesting[stage] -= 1
                if outer:
                    value = (
                        {"z": result[0], "m": result[1]}
                        if stage == "layer_output"
                        else result
                    )
                    jax.debug.callback(
                        lambda x: self.record(stage, x), value, ordered=True
                    )
                return result

            return observed

        targets = {
            "msa_layer_forward": lambda a, k: "layer_output",
            "pair_weighted_averaging_forward": lambda a, k: "pwa",
            "outer_product_mean_forward": lambda a, k: "opm",
            "pairformer_no_seq_layer_forward": lambda a, k: "pair_output",
            "transition_forward": lambda a, k: (
                "msa_transition" if a[1].shape[-1] == 64 else "transition_z"
            ),
            "triangle_multiplication_forward": lambda a, k: (
                "tri_mul_out" if a[3] == "outgoing" else "tri_mul_in"
            ),
            "triangle_attention_forward": lambda a, k: (
                "tri_att_start" if k["starting"] else "tri_att_end"
            ),
        }
        for name, stage_fn in targets.items():
            stack.enter_context(
                patch.object(module, name, wrap(getattr(module, name), stage_fn))
            )
        return stack


def native(args):
    import torch

    root, upstream = args.reference.resolve(), args.upstream.resolve()
    complete = json.loads((root / "capture-complete.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    effective = json.loads((root / "effective-model-settings.json").read_text())
    tape = json.loads((root / "tape.json").read_text())
    if complete.get("passed") is not True or tape["subsample_msa"]:
        raise ValueError("reference must be a completed full-MSA capture")
    if (
        not effective["use_kernels"]
        or effective["float32_matmul_precision"] != "highest"
        or not effective["cuda_autocast_enabled"]
        or effective["cuda_autocast_dtype"] != "torch.bfloat16"
        or effective["training"]
    ):
        raise ValueError("reference is not eval BF16/cuEq/highest policy")
    verify_bound_file(args.checkpoint, provenance["checkpoint_sha256"])
    before = source_hashes(upstream, "src")
    if before != provenance["upstream_python_source"]:
        raise ValueError("native Python source differs from completed capture")
    for name in (
        "trunk-boundaries/cycle-00/msa_module",
        "trunk-boundaries/input_embedder",
    ):
        binding = complete["artifacts"][name]
        verify_bound_file(root / f"{name}.npz", binding["arrays_sha256"])
        verify_bound_file(root / f"{name}.tree.json", binding["tree_sha256"])
    sys.path.insert(0, str(upstream / "src"))
    from boltz.model.modules.trunkv2 import MSAModule

    if (
        Path(inspect.getfile(MSAModule)).resolve()
        != upstream / "src/boltz/model/modules/trunkv2.py"
    ):
        raise ValueError("imported another native MSA module")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    profile = profile_from_hparams(checkpoint["hyper_parameters"])
    state = {
        k: v.detach().clone()
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("msa_module.")
    }
    del checkpoint
    model = MSAModule(**profile).eval()
    model.load_state_dict(
        {k.removeprefix("msa_module."): v for k, v in state.items()}, strict=True
    )
    with np.load(root / "features.npz", allow_pickle=False) as f:
        values = {key: f[key] for key in FEATURES}
    baseline = arrays(root / "trunk-boundaries/cycle-00/msa_module.npz")
    values.update(
        input_z=baseline["input_z"],
        emb=arrays(root / "trunk-boundaries/input_embedder.npz")[""],
    )
    validate_operands(values, profile)
    args.out.mkdir(parents=True, exist_ok=False)
    save_arrays(args.out / "operands.npz", values)
    save_arrays(
        args.out / "native-weights.npz", {k: v.numpy() for k, v in state.items()}
    )
    observer = Observer(args.out, "native")
    model = model.cuda()
    operands = {k: torch.from_numpy(v.copy()).cuda() for k, v in values.items()}
    with (
        observer.native_hooks(model),
        torch_policy(torch, precision="highest"),
        torch.inference_mode(),
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(
                operands["input_z"],
                operands["emb"],
                {k: operands[k] for k in FEATURES},
                use_kernels=True,
            )
        torch.cuda.synchronize()
        output = result.detach().float().cpu().numpy()
        runtime = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "cuda_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
    validate_counts(observer.counts)
    reproduction = comparison(output, baseline["delta_z"])
    save_arrays(args.out / "output.npz", {"z": output})
    if source_hashes(upstream, "src") != before:
        raise ValueError("native source changed during probe")
    report = {
        "arm": "native",
        "passed": reproduction["values_equal"],
        "reproduction": reproduction,
        "profile": profile,
        "runtime": runtime,
        "input_reference": str(root),
        "checkpoint_sha256": sha(args.checkpoint),
        "features_sha256": sha(root / "features.npz"),
        "capture_complete_sha256": sha(root / "capture-complete.json"),
        "effective_settings_sha256": sha(root / "effective-model-settings.json"),
        "native_source": before,
        "wrapper_sha256": sha(Path(__file__)),
        "counts": dict(observer.counts),
        "stages": observer.artifacts,
        "artifacts": {
            name: sha(args.out / name)
            for name in ("operands.npz", "native-weights.npz", "output.npz")
        },
        "scope": "same-input MSA only; full stage callbacks; no performance claim",
        "npz_compression": "deflate" if args.compressed else "stored",
    }
    save_new(args.out / "report.json", report)
    if not report["passed"]:
        raise ValueError(
            "standalone native MSA did not exactly reproduce original cycle 0; "
            "interpretation blocked"
        )
    print(json.dumps({"passed": True, "reproduction": reproduction}))


def foldjax(args):
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.boltz2.bridge.native import unflatten_pytree
    from foldjax.models.boltz2.bridge.torch_mapping import map_msa_module_state_dict
    from foldjax.models.boltz2.compile_policy import compiler_options
    from foldjax.models.boltz2.models.trunk_blocks import msa, trunk

    root = args.reference.resolve()
    reference = json.loads((root / "report.json").read_text())
    if reference.get("arm") != "native" or reference.get("passed") is not True:
        raise ValueError("standalone native reproduction has not passed")
    for name, digest in reference["artifacts"].items():
        verify_bound_file(root / name, digest)
    values = arrays(root / "operands.npz")
    validate_operands(values, reference["profile"])
    prefix = "d:trunk/d:msa_module/"
    with safe_open(args.weights, framework="numpy") as handle:
        encoded = {
            k[len(prefix) :]: handle.get_tensor(k)
            for k in handle.keys()
            if k.startswith(prefix)
        }
    candidate = unflatten_pytree(encoded, {})
    expected = map_msa_module_state_dict(arrays(root / "native-weights.npz"))
    a, b = flatten(candidate), flatten(expected)
    if set(a) != set(b) or any(
        a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k]) for k in a
    ):
        raise ValueError("converted MSA weights differ from original native checkpoint")
    params = trunk._cast_trunk_params({"msa_module": candidate}, jnp.bfloat16)[
        "msa_module"
    ]
    operands = {
        k: jnp.asarray(v.astype(np.int32) if v.dtype == np.int64 else v)
        for k, v in values.items()
    }
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(inspect.getfile(msa)).resolve().parents[6]
    before = source_hashes(source, "src")
    observer = Observer(args.out, "foldjax")
    options = compiler_options("bfloat16")
    if options != {"xla_allow_excess_precision": False}:
        raise ValueError("MSA probe requires preserved BF16 rounding boundaries")

    def run(params, values):
        return msa.msa_module_forward(
            params,
            values["input_z"],
            values["emb"],
            {k: values[k] for k in FEATURES},
            use_scan=True,
            chunk_size=128,
            triangle_backend="cueq",
            matmul_precision="highest",
            subsample_msa=False,
        )

    with (
        ExitStack() as controls,
        observer.jax_hooks(msa),
        jax.default_matmul_precision("highest"),
        patch.dict(os.environ, {"BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND": "cueq"}),
    ):
        if args.native_msa_norms:
            from bench.boltz_native_layer_norm import native_layer_norm
            from foldjax.models.boltz2.models.primitives import transition

            def norm(x, scale, bias, eps):
                return native_layer_norm(x.astype(jnp.float32), scale, bias, eps)[0]

            controls.enter_context(patch.object(msa, "_layer_norm", norm))
            controls.enter_context(patch.object(transition, "_layer_norm", norm))
        if args.native_dense_embedding:
            controls.enter_context(
                patch.object(msa, "_msa_input_embedding", native_dense_msa_embedding)
            )
        if args.native_row_shapes:
            for name in ("pair_weighted_averaging_forward", "transition_forward"):
                controls.enter_context(
                    patch.object(msa, name, full_row_control(getattr(msa, name)))
                )
        result = jax.jit(run, compiler_options=options)(params, operands)
        result.block_until_ready()
        jax.effects_barrier()
    validate_counts(observer.counts)
    save_arrays(args.out / "output.npz", {"z": np.asarray(result)})
    stages = {}
    for name, identity in reference["stages"].items():
        verify_bound_file(root / f"{name}.npz", identity["arrays_sha256"])
        verify_bound_file(root / f"{name}.tree.json", identity["tree_sha256"])
        actual, target = arrays(args.out / f"{name}.npz"), arrays(root / f"{name}.npz")
        if set(actual) != set(target):
            raise ValueError(f"stage schema differs: {name}")
        stages[name] = {key: comparison(actual[key], target[key]) for key in actual}
    if source_hashes(source, "src") != before:
        raise ValueError("FoldJAX source changed during probe")
    output_comparison = comparison(np.asarray(result), arrays(root / "output.npz")["z"])
    save_new(
        args.out / "report.json",
        {
            "arm": "foldjax",
            "capture_complete": True,
            "not_model_parity_admission": True,
            "controls": {
                "native_msa_norms": args.native_msa_norms,
                "native_dense_embedding": args.native_dense_embedding,
                "native_row_shapes": args.native_row_shapes,
                "normalization_scope": "MSA norms and MSA/pair transitions only",
                "row_shape_scope": (
                    "PWA and MSA/pair transitions only; "
                    "native hidden/head chunks retained"
                ),
            },
            "diagnostic_norm_sha256": sha(
                Path(__file__).with_name("boltz_native_layer_norm.py")
            )
            if args.native_msa_norms
            else None,
            "output_comparison": output_comparison,
            "stage_comparisons": stages,
            "counts": dict(observer.counts),
            "stages": observer.artifacts,
            "native_report_sha256": sha(root / "report.json"),
            "weights_sha256": sha(args.weights),
            "original_weight_leaves_exact": len(a),
            "source": before,
            "wrapper_sha256": sha(Path(__file__)),
            "compiler_options": options,
            "npz_compression": "deflate" if args.compressed else "stored",
            "runtime": {
                "jax": jax.__version__,
                "device": str(jax.devices()[0]),
                "device_kind": jax.devices()[0].device_kind,
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
            },
            "output_sha256": sha(args.out / "output.npz"),
        },
    )
    print(json.dumps({"output_comparison": output_comparison}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="arm", required=True)
    for name in ("native", "foldjax"):
        child = modes.add_parser(name)
        child.add_argument("--reference", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
        child.add_argument("--compressed", action="store_true")
        if name == "native":
            child.add_argument("--upstream", type=Path, required=True)
            child.add_argument("--checkpoint", type=Path, required=True)
        else:
            child.add_argument("--weights", type=Path, required=True)
            child.add_argument("--native-msa-norms", action="store_true")
            child.add_argument("--native-dense-embedding", action="store_true")
            child.add_argument("--native-row-shapes", action="store_true")
    args = parser.parse_args()
    # Full MSA stage arrays are large; stored NPZ avoids spending most of this
    # operator diagnostic in compression. Shared boundary writers use NumPy's
    # compressed entry point, so redirect only this dedicated process's writes.
    with patch.object(
        np, "savez_compressed", np.savez_compressed if args.compressed else np.savez
    ):
        (native if args.arm == "native" else foldjax)(args)


if __name__ == "__main__":
    main()
