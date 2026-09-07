"""Native-reproduced conditioning leaf operators, not full-model admission.

Capture the actual native module before replaying each leaf with the exact
native operand. This separates local operator error from upstream propagation.
The full module must reproduce the original five-sample capture's conditioning.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_conditioning_probe import FEATURES, OUTPUTS
from bench.boltz_downstream_probe import bound_arrays, load_trunk, source_identity
from bench.boltz_foldjax_capture import native_settings
from bench.boltz_msa_probe import source_hashes, verify_bound_file
from bench.boltz_relpos_probe import arrays, bf16_round, comparison, torch_policy


def assert_source_origin(source, functions):
    for function in functions:
        if not Path(inspect.getfile(function)).resolve().is_relative_to(source / "src"):
            raise ValueError(
                f"imported source outside selected tree: {function.__name__}"
            )


def welford4_moments(x, eps, *, fenced=False):
    """Diagnostic emulation of native CUDA vector-4 Welford reduction order.

    Only the three observed conditioning widths are covered. This is not a
    production kernel or a claim of identical GPU instruction selection.
    """
    import jax
    import jax.numpy as jnp

    boundary = jax.lax.optimization_barrier if fenced else lambda a: a

    width = x.shape[-1]
    if width not in (16, 128, 256):
        raise ValueError("Welford diagnostic supports widths 16, 128 and 256")
    groups = max(32, width // 4)
    values = jnp.pad(
        x.astype(jnp.float32), [(0, 0)] * (x.ndim - 1) + [(0, groups * 4 - width)]
    )
    values = values.reshape(*x.shape[:-1], groups // 32, 32, 4)
    mean = jnp.zeros(values.shape[:-1], jnp.float32)
    variance = jnp.zeros_like(mean)
    valid = jnp.arange(groups).reshape(groups // 32, 32) < width // 4
    for index in range(4):
        value = values[..., index]
        delta = boundary(value - mean)
        updated = boundary(mean + delta * jnp.float32(1.0 / (index + 1)))
        variance = boundary(variance + delta * boundary(value - updated))
        mean = updated
    count = jnp.broadcast_to(jnp.where(valid, 4.0, 0.0), mean.shape)

    def combine(a, b):
        # Native cuWelfordCombine's first argument is dataB, not dataA.
        mb, vb, cb = a
        ma, va, ca = b
        total = ca + cb
        reciprocal = jnp.where(total > 0, 1.0 / total, 0.0)
        na, nb = ca * reciprocal, cb * reciprocal
        delta = boundary(mb - ma)
        correction = boundary(boundary(delta * delta) * ca)
        return (
            boundary(na * ma + nb * mb),
            boundary(boundary(va + vb) + correction * nb),
            total,
        )

    data = (mean, variance, count)
    for offset in (16, 8, 4, 2, 1):
        data = combine(
            tuple(a[..., :offset] for a in data),
            tuple(a[..., offset : 2 * offset] for a in data),
        )
    data = tuple(a[..., 0] for a in data)
    if groups == 64:
        data = combine(
            tuple(a[..., :1] for a in data), tuple(a[..., 1:2] for a in data)
        )
    mean, variance, _ = data
    return mean, jax.lax.rsqrt(boundary(boundary(variance / width) + eps))


def welford4_layer_norm(x, scale, bias, eps):
    mean, rstd = welford4_moments(x, eps)
    return scale * (rstd * (x - mean)) + bias


def native_stats(args):
    import torch

    root = args.reference.resolve()
    report = json.loads((root / "report.json").read_text())
    if report.get("passed") is not True or report["torch"] != torch.__version__:
        raise ValueError("requires reproduced native conditioning and same Torch")
    verify_bound_file(root / "weights.npz", report["weights_sha256"])
    weights = arrays(root / "weights.npz")
    args.out.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, stage in report["stages"].items():
        if stage["kind"] != "LayerNorm":
            continue
        verify_bound_file(root / f"{name}.npz", stage["sha256"])
        values = arrays(root / f"{name}.npz")
        x = torch.from_numpy(values["input"].copy()).cuda()
        scale = torch.from_numpy(weights[f"{name}.weight"].copy()).cuda()
        bias = torch.from_numpy(weights[f"{name}.bias"].copy()).cuda()
        with torch.inference_mode():
            output, mean, rstd = torch.native_layer_norm(
                x, (x.shape[-1],), scale, bias, 1e-5
            )
            torch.cuda.synchronize()
        check = comparison(output.cpu().numpy(), values["output"])
        if not check["values_equal"]:
            raise ValueError(f"native statistics replay changed LayerNorm: {name}")
        path = args.out / f"{name}.npz"
        with path.open("xb") as f:
            np.savez(f, mean=mean.cpu().numpy(), rstd=rstd.cpu().numpy())
        records[name] = {"sha256": sha(path), "reproduction": check}
    save_new(
        args.out / "report.json",
        {
            "passed": True,
            "arm": "native-stats",
            "records": records,
            "native_report_sha256": sha(root / "report.json"),
            "wrapper_sha256": sha(Path(__file__)),
            "torch": torch.__version__,
        },
    )


def profile_from_hparams(hparams):
    names = (
        "token_s",
        "token_z",
        "atom_s",
        "atom_z",
        "atoms_per_window_queries",
        "atoms_per_window_keys",
        "atom_feature_dim",
        "use_no_atom_char",
        "use_atom_backbone_feat",
        "use_residue_feats_atoms",
    )
    profile = {name: hparams[name] for name in names}
    for name in (
        "atom_encoder_depth",
        "atom_encoder_heads",
        "token_transformer_depth",
        "token_transformer_heads",
        "atom_decoder_depth",
        "atom_decoder_heads",
        "conditioning_transition_layers",
    ):
        profile[name] = hparams["score_model_args"][name]
    if any(
        profile[k]
        for k in (
            "use_no_atom_char",
            "use_atom_backbone_feat",
            "use_residue_feats_atoms",
        )
    ):
        raise ValueError("probe does not cover optional atom feature profiles")
    return profile


def selected_operators():
    names = [
        "pairwise_conditioner.dim_pairwise_init_proj.0",
        "pairwise_conditioner.dim_pairwise_init_proj.1",
    ]
    for index in range(2):
        names.extend(
            f"pairwise_conditioner.transitions.{index}.{leaf}"
            for leaf in ("norm", "fc1", "fc2", "silu", "fc3")
        )
    for group in ("atom_enc_proj_z", "atom_dec_proj_z", "token_trans_proj_z"):
        names.extend(f"{group}.0.{leaf}" for leaf in ("0", "1"))
    return names


def native(args):
    import torch

    root, upstream = args.reference.resolve(), args.upstream.resolve()
    meta, _ = native_settings(root)
    provenance = json.loads((root / "provenance.json").read_text())
    verify_bound_file(args.checkpoint, provenance["checkpoint_sha256"])
    verify_bound_file(root / "features.npz", args.features_sha256)
    source = source_hashes(upstream, "src")
    if source != provenance["upstream_python_source"]:
        raise ValueError("native source differs from original capture")
    trunk, _ = load_trunk(root, meta)
    expected, _ = bound_arrays(root, "trunk-boundaries/diffusion_conditioning", OUTPUTS)
    sys.path.insert(0, str(upstream / "src"))
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning

    if Path(inspect.getfile(DiffusionConditioning)).resolve() != (
        upstream / "src/boltz/model/modules/diffusion_conditioning.py"
    ):
        raise ValueError("wrong native module imported")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    profile = profile_from_hparams(checkpoint["hyper_parameters"])
    state = {
        k.removeprefix("diffusion_conditioning."): v.detach().clone()
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("diffusion_conditioning.")
    }
    del checkpoint
    model = DiffusionConditioning(**profile).eval()
    model.load_state_dict(state, strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "weights.npz").open("xb") as f:
        np.savez(f, **{k: v.numpy() for k, v in state.items()})
    stages = {}

    def hook(name):
        def observe(module, inputs, output):
            if name in stages:
                raise ValueError(f"duplicate conditioning operator: {name}")
            kind = type(module).__name__
            if kind not in {"LayerNorm", "Linear", "SiLU"} or len(inputs) != 1:
                raise ValueError(f"unsupported leaf operator: {name}/{kind}")
            path = args.out / f"{name}.npz"
            with path.open("xb") as f:
                np.savez(
                    f,
                    input=inputs[0].detach().float().cpu().numpy(),
                    output=output.detach().float().cpu().numpy(),
                )
            stages[name] = {
                "kind": kind,
                "input_dtype": str(inputs[0].dtype),
                "output_dtype": str(output.dtype),
                "sha256": sha(path),
            }

        return observe

    with np.load(root / "features.npz", allow_pickle=False) as f:
        features = {k: torch.from_numpy(f[k].copy()).cuda() for k in FEATURES}
    trunk = {k: torch.from_numpy(v.copy()).cuda() for k, v in trunk.items()}
    model.cuda()
    with (
        ExitStack() as hooks,
        torch_policy(torch, precision="highest"),
        torch.inference_mode(),
    ):
        for name in selected_operators():
            hooks.callback(
                model.get_submodule(name).register_forward_hook(hook(name)).remove
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, c, _, enc, dec, token = model(
                trunk["s"], trunk["z"], trunk["relative_position_encoding"], features
            )
        torch.cuda.synchronize()
    if set(stages) != set(selected_operators()):
        raise ValueError("incomplete native conditioning trace")
    outputs = dict(zip(OUTPUTS, (q, c, enc, dec, token), strict=True))
    reproduction = {
        k: comparison(v.detach().float().cpu().numpy(), expected[k])
        for k, v in outputs.items()
    }
    passed = all(v["values_equal"] for v in reproduction.values())
    if source != source_hashes(upstream, "src"):
        raise ValueError("native source changed during capture")
    verify_bound_file(args.checkpoint, provenance["checkpoint_sha256"])
    verify_bound_file(root / "features.npz", args.features_sha256)
    save_new(
        args.out / "report.json",
        {
            "arm": "native",
            "passed": passed,
            "not_model_parity_admission": True,
            "reproduction": reproduction,
            "stages": stages,
            "profile": profile,
            "weights_sha256": sha(args.out / "weights.npz"),
            "native_source": source,
            "wrapper_sha256": sha(Path(__file__)),
            "torch": torch.__version__,
            "features_sha256": args.features_sha256,
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "reference_complete_sha256": sha(root / "capture-complete.json"),
        },
    )
    print(json.dumps({"passed": passed, "reproduction": reproduction}), flush=True)
    if not passed:
        raise RuntimeError("native conditioning failed original-output reproduction")


def foldjax(args):
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.boltz2.bridge.native import unflatten_pytree
    from foldjax.models.boltz2.bridge.torch_mapping import (
        map_diffusion_conditioning_state_dict,
    )
    from foldjax.models.boltz2.compile_policy import compiler_options
    from foldjax.models.boltz2.models.primitives._common import layer_norm, linear
    from foldjax.models.boltz2.models.primitives.native_amp_norm import (
        amp_affine,
        amp_layer_norm,
    )

    source = Path(__file__).resolve().parents[1]
    assert_source_origin(source, (layer_norm, linear, amp_layer_norm, amp_affine))
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("conditioning operator replay requires exactly one GPU")

    root = args.reference.resolve()
    report = json.loads((root / "report.json").read_text())
    if report.get("arm") != "native" or report.get("passed") is not True:
        raise ValueError("native original-output reproduction must pass first")
    verify_bound_file(root / "weights.npz", report["weights_sha256"])
    weights = arrays(root / "weights.npz")
    prefix = "d:conditioned_diffusion/d:diffusion_conditioning/"
    with safe_open(args.weights, framework="numpy") as f:
        encoded = {
            k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)
        }
    converted = unflatten_pytree(encoded, {})
    mapped = map_diffusion_conditioning_state_dict(
        {f"diffusion_conditioning.{k}": v for k, v in weights.items()}
    )
    if jax.tree.structure(mapped) != jax.tree.structure(converted):
        raise ValueError("native/converted conditioning weight structures differ")
    for a, b in zip(jax.tree.leaves(mapped), jax.tree.leaves(converted), strict=True):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            raise ValueError("native/converted conditioning weights differ")
    sources = source_identity(Path(__file__).resolve().parents[1])
    weight_hash = sha(args.weights)
    stats_report = None
    if args.native_stats:
        stats_report = json.loads((args.native_stats / "report.json").read_text())
        if stats_report.get("passed") is not True or stats_report.get(
            "native_report_sha256"
        ) != sha(root / "report.json"):
            raise ValueError("native statistics are not bound to this replay")
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, stage in report["stages"].items():
        path = root / f"{name}.npz"
        verify_bound_file(path, stage["sha256"])
        values = arrays(path)
        x = jnp.asarray(values["input"])
        scale = (
            jnp.asarray(weights[f"{name}.weight"])
            if stage["kind"] != "SiLU"
            else jnp.ones(())
        )
        bias = jnp.asarray(
            weights.get(f"{name}.bias", np.zeros(scale.shape[0] if scale.ndim else ()))
        )

        def run(x, scale, bias):
            if stage["kind"] == "LayerNorm":
                return layer_norm(x, scale, bias, 1e-5)
            if stage["kind"] == "Linear":
                return linear(x, scale.T, compute_dtype=jnp.bfloat16)
            return jax.nn.silu(x).astype(jnp.bfloat16)

        with jax.default_matmul_precision("highest"):
            output = jax.jit(run, compiler_options=compiler_options("bfloat16"))(
                x, scale, bias
            )
            output.block_until_ready()
        stored = np.asarray(output.astype(jnp.float32))
        results[name] = {"kind": stage["kind"], **comparison(stored, values["output"])}
        if stage["kind"] == "LayerNorm":
            production = np.asarray(jax.jit(amp_layer_norm)(x, scale, bias, 1e-5))
            results[name]["production_amp_norm"] = comparison(
                production, values["output"]
            )

            def shared_affine(x, scale, bias):
                normalized = amp_layer_norm(
                    x, jnp.ones_like(scale), jnp.zeros_like(bias), 1e-5
                )
                return amp_affine(normalized, scale, bias)

            shared = np.asarray(jax.jit(shared_affine)(x, scale, bias))
            results[name]["production_shared_norm_affine"] = comparison(
                shared, values["output"]
            )
            results[name]["bf16_operand"] = comparison(
                bf16_round(stored), bf16_round(values["output"])
            )
            if args.norm_controls:
                with jax.default_matmul_precision("highest"):
                    candidate = jax.jit(
                        welford4_layer_norm,
                        compiler_options=compiler_options("bfloat16"),
                    )(x, scale, bias, 1e-5)
                    candidate.block_until_ready()
                alternative = np.asarray(candidate)
                results[name]["welford4"] = comparison(alternative, values["output"])
                results[name]["welford4_bf16_operand"] = comparison(
                    bf16_round(alternative), bf16_round(values["output"])
                )
            if stats_report:
                path = args.native_stats / f"{name}.npz"
                verify_bound_file(path, stats_report["records"][name]["sha256"])
                moments = {k: jnp.asarray(v) for k, v in arrays(path).items()}

                def affine(x, scale, bias, mean, rstd):
                    return scale * (rstd * (x - mean)) + bias

                restored = jax.jit(affine)(x, scale, bias, **moments)
                restored = np.asarray(restored)
                results[name]["native_stats_affine"] = comparison(
                    restored, values["output"]
                )
                results[name]["native_stats_affine_bf16"] = comparison(
                    bf16_round(restored), bf16_round(values["output"])
                )

                def affine_barrier(x, scale, bias, mean, rstd):
                    centered = jax.lax.optimization_barrier(x - mean)
                    normalized = jax.lax.optimization_barrier(rstd * centered)
                    return scale * normalized + bias

                fenced = np.asarray(jax.jit(affine_barrier)(x, scale, bias, **moments))
                results[name]["native_stats_affine_barrier"] = comparison(
                    fenced, values["output"]
                )
                results[name]["native_stats_affine_barrier_bf16"] = comparison(
                    bf16_round(fenced), bf16_round(values["output"])
                )
                mean, rstd = jax.jit(welford4_moments)(x, 1e-5)
                results[name]["welford4_stats"] = {
                    "mean": comparison(np.asarray(mean), np.asarray(moments["mean"])),
                    "rstd": comparison(np.asarray(rstd), np.asarray(moments["rstd"])),
                }
                fenced_mean, fenced_rstd = jax.jit(
                    lambda x: welford4_moments(x, 1e-5, fenced=True)
                )(x)
                results[name]["welford4_fenced_stats"] = {
                    "mean": comparison(
                        np.asarray(fenced_mean), np.asarray(moments["mean"])
                    ),
                    "rstd": comparison(
                        np.asarray(fenced_rstd), np.asarray(moments["rstd"])
                    ),
                }
                fenced_all = np.asarray(
                    jax.jit(affine_barrier)(x, scale, bias, fenced_mean, fenced_rstd)
                )
                results[name]["welford4_fenced"] = comparison(
                    fenced_all, values["output"]
                )
                results[name]["welford4_fenced_bf16"] = comparison(
                    bf16_round(fenced_all), bf16_round(values["output"])
                )
                if args.explicit_fma:
                    from bench.boltz_native_layer_norm import native_layer_norm

                    exact, exact_mean, exact_rstd = jax.jit(native_layer_norm)(
                        x, scale, bias
                    )
                    exact = np.asarray(exact)
                    results[name]["explicit_fma"] = comparison(exact, values["output"])
                    results[name]["explicit_fma_bf16"] = comparison(
                        bf16_round(exact), bf16_round(values["output"])
                    )
                    results[name]["explicit_fma_stats"] = {
                        "mean": comparison(
                            np.asarray(exact_mean), np.asarray(moments["mean"])
                        ),
                        "rstd": comparison(
                            np.asarray(exact_rstd), np.asarray(moments["rstd"])
                        ),
                    }
        print(json.dumps({name: results[name]}), flush=True)
    if sources != source_identity(
        Path(__file__).resolve().parents[1]
    ) or weight_hash != sha(args.weights):
        raise ValueError("source/weights changed during probe")
    save_new(
        args.out / "report.json",
        {
            "arm": "foldjax",
            "capture_complete": True,
            "not_model_parity_admission": True,
            "same_native_operand": True,
            "weight_leaves_exact": len(jax.tree.leaves(mapped)),
            "results": results,
            "source": sources,
            "weights_sha256": weight_hash,
            "native_report_sha256": sha(root / "report.json"),
            "native_stats_report_sha256": sha(args.native_stats / "report.json")
            if args.native_stats
            else None,
            "compiler_options": compiler_options("bfloat16"),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax", "native-stats"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--features-sha256")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--norm-controls", action="store_true")
    parser.add_argument("--native-stats", type=Path)
    parser.add_argument("--explicit-fma", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    required = (
        ("upstream", "checkpoint", "features_sha256")
        if args.arm == "native"
        else ("weights",)
        if args.arm == "foldjax"
        else ()
    )
    if any(getattr(args, k) is None for k in required):
        parser.error(f"{args.arm} requires {required}")
    if args.explicit_fma and (args.arm != "foldjax" or not args.native_stats):
        parser.error("--explicit-fma requires foldjax and --native-stats")
    if args.arm != "foldjax" and (args.norm_controls or args.native_stats):
        parser.error("norm controls and native statistics are FoldJAX-only options")
    {"native": native, "foldjax": foldjax, "native-stats": native_stats}[args.arm](args)


if __name__ == "__main__":
    main()
