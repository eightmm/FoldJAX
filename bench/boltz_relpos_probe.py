"""Isolate Boltz relative-position indices, weights, and BF16 GEMM reduction.

This is a captured-feature operator counterfactual, not independent input
parity, model admission, or a benchmark. Native and FoldJAX arms run in their
own existing environments; importing this module requires only NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

FEATURES = (
    "asym_id",
    "residue_index",
    "entity_id",
    "token_index",
    "sym_id",
    "cyclic_period",
)
PROFILE = {"r_max": 32, "s_max": 2, "fix_sym_check": True, "cyclic_pos_enc": True}
FJ_WEIGHT_KEY = "d:trunk/d:rel_pos/d:linear_layer/d:kernel"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_features(features):
    if set(features) != set(FEATURES):
        raise ValueError(
            "relative-position features must contain exactly the six consumed leaves"
        )
    shape = np.asarray(features[FEATURES[0]]).shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("expected nonempty batch/token feature axes")
    for key, value in features.items():
        value = np.asarray(value)
        if (
            value.shape != shape
            or value.dtype.kind not in "ifu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"invalid relative-position feature: {key}")
        if key != "cyclic_period" and value.dtype.kind not in "iu":
            raise ValueError(f"discrete feature is not integer: {key}")


def dense_relative_features(features):
    """Independent NumPy encoder for the fixed released native profile."""
    validate_features(features)
    asym, residue, entity, token, sym, period = (features[key] for key in FEATURES)
    same_chain = asym[:, :, None] == asym[:, None, :]
    same_residue = residue[:, :, None] == residue[:, None, :]
    same_entity = entity[:, :, None] == entity[:, None, :]
    delta = residue[:, :, None] - residue[:, None, :]
    if np.any(period > 0):
        period = np.where(period > 0, period, 10000)
        delta = (delta - period * np.rint(delta / period)).astype(np.int64)
    residue_bin = np.where(same_chain, np.clip(delta + 32, 0, 64), 65)
    token_bin = np.where(
        same_chain & same_residue,
        np.clip(token[:, :, None] - token[:, None, :] + 32, 0, 64),
        65,
    )
    chain_bin = np.where(
        ~same_entity, 5, np.clip(sym[:, :, None] - sym[:, None, :] + 2, 0, 4)
    )
    return np.concatenate(
        (
            np.eye(66, dtype=np.float32)[residue_bin],
            np.eye(66, dtype=np.float32)[token_bin],
            same_entity[..., None].astype(np.float32),
            np.eye(6, dtype=np.float32)[chain_bin],
        ),
        axis=-1,
    )


def bf16_round(value):
    """Round finite real values directly to BF16, returning lossless FP32 storage."""
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("BF16 rounding probe requires finite operands")
    _, exponent = np.frexp(value)
    quantum = np.ldexp(np.ones(value.shape), np.maximum(exponent - 8, -133))
    return (np.rint(value / quantum) * quantum).astype(np.float32)


def exact_sparse_projection(dense, weight):
    """Sum the actual binary input's 3/4 selected BF16 weights in float64."""
    dense, weight = np.asarray(dense), np.asarray(weight)
    if dense.shape[-1] != 139 or weight.ndim != 2 or weight.shape[1] != 139:
        raise ValueError("released relative-position projection requires K=139")
    if not np.isin(dense, [0, 1]).all():
        raise ValueError("relative-position dense input is not binary")
    ranges = ((0, 66), (66, 132), (133, 139))
    for lo, hi in ranges:
        if not np.all(dense[..., lo:hi].sum(-1) == 1):
            raise ValueError("each categorical relative-position block must be one-hot")
    table = bf16_round(weight).astype(np.float64).T
    selected = [np.argmax(dense[..., lo:hi], -1) + lo for lo, hi in ranges]
    result = table[selected[0]] + table[selected[1]]
    result += dense[..., 132, None] * table[132]
    result += table[selected[2]]
    return result


def comparison(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError("projection shapes differ")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite projection output")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(difference), initial=0)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "unequal": int(np.count_nonzero(difference)),
        "values_equal": bool(np.array_equal(actual, expected)),
    }


def compare_results(results, reference):
    return {name: comparison(value, reference) for name, value in results.items()}


@contextmanager
def torch_policy(torch, *, reduction=None, precision=None):
    previous_reduction = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    previous_precision = torch.get_float32_matmul_precision()
    try:
        if reduction is not None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                reduction
            )
        if precision is not None:
            torch.set_float32_matmul_precision(precision)
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            previous_reduction
        )
        torch.set_float32_matmul_precision(previous_precision)


def native(args):
    import torch

    reference = args.reference.resolve()
    provenance = json.loads((reference / "provenance.json").read_text())
    complete = json.loads((reference / "capture-complete.json").read_text())
    if complete.get("passed") is not True:
        raise ValueError("native reference is incomplete")
    stage = reference / "trunk-boundaries/rel_pos.npz"
    binding = complete["artifacts"]["trunk-boundaries/rel_pos"]
    if (
        sha(stage) != binding["arrays_sha256"]
        or sha(stage.with_suffix(".tree.json")) != binding["tree_sha256"]
    ):
        raise ValueError("native relative-position reference hash mismatch")
    tree = json.loads(stage.with_suffix(".tree.json").read_text())
    if tree[""]["native_dtype"] != "torch.bfloat16":
        raise ValueError("reference is not native BF16 relative-position output")
    checkpoint_sha = sha(args.checkpoint)
    if checkpoint_sha != provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint does not match the captured publisher run")
    upstream = args.upstream.resolve()
    source_path = upstream / "src/boltz/model/modules/encodersv2.py"
    source_sha = sha(source_path)
    if (
        source_sha
        != provenance["upstream_python_source"][str(source_path.relative_to(upstream))]
    ):
        raise ValueError("native encoder source differs from reference")
    sys.path.insert(0, str(upstream / "src"))
    from boltz.model.modules.encodersv2 import RelativePositionEncoder

    if Path(inspect.getfile(RelativePositionEncoder)).resolve() != source_path:
        raise ValueError("imported another Boltz encoder")
    features_sha = sha(reference / "features.npz")
    with np.load(reference / "features.npz", allow_pickle=False) as archive:
        features = {key: archive[key] for key in FEATURES}
    validate_features(features)
    settings = json.loads((reference / "effective-model-settings.json").read_text())
    if (
        settings["float32_matmul_precision"] != "highest"
        or settings["cuda_matmul_allow_tf32"]
    ):
        raise ValueError(
            "probe currently targets the captured highest/TF32-disabled profile"
        )
    # The local publisher checkpoint is explicitly supplied and hash-bound to
    # the completed capture; mmap avoids copying unrelated model parameters.
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    weight = checkpoint["state_dict"]["rel_pos.linear_layer.weight"].detach().clone()
    del checkpoint
    if weight.dtype != torch.float32 or weight.shape[1] != 139:
        raise ValueError("unexpected native relative-position weight")
    model = RelativePositionEncoder(weight.shape[0], **PROFILE).eval()
    model.load_state_dict({"linear_layer.weight": weight}, strict=True)
    model = model.cuda()
    tensors = {
        key: torch.from_numpy(value.copy()).cuda() for key, value in features.items()
    }
    captured_input, hook_calls = [], []

    def pre_hook(module, inputs):
        if len(inputs) != 1 or inputs[0].dtype != torch.float32:
            raise ValueError("unexpected native pre-Linear input")
        value = inputs[0].detach().cpu().numpy()
        if captured_input:
            if not np.array_equal(value, captured_input[0]):
                raise ValueError(
                    "dense input changed between reduction counterfactuals"
                )
        else:
            captured_input.append(value.copy())
        hook_calls.append({"dtype": str(inputs[0].dtype), "shape": list(value.shape)})

    handle = model.linear_layer.register_forward_pre_hook(pre_hook)
    outputs, policies = {}, {}
    default_reduction = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    try:
        with torch.inference_mode():
            for label, reduction in (
                ("autocast_default", default_reduction),
                ("autocast_reduction_true", True),
                ("autocast_reduction_false", False),
            ):
                with torch_policy(torch, reduction=reduction, precision="highest"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(tensors)
                    if output.dtype != torch.bfloat16:
                        raise ValueError("native autocast did not return BF16")
                    policies[label] = {
                        "bf16_reduced_precision_reduction": reduction,
                        "float32_matmul_precision": (
                            torch.get_float32_matmul_precision()
                        ),
                        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "output_dtype": str(output.dtype),
                    }
                    outputs[label] = output.float().cpu().numpy()
            if len(hook_calls) != 3:
                raise ValueError("missing or extra native Linear hook calls")
            dense = captured_input[0]
            with torch_policy(torch, reduction=False, precision="highest"):
                with torch.autocast("cuda", enabled=False):
                    control = torch.nn.functional.linear(
                        torch.from_numpy(dense).cuda().bfloat16().float(),
                        weight.cuda().bfloat16().float(),
                    )
                outputs["fp32_dot_of_bf16_operands"] = control.cpu().numpy()
                outputs["fp32_dot_of_bf16_operands_then_bf16"] = (
                    control.bfloat16().float().cpu().numpy()
                )
    finally:
        handle.remove()
    if not np.array_equal(dense_relative_features(features), dense):
        raise ValueError(
            "independent dense feature formula differs from actual native input"
        )
    outputs["cpu_exact_sum_fp64"] = exact_sparse_projection(dense, weight.numpy())
    outputs["cpu_exact_sum_then_bf16"] = bf16_round(outputs["cpu_exact_sum_fp64"])
    captured = arrays(stage)[""]
    outputs["captured_native"] = captured
    report = {
        "scope": __doc__,
        "arm": "native",
        "profile": PROFILE,
        "reference": str(reference),
        "checkpoint_sha256": checkpoint_sha,
        "features_sha256": features_sha,
        "reference_stage_sha256": sha(stage),
        "reference_provenance_sha256": sha(reference / "provenance.json"),
        "source_sha256": source_sha,
        "wrapper_sha256": sha(Path(__file__)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "policies": policies,
        "initial_reduction_policy": default_reduction,
        "hook_calls": hook_calls,
        "dense_formula_exact": True,
        "vs_captured_native": compare_results(outputs, captured),
        "vs_cpu_exact_bf16": compare_results(
            outputs, outputs["cpu_exact_sum_then_bf16"]
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.out / "operands.npz", dense=dense, weight=weight.numpy(), **features
    )
    np.savez_compressed(args.out / "outputs.npz", **outputs)
    if sha(source_path) != source_sha:
        raise ValueError("native encoder source changed during probe")
    if sha(reference / "features.npz") != features_sha:
        raise ValueError("native features changed during probe")
    report["artifacts"] = {
        name: sha(args.out / name) for name in ("operands.npz", "outputs.npz")
    }
    save(args.out / "report.json", report)
    print(json.dumps(report["vs_captured_native"], indent=2), flush=True)


def foldjax(args):
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.boltz2.models.primitives._common import linear
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        relative_position_forward,
    )

    reference = json.loads((args.reference / "report.json").read_text())
    if (
        reference.get("arm") != "native"
        or reference.get("dense_formula_exact") is not True
    ):
        raise ValueError("not a completed native operator probe")
    for name, expected in reference["artifacts"].items():
        if sha(args.reference / name) != expected:
            raise ValueError(f"native probe artifact changed: {name}")
    operands, native_outputs = (
        arrays(args.reference / "operands.npz"),
        arrays(args.reference / "outputs.npz"),
    )
    features = {key: operands[key] for key in FEATURES}
    validate_features(features)
    with safe_open(args.weights, framework="np") as archive:
        weight = archive.get_tensor(FJ_WEIGHT_KEY)
    if weight.dtype != np.float32 or not np.array_equal(weight, operands["weight"].T):
        raise ValueError(
            "FoldJAX original relative-position weight differs from native weight.T"
        )
    for key in FEATURES:
        if key != "cyclic_period" and np.any(
            features[key] != features[key].astype(np.int32)
        ):
            raise ValueError(
                "native feature cannot be represented exactly in JAX int32"
            )
    feats = {
        key: jnp.asarray(value.astype(np.int32) if key != "cyclic_period" else value)
        for key, value in features.items()
    }
    kernel = jnp.asarray(weight, dtype=jnp.bfloat16)
    dense = jnp.asarray(operands["dense"])
    with jax.default_matmul_precision("highest"):
        encoded = relative_position_forward(
            {"linear_layer": {"kernel": jnp.eye(139)}}, feats, **PROFILE
        )
        if not np.array_equal(np.asarray(encoded), operands["dense"]):
            raise ValueError(
                "production sparse feature indices differ from native dense input"
            )
        results = {
            "foldjax_sparse": jax.jit(
                lambda p, f: relative_position_forward(p, f, **PROFILE)
            )({"linear_layer": {"kernel": kernel}}, feats),
            "foldjax_sparse_cast_inside_jit": jax.jit(
                lambda w, f: relative_position_forward(
                    {"linear_layer": {"kernel": w.astype(jnp.bfloat16)}},
                    f,
                    **PROFILE,
                )
            )(jnp.asarray(weight), feats),
            "foldjax_sparse_no_excess_precision": jax.jit(
                lambda w, f: relative_position_forward(
                    {"linear_layer": {"kernel": w.astype(jnp.bfloat16)}},
                    f,
                    **PROFILE,
                ),
                compiler_options={"xla_allow_excess_precision": False},
            )(jnp.asarray(weight), feats),
            "foldjax_sparse_barrier_after_cast": jax.jit(
                lambda w, f: relative_position_forward(
                    {
                        "linear_layer": {
                            "kernel": jax.lax.optimization_barrier(
                                w.astype(jnp.bfloat16)
                            )
                        }
                    },
                    f,
                    **PROFILE,
                )
            )(jnp.asarray(weight), feats),
            "foldjax_sparse_reduce_precision": jax.jit(
                lambda w, f: relative_position_forward(
                    {
                        "linear_layer": {
                            "kernel": jax.lax.reduce_precision(
                                w, exponent_bits=8, mantissa_bits=7
                            ).astype(jnp.bfloat16)
                        }
                    },
                    f,
                    **PROFILE,
                )
            )(jnp.asarray(weight), feats),
            "foldjax_sparse_fp32_control": jax.jit(
                lambda w, f: relative_position_forward(
                    {"linear_layer": {"kernel": w}}, f, **PROFILE
                )
            )(jnp.asarray(weight), feats),
            "foldjax_dense_shared_linear": jax.jit(linear)(dense, kernel),
            "foldjax_dense_fp32_accum_then_bf16": jax.jit(
                lambda x, w: jnp.matmul(
                    x.astype(jnp.bfloat16), w, preferred_element_type=jnp.float32
                ).astype(jnp.bfloat16)
            )(dense, kernel),
        }
        results = {
            key: np.asarray(value, dtype=np.float32) for key, value in results.items()
        }
    report = {
        "scope": __doc__,
        "arm": "foldjax",
        "profile": PROFILE,
        "wrapper_sha256": sha(Path(__file__)),
        "reference_report_sha256": sha(args.reference / "report.json"),
        "weights_sha256": sha(args.weights),
        "weight_key": FJ_WEIGHT_KEY,
        "original_weight_exact": True,
        "production_feature_indices_exact": True,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "matmul_precision": "highest",
        "source_sha256": {
            str(Path(inspect.getfile(fn)).resolve()): sha(inspect.getfile(fn))
            for fn in (linear, relative_position_forward)
        },
        "vs_captured_native": compare_results(
            results, native_outputs["captured_native"]
        ),
        "vs_native_reduction_true": compare_results(
            results, native_outputs["autocast_reduction_true"]
        ),
        "vs_native_reduction_false": compare_results(
            results, native_outputs["autocast_reduction_false"]
        ),
        "vs_cpu_exact_bf16": compare_results(
            results, native_outputs["cpu_exact_sum_then_bf16"]
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "outputs.npz", **results)
    report["outputs_sha256"] = sha(args.out / "outputs.npz")
    save(args.out / "report.json", report)
    print(json.dumps(report["vs_captured_native"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="arm", required=True)
    for name in ("native", "foldjax"):
        command = commands.add_parser(name)
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "native":
            command.add_argument("--upstream", type=Path, required=True)
            command.add_argument("--checkpoint", type=Path, required=True)
        else:
            command.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new output directory")
    (native if args.arm == "native" else foldjax)(args)


if __name__ == "__main__":
    main()
