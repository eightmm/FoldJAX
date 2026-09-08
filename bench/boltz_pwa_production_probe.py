"""Actual default PWA on exact first-layer native inputs, not model admission.

Native layer0 m was BF16, archived losslessly as FP32. Keep those exact values
in FP32 to reproduce FoldJAX's scan-carry dtype without recomputing embedding.
This executes all eight heads, gates and projections, without a decomposition
or normalization patch. It does not identify individual head-stage errors.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_msa_probe import source_hashes, verify_bound_file
from bench.boltz_pwa_full_mma_probe import add_stats, empty_stats, finish_stats

ROWS, TOKENS, M_WIDTH, Z_WIDTH, HEADS, HEAD_WIDTH = 4436, 437, 64, 128, 8, 32
CHUNK_ROWS = 64


def array_identity(value):
    if not value.flags.c_contiguous:
        raise ValueError("captured arrays must retain contiguous storage")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
    }


def validate_fp32(value, *, bf16=False):
    if value.dtype != np.float32 or not value.flags.c_contiguous:
        raise ValueError("capture must use contiguous lossless FP32 storage")
    flat = value.reshape(-1)
    for start in range(0, flat.size, 1 << 20):
        part = flat[start : start + (1 << 20)]
        if not np.isfinite(part).all():
            raise ValueError("nonfinite native operand/output")
        if bf16 and np.any(part.view(np.uint32) & 0xFFFF):
            raise ValueError("native BF16 capture is not losslessly representable")


def require_same_bytes(left, right):
    if array_identity(left) != array_identity(right):
        raise ValueError("PWA arrays differ from the original native MSA capture")


def select_arrays(path, names, *, exact=False):
    with np.load(path, allow_pickle=False) as archive:
        if not set(names) <= set(archive.files) or (
            exact and set(names) != set(archive.files)
        ):
            raise ValueError("unknown or missing native array schema")
        return {name: archive[name] for name in names}


def weight_shapes():
    return {
        "norm_m.weight": (M_WIDTH,),
        "norm_m.bias": (M_WIDTH,),
        "norm_z.weight": (Z_WIDTH,),
        "norm_z.bias": (Z_WIDTH,),
        "proj_m.weight": (HEADS * HEAD_WIDTH, M_WIDTH),
        "proj_z.weight": (HEADS, Z_WIDTH),
        "proj_g.weight": (HEADS * HEAD_WIDTH, M_WIDTH),
        "proj_o.weight": (M_WIDTH, HEADS * HEAD_WIDTH),
    }


def load_reference(root, msa_root):
    report_bytes = (root / "report.json").read_bytes()
    msa_bytes = (msa_root / "report.json").read_bytes()
    report, msa = json.loads(report_bytes), json.loads(msa_bytes)
    if (
        report.get("arm") != "native"
        or report.get("passed") is not True
        or report.get("rows") != ROWS
        or report.get("native_decomposition", {}).get("values_equal") is not True
        or report.get("row_slice_vs_full", {}).get("values_equal") is not True
        or msa.get("arm") != "native"
        or msa.get("passed") is not True
        or report.get("native_source") != msa.get("native_source")
        or report.get("torch") != msa.get("runtime", {}).get("torch")
        or not report.get("native_source")
    ):
        raise ValueError("requires the same-source full native PWA/MSA reproduction")
    verify_bound_file(msa_root / "report.json", report["native_msa_report_sha256"])
    files = {"pwa/report.json": root / "report.json"}
    files["msa/report.json"] = msa_root / "report.json"
    for filename in ("inputs.npz", "weights.npz", "stages.npz"):
        verify_bound_file(root / filename, report["artifacts"][filename])
        files[f"pwa/{filename}"] = root / filename
    for filename in ("operands.npz", "native-weights.npz"):
        verify_bound_file(msa_root / filename, msa["artifacts"][filename])
        files[f"msa/{filename}"] = msa_root / filename
    shape = [1, ROWS, TOKENS, M_WIDTH]
    for stage in ("input_m", "pwa"):
        prefix = f"layers/00/{stage}"
        for suffix, key in ((".npz", "arrays_sha256"), (".tree.json", "tree_sha256")):
            path = msa_root / f"{prefix}{suffix}"
            verify_bound_file(path, msa["stages"][prefix][key])
            files[f"msa/{prefix}{suffix}"] = path
        metadata = json.loads((msa_root / f"{prefix}.tree.json").read_text())
        if metadata != {
            "": {
                "native_dtype": "torch.bfloat16",
                "storage_dtype": "float32",
                "shape": shape,
            }
        }:
            raise ValueError("first native m/PWA must be BF16, not true FP32 carry")
    binding = {name: sha(path) for name, path in files.items()}
    if (
        binding["pwa/report.json"] != hashlib.sha256(report_bytes).hexdigest()
        or binding["msa/report.json"] != hashlib.sha256(msa_bytes).hexdigest()
    ):
        raise ValueError("native report changed while loading")
    inputs = select_arrays(root / "inputs.npz", ("m", "z", "mask"), exact=True)
    expected_shapes = {
        "m": tuple(shape),
        "z": (1, TOKENS, TOKENS, Z_WIDTH),
        "mask": (1, TOKENS),
    }
    for name, value in inputs.items():
        if value.shape != expected_shapes[name]:
            raise ValueError("native PWA input shape differs")
        validate_fp32(value, bf16=name == "m")
    if not np.isin(inputs["mask"], (0, 1)).all():
        raise ValueError("native token mask must be binary")
    require_same_bytes(
        inputs["m"],
        select_arrays(msa_root / "layers/00/input_m.npz", ("",), exact=True)[""],
    )
    original = select_arrays(msa_root / "operands.npz", ("input_z", "token_pad_mask"))
    require_same_bytes(inputs["z"], original["input_z"])
    require_same_bytes(inputs["mask"], original["token_pad_mask"])
    del original
    weights = select_arrays(root / "weights.npz", weight_shapes(), exact=True)
    prefix = "msa_module.layers.0.pair_weighted_averaging."
    original = select_arrays(
        msa_root / "native-weights.npz", [prefix + name for name in weights]
    )
    for name, value in weights.items():
        if value.shape != weight_shapes()[name]:
            raise ValueError("native PWA weight shape differs")
        validate_fp32(value)
        require_same_bytes(value, original[prefix + name])
    del original
    # Materialize only the actual output, never the ~15GB per-head decomposition.
    target = select_arrays(root / "stages.npz", ("actual",))["actual"]
    if target.shape != tuple(shape):
        raise ValueError("native PWA output shape differs")
    validate_fp32(target, bf16=True)
    require_same_bytes(
        target, select_arrays(msa_root / "layers/00/pwa.npz", ("",), exact=True)[""]
    )
    if binding != {name: sha(path) for name, path in files.items()}:
        raise ValueError("native reference changed while loading")
    return inputs, weights, target, files, binding


def source_binding(root):
    return {**source_hashes(root, "src"), **source_hashes(root, "bench")}


def production_forward(params, values):
    from foldjax.models.boltz2.models.trunk_blocks import msa

    token_mask = values["mask"]
    return msa.pair_weighted_averaging_forward(
        params,
        values["m"],
        values["z"],
        token_mask[:, :, None] * token_mask[:, None, :],
        row_chunk_size=None,
    )


def native_logits_forward(params, values):
    """Counterfactual only: replace exactly eight head-wise logits projections."""
    from foldjax.models.boltz2.models.trunk_blocks import msa

    logits = values["native_logits"]
    if logits.shape != (1, 437, 437, 8):
        raise ValueError("native logits require the full eight-head reference")
    original = msa._linear
    seen = 0

    def linear(value, kernel, *args, **kwargs):
        nonlocal seen
        if value.shape == (1, 437, 437, 128) and kernel.shape == (128, 1):
            if seen >= 8:
                raise ValueError("unexpected extra logits projection")
            result = logits[..., seen:seen + 1].astype(kernel.dtype)
            seen += 1
            return result
        return original(value, kernel, *args, **kwargs)

    with patch.object(msa, "_linear", linear):
        result = production_forward(params, values)
    if seen != 8:
        raise ValueError("native logits intervention missed a projection")
    return result


def native_softmax_forward(params, values):
    """Counterfactual only: bypass exactly eight FP32 softmax outputs."""
    import jax

    weights = values["native_softmax"]
    if weights.shape != (1, 8, 437, 437):
        raise ValueError("native softmax requires the full eight-head reference")
    seen = 0

    def softmax(value, axis=-1, **kwargs):
        nonlocal seen
        if value.shape != (1, 1, 437, 437) or axis != -1 or kwargs or seen >= 8:
            raise ValueError("unexpected softmax boundary")
        result = weights[:, seen:seen + 1]
        seen += 1
        return result

    with patch.object(jax.nn, "softmax", softmax):
        result = production_forward(params, values)
    if seen != 8:
        raise ValueError("native softmax intervention missed a head")
    return result


def warp_softmax_forward(params, values):
    """Compute softmax from actual logits with the observed warp sum order."""
    import jax

    from bench.boltz_pwa_softmax_probe import warp_softmax

    seen = 0

    def softmax(value, axis=-1, **kwargs):
        nonlocal seen
        if value.shape != (1, 1, 437, 437) or axis != -1 or kwargs or seen >= 8:
            raise ValueError("unexpected warp softmax boundary")
        seen += 1
        return warp_softmax(value)

    with patch.object(jax.nn, "softmax", softmax):
        forward = (
            native_logits_forward if "native_logits" in values else production_forward
        )
        result = forward(params, values)
    if seen != 8:
        raise ValueError("warp softmax intervention missed a head")
    return result


def computed_logits_warp_forward(params, values):
    from bench.boltz_pwa_logits_probe import strided_four_forward
    from foldjax.models.boltz2.models.trunk_blocks import msa

    original = msa._linear
    seen = 0

    def linear(value, kernel, *args, **kwargs):
        nonlocal seen
        if value.shape == (1, 437, 437, 128) and kernel.shape == (128, 1):
            if args or kwargs or seen >= 8:
                raise ValueError("unexpected computed logits boundary")
            seen += 1
            return strided_four_forward(value, kernel)
        return original(value, kernel, *args, **kwargs)

    with patch.object(msa, "_linear", linear):
        result = warp_softmax_forward(params, values)
    if seen != 8:
        raise ValueError("computed logits missed a head")
    return result


def run_production(
    source, inputs, weights, out, *, warp_reduction=False, computed_logits=False
):
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.compile_policy import compiler_options
    from foldjax.models.boltz2.models.trunk_blocks import msa

    if Path(inspect.getfile(msa)).resolve() != (
        source / "src/foldjax/models/boltz2/models/trunk_blocks/msa.py"
    ):
        raise ValueError("FoldJAX imported outside the selected source snapshot")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("production PWA probe requires one CUDA GPU")
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
    operands = {name: jnp.asarray(value) for name, value in inputs.items()}

    rows = msa._auto_pair_averaging_chunk(operands["m"], params)
    if rows != 1199:
        raise ValueError("actual default row profile differs from 1199/1199/1199/839")
    options = compiler_options("bfloat16")
    if options != {"xla_allow_excess_precision": False}:
        raise ValueError("production BF16 compiler rounding policy changed")
    with jax.default_matmul_precision("highest"):
        compiled = (
            jax.jit(
                computed_logits_warp_forward if computed_logits else
                warp_softmax_forward if warp_reduction else
                native_softmax_forward if "native_softmax" in inputs else
                native_logits_forward
                if "native_logits" in inputs else production_forward,
                compiler_options=options,
            )
            .lower(params, operands)
            .compile()
        )
        hlo = out / "compiled.hlo.txt"
        with hlo.open("x") as stream:
            stream.write(compiled.as_text())
        result = compiled(params, operands)
        result.block_until_ready()
    if result.dtype != jnp.bfloat16 or result.shape != inputs["m"].shape:
        raise ValueError("production PWA output dtype/shape changed")
    return result, {
        "jax": jax.__version__,
        "device_kind": devices[0].device_kind,
        "platform": devices[0].platform,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "compiler_options": options,
        "compiled_hlo_sha256": sha(hlo),
        "row_chunk_size_argument": None,
        "actual_row_chunks": [1199, 1199, 1199, 839],
        "heads": HEADS,
        "production_output_dtype": "bfloat16",
        "instrumented_decomposition": False,
        "native_logits_intervention": "native_logits" in inputs,
        "native_softmax_intervention": "native_softmax" in inputs,
        "warp_reduction_control": warp_reduction,
        "computed_logits_control": computed_logits,
    }


def save_comparison(result, target, out):
    stats, chunks = empty_stats(), []
    for start in range(0, target.shape[1], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, target.shape[1])
        actual = np.asarray(result[:, start:stop], dtype=np.float32)
        expected = target[:, start:stop]
        add_stats(stats, actual, expected)
        path = out / f"rows-{start:04d}-{stop:04d}.npz"
        with path.open("xb") as stream:
            np.savez(stream, actual=actual)
        chunks.append(
            {"file": path.name, "sha256": sha(path), "start": start, "stop": stop}
        )
    return finish_stats(stats), chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reference", "msa-reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    intervention = parser.add_mutually_exclusive_group()
    intervention.add_argument("--native-logits", action="store_true")
    intervention.add_argument("--native-softmax", action="store_true")
    parser.add_argument("--warp-reduction", action="store_true")
    parser.add_argument("--computed-logits", action="store_true")
    args = parser.parse_args()
    if args.warp_reduction and args.native_softmax:
        parser.error("warp reduction cannot bypass softmax with reference weights")
    if args.computed_logits and (
        not args.warp_reduction or args.native_logits or args.native_softmax
    ):
        parser.error("computed logits requires warp reduction without substitutions")
    source, root, msa_root, out = (
        path.resolve()
        for path in (args.source_root, args.reference, args.msa_reference, args.out)
    )
    if Path(__file__).resolve() != source / "bench/boltz_pwa_production_probe.py":
        raise ValueError("execute the probe from the selected source snapshot")
    if any(out.is_relative_to(path) for path in (source, root, msa_root)):
        raise ValueError("output must be outside source and native references")
    if out.exists():
        raise FileExistsError(out)
    sources = source_binding(source)
    inputs, weights, target, files, binding = load_reference(root, msa_root)
    if args.native_logits:
        names = [f"head{head}/logits" for head in range(8)]
        stages = select_arrays(root / "stages.npz", names)
        logits = np.concatenate([stages[name] for name in names], axis=-1)
        validate_fp32(logits, bf16=True)
        if logits.shape != (1, 437, 437, 8):
            raise ValueError("native logits shape differs from full reference")
        inputs["native_logits"] = logits
    if args.native_softmax:
        names = [f"head{head}/weights" for head in range(8)]
        stages = select_arrays(root / "stages.npz", names)
        native_weights = np.concatenate([stages[name] for name in names], axis=1)
        validate_fp32(native_weights)
        if native_weights.shape != (1, 8, 437, 437):
            raise ValueError("native softmax shape differs from full reference")
        inputs["native_softmax"] = native_weights
    array_bindings = {
        "inputs": {name: array_identity(value) for name, value in inputs.items()},
        "weights": {name: array_identity(value) for name, value in weights.items()},
        "target": array_identity(target),
    }
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "capture_complete": False,
        "passed": False,
        "not_model_parity_admission": True,
        "native_logits_intervention": args.native_logits,
        "native_softmax_intervention": args.native_softmax,
        "warp_reduction_control": args.warp_reduction,
        "computed_logits_control": args.computed_logits,
        "scope": (
            "actual first-layer default full PWA, "
            "all eight heads including gates and projections"
        ),
        "source": sources,
        "reference_binding": binding,
        "array_bindings": array_bindings,
        "input_m_policy": {
            "native_original_dtype": "torch.bfloat16",
            "archive_storage_dtype": "float32",
            "candidate_scan_carry_dtype": "float32",
            "lossless_bf16_values_verified": True,
            "native_true_fp32_carry": False,
            "candidate_embedding_recomputed": False,
            "native_layer": 0,
        },
    }
    if args.native_logits:
        report["scope"] = (
            "counterfactual native logits substitution; not production output"
        )
    if args.native_softmax:
        report["scope"] = (
            "counterfactual native softmax substitution; not production output"
        )
    if args.warp_reduction:
        report["scope"] = (
            "computed warp softmax control; not runtime default; "
            + ("native logits substituted" if args.native_logits else "FoldJAX logits")
        )
    try:
        result, runtime = run_production(
            source, inputs, weights, out, warp_reduction=args.warp_reduction,
            computed_logits=args.computed_logits,
        )
        comparison, chunks = save_comparison(result, target, out)
        report.update(runtime=runtime, comparison=comparison, output_chunks=chunks)
        report["output_bindings_unchanged"] = (
            all(sha(out / chunk["file"]) == chunk["sha256"] for chunk in chunks)
            and sha(out / "compiled.hlo.txt") == runtime["compiled_hlo_sha256"]
        )
        if not report["output_bindings_unchanged"]:
            raise ValueError("saved production outputs changed before completion")
        if array_bindings != {
            "inputs": {name: array_identity(value) for name, value in inputs.items()},
            "weights": {name: array_identity(value) for name, value in weights.items()},
            "target": array_identity(target),
        }:
            raise ValueError("in-memory native arrays changed during production replay")
        report["capture_complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            report["bindings_unchanged"] = sources == source_binding(
                source
            ) and binding == {name: sha(path) for name, path in files.items()}
        except OSError as error:
            report["bindings_unchanged"] = False
            report["binding_error"] = f"{type(error).__name__}: {error}"
        report["passed"] = (
            report["capture_complete"]
            and report["bindings_unchanged"]
            and report["comparison"]["bitwise_equal"]
        )
        save_new(out / "report.json", report)
    if not report["bindings_unchanged"]:
        raise ValueError("bound source/reference changed during production replay")
    print(
        json.dumps(
            {key: report[key] for key in ("capture_complete", "passed", "comparison")}
        )
    )


if __name__ == "__main__":
    main()
