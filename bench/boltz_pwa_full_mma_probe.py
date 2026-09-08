"""Full native PWA contraction with the fixed residue-first k8 diagnostic.

Register fragments are packed on device from lossless BF16 words. This probe
does not select a production backend or infer a private cuBLAS CTA identity.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.boltz_pwa_mma_probe import (
    _flatten_inline_asm_results,
    bf16_round,
    bitwise_comparison,
    mma_assembly,
    save_array,
    save_new,
    sha,
)
from bench.boltz_pwa_mma_probe import source_binding as fixed_tile_source_binding

ROWS, TOKENS, CHANNELS, CHUNK_ROWS = 4436, 437, 32, 64
DEFAULT_ROW_SLICES = ((0, 1199), (1199, 2398), (2398, 3597), (3597, 4436))


def residue_k(position, xp):
    # CUTLASS v3.8.0 predicated_tile_access_iterator.h:187-220, 514-523.
    # https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h
    # Fixed schedule [0:21, zero*11, 21:437]; no search over K ordering.
    return xp.where(position < 21, position, xp.where(position < 32, -1, position - 11))


def fragment_indices(tile_m, tile_n, step, xp):
    lane = xp.arange(32, dtype=xp.int32)
    group, thread = lane // 4, lane % 4
    kk = tuple(residue_k(step * 8 + 2 * thread + half, xp) for half in (0, 1))
    mm = tuple(tile_m * 16 + group + register * 8 for register in (0, 1))
    nn = tile_n * 8 + group
    return mm, nn, kk


def schedule_proof():
    positions = residue_k(np.arange(448, dtype=np.int32), np)
    real = positions >= 0
    if not np.array_equal(positions[real], np.arange(437)) or not np.array_equal(
        np.flatnonzero(~real), np.arange(21, 32)
    ):
        raise ValueError("fixed K schedule did not retain every original term once")
    return {
        "original_terms_once_in_order": 437,
        "inserted_zero_positions": list(range(21, 32)),
        "mma_steps": 56,
    }


def full_mma_call(weights, values, initial):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    if (
        weights.shape != (TOKENS, TOKENS)
        or values.ndim != 3
        or values.shape[1:] != (TOKENS, CHANNELS)
        or values.shape[0] < 1
        or weights.dtype != jnp.uint16
        or values.dtype != jnp.uint16
        or initial.shape != (1,)
        or initial.dtype != jnp.float32
    ):
        raise ValueError("full MMA requires BF16 words [437,437] and [rows,437,32]")
    asm, constraints = mma_assembly(8)

    def kernel(w_ref, v_ref, c_ref, out_ref, lane_ref):
        tile_m, tile_n = pl.program_id(0), pl.program_id(1)

        def body(step, carry):
            mm, nn, kk = fragment_indices(tile_m, tile_n, step, jnp)
            registers = []
            for m in mm:
                halves = tuple(
                    pt.load(
                        v_ref.at[m // CHANNELS, jnp.maximum(k, 0), m % CHANNELS],
                        mask=k >= 0,
                        other=0,
                    ).astype(jnp.uint32)
                    for k in kk
                )
                registers.append(halves[0] | (halves[1] << jnp.uint32(16)))
            halves = tuple(
                pt.load(
                    w_ref.at[nn, jnp.maximum(k, 0)],
                    mask=(k >= 0) & (nn < TOKENS),
                    other=0,
                ).astype(jnp.uint32)
                for k in kk
            )
            registers.append(halves[0] | (halves[1] << jnp.uint32(16)))
            return tuple(
                pt.elementwise_inline_asm(
                    asm,
                    args=(*registers, *carry),
                    constraints=constraints,
                    pack=1,
                    result_shape_dtypes=[jax.ShapeDtypeStruct((32,), jnp.float32)] * 4,
                )
            )

        carry = (jnp.broadcast_to(c_ref[0], (32,)),) * 4
        carry = jax.lax.fori_loop(0, 56, body, carry)
        lane = jnp.arange(32, dtype=jnp.int32)
        for register, value in enumerate(carry):
            m = tile_m * 16 + lane // 4 + (register // 2) * 8
            n = tile_n * 8 + 2 * (lane % 4) + register % 2
            pt.store(
                out_ref.at[m // CHANNELS, n, m % CHANNELS],
                value,
                mask=n < TOKENS,
            )
        physical_lane = pt.elementwise_inline_asm(
            "mov.u32 $0, %laneid;",
            args=(lane,),
            constraints="=r,r",
            pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct((32,), jnp.int32)],
        )[0]
        pt.store(lane_ref.at[lane], physical_lane, mask=(tile_m == 0) & (tile_n == 0))

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(values.shape, jnp.float32),
            jax.ShapeDtypeStruct((32,), jnp.int32),
        ),
        grid=(values.shape[0] * 2, 55),
        compiler_params=pt.CompilerParams(num_warps=1, num_stages=1),
    )(weights, values, initial)


def bf16_words(value, chunk_rows=CHUNK_ROWS):
    """Convert already-BF16 FP32 storage in bounded host slices, without rounding."""
    if value.dtype != np.float32 or value.ndim < 1 or chunk_rows < 1:
        raise ValueError("expected FP32 storage and positive chunk rows")
    result = np.empty(value.shape, np.uint16)
    for start in range(0, value.shape[0], chunk_rows):
        source = value[start : start + chunk_rows]
        bits = source.view(np.uint32)
        if not np.isfinite(source).all() or (bits & 0xFFFF).any():
            raise ValueError("consumer input is not finite lossless BF16")
        np.right_shift(
            bits, 16, out=result[start : start + chunk_rows], casting="unsafe"
        )
    return result


def basis_inputs():
    row = np.arange(TOKENS)
    weights = ((row[:, None] + row[None, :]) % 3 - 1).astype(np.float32)
    weights[:16] = np.eye(TOKENS, dtype=np.float32)[:16]
    weights[-1] = 0
    weights[-1, -1] = 1
    values = (
        (np.arange(2)[:, None, None] + 2 * row[None, :, None] + np.arange(32)) % 5 - 2
    ).astype(np.float32)
    values[1, :, 16:] = np.eye(TOKENS, dtype=np.float32)[:, -16:]
    initial = np.array([3], np.float32)
    expected = initial[0] + np.einsum("ij,sjd->sid", weights, values)
    return weights, values, initial, expected


def reference_binding(root):
    return {
        name: sha(root / name)
        for name in (
            "report.json",
            "raw_fp32_control.private.json",
            "inputs.npz",
            "raw_fp32_output.npz",
        )
    }


def load_reference(root):
    binding = reference_binding(root)
    report = json.loads((root / "report.json").read_text())
    control = report.get("raw_fp32_control") or {}
    if (
        report.get("arm") != "native"
        or report.get("passed") is not True
        or report.get("head") != 2
        or control.get("capture_complete") is not True
        or control.get("bridge_gate_passed") is not True
        or report.get("input_shapes")
        != {
            "weights": [1, 1, TOKENS, TOKENS],
            "values": [1, ROWS, TOKENS, CHANNELS],
        }
    ):
        raise ValueError("requires the full native head2/K437/S4436 raw-FP32 bridge")
    expected_hashes = {
        "inputs.npz": report["inputs_sha256"],
        "raw_fp32_control.private.json": control["private_metadata_sha256"],
        "raw_fp32_output.npz": control["output_sha256"],
    }
    if any(binding[name] != digest for name, digest in expected_hashes.items()):
        raise ValueError("native report/artifact hash mismatch")
    private = json.loads((root / "raw_fp32_control.private.json").read_text())
    provenance = private["provenance"]
    if (
        provenance["inputs_sha256"] != report["inputs_sha256"]
        or provenance["selected_head"] != 2
        or provenance["source"] != report["source"]
        or private["output_sha256"] != control["output_sha256"]
        or private["baseline"]["comparison"]["bitwise_equal"] is not True
        or private.get("bridge_gate_passed") is not True
        or private.get("capture_complete") is not True
    ):
        raise ValueError("native raw-output baseline/provenance is unproven")
    with np.load(root / "inputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"weights", "values"}:
            raise ValueError("unknown native input schema")
        weights, values = archive["weights"], archive["values"]
    with np.load(root / "raw_fp32_output.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"raw_output", "rounded_output"}:
            raise ValueError("unknown native raw-output schema")
        target = archive["raw_output"]
    if (
        list(weights.shape) != report["input_shapes"]["weights"]
        or list(values.shape) != report["input_shapes"]["values"]
        or target.shape != values.shape
        or target.dtype != np.float32
    ):
        raise ValueError("native full array shapes/storage differ")
    for start in range(0, ROWS, CHUNK_ROWS):
        if not np.isfinite(target[0, start : start + CHUNK_ROWS]).all():
            raise ValueError("native raw-FP32 target is nonfinite")
    return bf16_words(weights[0, 0]), bf16_words(values[0]), target[0], binding


def compile_and_run(weights, values, initial, out, label):
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.triton import lowering, primitives

    traces = []
    original = lowering.lower_jaxpr_to_triton_module

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        traces.append(result.module.operation.get_asm(enable_debug_info=True))
        return result

    options = {
        "xla_dump_to": str(out / f"{label}.compiler"),
        "xla_dump_hlo_as_text": True,
        "xla_gpu_dump_llvmir": True,
        "xla_allow_excess_precision": False,
    }
    operands = tuple(jnp.asarray(x) for x in (weights, values, initial))
    primitive = primitives.elementwise_inline_asm_p
    rule = lowering.triton_lowering_rules[primitive]
    try:
        with (
            patch.object(lowering, "lower_jaxpr_to_triton_module", observe),
            patch.dict(
                lowering.triton_lowering_rules,
                {primitive: _flatten_inline_asm_results(rule)},
            ),
        ):
            executable = (
                jax.jit(full_mma_call, compiler_options=options)
                .lower(*operands)
                .compile()
            )
    finally:
        for index, text in enumerate(traces):
            with (out / f"{label}.{index}.triton.mlir").open("x") as stream:
                stream.write(text)
    if not traces or not all(mma_assembly(8)[0].split()[0] in text for text in traces):
        raise RuntimeError("actual compiler IR did not expose the fixed k8 instruction")
    with (out / f"{label}.hlo.txt").open("x") as stream:
        stream.write(executable.as_text())
    output, lanes = executable(*operands)
    output.block_until_ready()
    if not np.array_equal(jax.device_get(lanes), np.arange(32, dtype=np.int32)):
        raise ValueError("physical lane mapping differs")
    files = {
        str(path.relative_to(out)): sha(path)
        for path in out.rglob("*")
        if path.is_file() and path.relative_to(out).parts[0].startswith(f"{label}.")
    }
    return output, {
        "compiler_options": options,
        "actual_triton_ir_k8_verified": True,
        "probe_only_multi_result_ir_unwrap": True,
        "physical_lane_check_first_tile": True,
        "ptx_available": any(name.endswith(".ptx") for name in files),
        "sass_verified": False,
        "artifacts": files,
    }


def runtime_compile_and_run(weights, values, initial, out, label, *, cache=None):
    """Use the installed runtime primitive, without replacing any compiler rule."""
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives import native_pwa_mma as native

    operands = tuple(
        jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.bfloat16)
        for x in (weights, values)
    ) + (jnp.asarray(initial),)
    key = tuple((x.shape, str(x.dtype)) for x in operands)
    if cache is not None and key in cache:
        executable, evidence, first_label = cache[key]
        return _execute_runtime(
            executable, operands, {**evidence, "compilation_reused_from": first_label}
        )
    options = {
        "xla_dump_to": str(out / f"{label}.compiler"),
        "xla_dump_hlo_as_text": True,
        "xla_gpu_dump_llvmir": True,
        "xla_allow_excess_precision": False,
    }
    log = out / f"{label}.runtime-lowering.txt"
    with log.open("x") as stream, redirect_stdout(stream):
        executable = (
            jax.jit(partial(native._pwa_outputs, debug=True), compiler_options=options)
            .lower(*operands)
            .compile()
        )
    text = log.read_text()
    if (
        "The Triton module for pallas_call" not in text
        or "tt.elementwise_inline_asm" not in text
        or native._ASM not in text
    ):
        raise RuntimeError("runtime compiler did not expose actual carried-k8 IR")
    with (out / f"{label}.hlo.txt").open("x") as stream:
        stream.write(executable.as_text())
    files = {
        str(path.relative_to(out)): sha(path)
        for path in out.rglob("*")
        if path.is_file() and path.relative_to(out).parts[0].startswith(f"{label}.")
    }
    evidence = {
        "compiler_options": options,
        "actual_triton_ir_k8_verified": True,
        "probe_only_multi_result_ir_unwrap": False,
        "own_runtime_mma_primitive": True,
        "physical_lane_check_first_tile": True,
        "ptx_available": any(name.endswith(".ptx") for name in files),
        "sass_verified": False,
        "artifacts": files,
    }
    if cache is not None:
        cache[key] = executable, evidence, label
    return _execute_runtime(executable, operands, evidence)


def _execute_runtime(executable, operands, evidence):
    import jax
    import jax.numpy as jnp

    raw, rounded, lanes = jax.block_until_ready(executable(*operands))
    if not np.array_equal(jax.device_get(lanes), np.arange(32, dtype=np.int32)):
        raise ValueError("runtime physical lane mapping differs")
    if raw.dtype != jnp.float32 or rounded.dtype != jnp.bfloat16:
        raise TypeError("runtime output storage differs")
    return {"raw": raw, "rounded": rounded}, evidence


def empty_stats():
    return {
        "count": 0,
        "bitwise_unequal": 0,
        "nonfinite": 0,
        "sum_squares": 0.0,
        "max_abs": 0.0,
    }


def add_stats(stats, actual, target):
    if (
        actual.shape != target.shape
        or actual.dtype != np.float32
        or target.dtype != np.float32
    ):
        raise ValueError("comparison chunk shape/storage differs")
    stats["count"] += actual.size
    stats["bitwise_unequal"] += int(
        np.count_nonzero(actual.view(np.uint32) != target.view(np.uint32))
    )
    finite = np.isfinite(actual) & np.isfinite(target)
    stats["nonfinite"] += int(np.count_nonzero(~finite))
    difference = actual[finite].astype(np.float64) - target[finite].astype(np.float64)
    stats["sum_squares"] += float(np.sum(difference * difference))
    stats["max_abs"] = max(
        stats["max_abs"], float(np.max(np.abs(difference), initial=0))
    )


def finish_stats(stats):
    finite = stats["nonfinite"] == 0
    return {
        "count": stats["count"],
        "bitwise_unequal": stats["bitwise_unequal"],
        "nonfinite": stats["nonfinite"],
        "bitwise_equal": finite
        and stats["count"] > 0
        and stats["bitwise_unequal"] == 0,
        "rmse": math.sqrt(stats["sum_squares"] / stats["count"])
        if finite and stats["count"]
        else None,
        "max_abs": stats["max_abs"] if finite else None,
    }


def round_finite_output(value):
    """Round finite outputs; preserve failed nonfinite bits for diagnosis only."""
    finite = np.isfinite(value)
    if finite.all():
        return bf16_round(value)
    result = value.copy()
    result[finite] = bf16_round(value[finite])
    return result


def compare_stream(target, read_chunk, out, progress, *, read_rounded_chunk=None):
    raw, rounded = empty_stats(), empty_stats()
    bridge = empty_stats()
    progress["chunks"] = []
    for start in range(0, target.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, target.shape[0])
        actual = np.asarray(read_chunk(start, stop))
        expected = target[start:stop]
        actual_bf16 = round_finite_output(actual)
        if read_rounded_chunk is not None:
            device_bf16 = np.asarray(read_rounded_chunk(start, stop))
            add_stats(bridge, device_bf16, actual_bf16)
            actual_bf16 = device_bf16
        expected_bf16 = round_finite_output(expected)
        add_stats(raw, actual, expected)
        add_stats(rounded, actual_bf16, expected_bf16)
        path = out / f"rows-{start:04d}-{stop:04d}.npz"
        progress["chunks"].append(
            {
                "start": start,
                "stop": stop,
                "file": path.name,
                "sha256": save_array(
                    path, raw_output=actual, rounded_output=actual_bf16
                ),
            }
        )
    if raw["count"] != target.size:
        raise ValueError("stream comparison did not cover the full native output")
    progress.update({"raw_fp32": finish_stats(raw), "bf16": finish_stats(rounded)})
    if read_rounded_chunk is not None:
        progress["device_bf16_rounding_bridge"] = finish_stats(bridge)
    return {
        "raw_fp32": raw,
        "bf16": rounded,
        **(
            {"device_bf16_rounding_bridge": bridge}
            if read_rounded_chunk is not None
            else {}
        ),
    }


def array_identity(value):
    if not value.flags.c_contiguous:
        raise ValueError("row-slice artifacts require unchanged contiguous storage")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides_bytes": list(value.strides),
        "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
    }


def native_row_slice(weights, values, target, start, stop):
    if (
        (start, stop) not in DEFAULT_ROW_SLICES
        or weights.shape != (TOKENS, TOKENS)
        or values.shape != (ROWS, TOKENS, CHANNELS)
        or target.shape != values.shape
        or weights.dtype != np.uint16
        or values.dtype != np.uint16
        or target.dtype != np.float32
    ):
        raise ValueError("requires one frozen row slice of the full native operands")
    sliced_values, sliced_target = values[start:stop], target[start:stop]
    for full, view in ((values, sliced_values), (target, sliced_target)):
        if (
            not np.shares_memory(full, view)
            or view.strides != full.strides
            or view.ctypes.data - full.ctypes.data != start * full.strides[0]
        ):
            raise ValueError("row slicing changed operand storage or offsets")
    return (
        weights,
        sliced_values,
        sliced_target,
        {
            "start": start,
            "stop": stop,
            "weights": array_identity(weights),
            "values": array_identity(sliced_values),
            "target": array_identity(sliced_target),
            "values_byte_offset": start * values.strides[0],
            "target_byte_offset": start * target.strides[0],
            "shares_original_storage": True,
        },
    )


def _compare_runtime_device_output(target, output, out, progress):
    import jax

    readers = {}

    def read(array, start, stop):
        size = stop - start
        if size not in readers:
            readers[size] = jax.jit(
                lambda array, index: jax.lax.dynamic_slice_in_dim(
                    array, index, size, axis=0
                )
            )
        return np.asarray(
            jax.device_get(readers[size](array, np.int32(start))), np.float32
        )

    return compare_stream(
        target,
        partial(read, output["raw"]),
        out,
        progress,
        read_rounded_chunk=partial(read, output["rounded"]),
    )


def run_default_row_chunks(weights, values, target, out, progress):
    """Compare native full-output slices, never a newly shaped native GEMM."""
    cursor = 0
    for start, stop in DEFAULT_ROW_SLICES:
        if start != cursor or stop <= start:
            raise ValueError("frozen row slices have a gap, overlap or empty interval")
        cursor = stop
    if cursor != ROWS:
        raise ValueError("frozen row slices do not cover the native full MSA")
    keys = ("raw_fp32", "bf16", "device_bf16_rounding_bridge")
    totals = {key: empty_stats() for key in keys}
    cache = {}
    progress.update(
        row_slices=[],
        reference_mode="row slices of original full native output",
        native_subshape_replayed=False,
        production_dispatcher_widened=False,
    )
    for start, stop in DEFAULT_ROW_SLICES:
        w, v, expected, binding = native_row_slice(weights, values, target, start, stop)
        label = f"full.rows-{start:04d}-{stop:04d}"
        directory = out / f"rows-{start:04d}-{stop:04d}"
        directory.mkdir()
        entry = {"binding": binding, "artifact_directory": directory.name}
        progress["row_slices"].append(entry)
        output, evidence = runtime_compile_and_run(
            w, v, np.zeros(1, np.float32), out, label, cache=cache
        )
        entry["compiler"] = evidence
        stats = _compare_runtime_device_output(expected, output, directory, entry)
        if native_row_slice(weights, values, target, start, stop)[3] != binding:
            raise ValueError("native row operands changed during execution")
        entry["slice_binding_unchanged"] = True
        for key in keys:
            for field in ("count", "bitwise_unequal", "nonfinite", "sum_squares"):
                totals[key][field] += stats[key][field]
            totals[key]["max_abs"] = max(totals[key]["max_abs"], stats[key]["max_abs"])
        del output
    if any(stats["count"] != target.size for stats in totals.values()):
        raise ValueError("row comparison did not cover each native output once")
    progress.update({key: finish_stats(stats) for key, stats in totals.items()})


def source_binding(root):
    if (
        Path(inspect.getfile(source_binding)).resolve()
        != root / "bench/boltz_pwa_full_mma_probe.py"
    ):
        raise ValueError("imported another source snapshot")
    return {
        **fixed_tile_source_binding(root),
        "bench/boltz_pwa_full_mma_probe.py": sha(
            root / "bench/boltz_pwa_full_mma_probe.py"
        ),
    }


def runtime_source_binding(root):
    from foldjax.models.boltz2.models.primitives import native_pwa_mma
    from foldjax.models.boltz2.models.trunk_blocks import msa

    result = source_binding(root)
    for module, relative in (
        (
            native_pwa_mma,
            "src/foldjax/models/boltz2/models/primitives/native_pwa_mma.py",
        ),
        (msa, "src/foldjax/models/boltz2/models/trunk_blocks/msa.py"),
    ):
        path = root / relative
        if Path(inspect.getfile(module)).resolve() != path:
            raise ValueError("imported runtime from another source snapshot")
        result[relative] = sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--implementation", choices=("probe", "runtime"), default="probe"
    )
    parser.add_argument("--default-row-chunks", action="store_true")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.out.resolve().is_relative_to(args.source_root):
        parser.error("private artifacts must be outside the source snapshot")
    runtime = args.implementation == "runtime"
    if args.default_row_chunks and not runtime:
        parser.error("--default-row-chunks requires --implementation runtime")
    bind_source = runtime_source_binding if runtime else source_binding
    run_compiled = runtime_compile_and_run if runtime else compile_and_run
    source = bind_source(args.source_root)
    schedule = schedule_proof()
    weights, values, target, reference = load_reference(args.reference)
    import jax
    import jaxlib
    from jax._src.pallas.triton import lowering, primitives

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("requires exactly one GPU, without a fallback")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {
        "implementation": args.implementation,
        "source": source,
        "reference": reference,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "compiler_source": {
            module.__name__: sha(Path(inspect.getfile(module)))
            for module in (lowering, primitives)
        },
        "full_shape": [1, ROWS, TOKENS, CHANNELS],
        "schedule": "k8 prefix21/zero11/remainder416, 56 carried-C MMA calls",
        "schedule_proof": schedule,
        "grid": [ROWS * 2, 55],
        "storage": {
            "input_logical_dtype": "bfloat16",
            "input_device_storage": "uint16, lossless BF16 words",
            "output_device_storage": "float32",
            "rounded_archive_storage": "float32, lossless BF16 values",
            "rounded_archive_nonfinite": (
                "original failed FP32 bits retained, not a BF16 cast claim"
            ),
            "chunk_rows": CHUNK_ROWS,
            "mandatory_gpu_array_bytes": weights.nbytes + values.nbytes + target.nbytes,
            "workspace_measured": False,
        },
        "native_private_cta_identity_proven": False,
        "not_model_parity_admission": True,
        "capture_complete": False,
        "passed": False,
    }
    if runtime:
        report["storage"]["mandatory_gpu_array_bytes"] += target.size * 2
        report["storage"]["rounded_archive_nonfinite"] = "device BF16 values retained"
        report["remaining_boundary"] = (
            "actual auto-row-chunks retain existing einsum; only full S4436 is selected"
        )
    if args.default_row_chunks:
        report["row_chunk_control"] = True
        report["grid"] = [
            [(stop - start) * 2, 55] for start, stop in DEFAULT_ROW_SLICES
        ]
        report["storage"]["mandatory_gpu_array_bytes"] = (
            weights.nbytes
            + max(stop - start for start, stop in DEFAULT_ROW_SLICES)
            * TOKENS
            * CHANNELS
            * 8
        )
    try:
        bw, bv, initial, expected = basis_inputs()
        basis, evidence = run_compiled(
            bf16_words(bw), bf16_words(bv), initial, args.out, "basis"
        )
        rounded_basis = basis["rounded"] if runtime else None
        basis = basis["raw"] if runtime else basis
        report["basis"] = {
            "comparison": bitwise_comparison(
                np.asarray(jax.device_get(basis)), expected
            ),
            "compiler": evidence,
        }
        if not report["basis"]["comparison"]["bitwise_equal"]:
            raise ValueError("same-kernel integer/one-hot/tail/nonzero-C basis failed")
        if runtime:
            report["basis"]["device_bf16"] = bitwise_comparison(
                np.asarray(jax.device_get(rounded_basis), dtype=np.float32),
                bf16_round(expected),
            )
            if not report["basis"]["device_bf16"]["bitwise_equal"]:
                raise ValueError("runtime basis BF16 output differs")
        del basis
        if args.default_row_chunks:
            report["full"] = {}
            run_default_row_chunks(weights, values, target, args.out, report["full"])
        else:
            output, evidence = run_compiled(
                weights, values, np.zeros(1, np.float32), args.out, "full"
            )
            rounded_output = output["rounded"] if runtime else None
            output = output["raw"] if runtime else output
            report["full"] = {"compiler": evidence}
            readers = {}

            def read_chunk(start, stop, *, array=output):
                size = stop - start
                if size not in readers:
                    readers[size] = jax.jit(
                        lambda array, index: jax.lax.dynamic_slice_in_dim(
                            array, index, size, axis=0
                        )
                    )
                return np.asarray(
                    jax.device_get(readers[size](array, np.int32(start))),
                    dtype=np.float32,
                )

            compare_stream(
                target,
                read_chunk,
                args.out,
                report["full"],
                read_rounded_chunk=partial(read_chunk, array=rounded_output)
                if runtime
                else None,
            )
        report["capture_complete"] = True
        report["passed"] = all(
            report["full"][key]["bitwise_equal"] for key in ("raw_fp32", "bf16")
        )
        if runtime:
            report["passed"] &= report["full"]["device_bf16_rounding_bridge"][
                "bitwise_equal"
            ]
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error)[:2000],
            "fallback_used": False,
        }
        report["failed_compiler_artifacts"] = {
            str(path.relative_to(args.out)): sha(path)
            for path in args.out.rglob("*")
            if path.is_file()
            and path.relative_to(args.out).parts[0].startswith(("basis.", "full."))
        }
    try:
        if (
            bind_source(args.source_root) != source
            or reference_binding(args.reference) != reference
            or {
                module.__name__: sha(Path(inspect.getfile(module)))
                for module in (lowering, primitives)
            }
            != report["compiler_source"]
        ):
            raise ValueError("source or native reference changed during execution")
        report["bindings_unchanged"] = True
    except Exception as error:
        report["bindings_unchanged"] = False
        report["passed"] = False
        report["binding_error"] = str(error)[:2000]
    save_new(args.out / "report.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("capture_complete", "passed", "bindings_unchanged")
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
