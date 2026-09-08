"""One-warp BF16 MMA grouping diagnostic; never a production fallback.

Four deterministic tiles retain every native K operand. CPU register packing
is lossless; each GPU instruction consumes the previous FP32 accumulator as C.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.boltz_pwa_averaging_probe import (
    bf16_round,
    bitwise_comparison,
    save_array,
    save_new,
    sha,
    verify_bound_file,
)
from bench.boltz_pwa_averaging_probe import source_binding as averaging_source_binding

TILE_COUNT = 4
# Opcode/register counts: CUTLASS v3.8.0 arch/mma_sm80.h, lines 75-129, 337-397.
# https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cutlass/arch/mma_sm80.h
# Lane mapping: PTX ISA 9.3, Matrix Fragments for mma.m16n8k8/mma.m16n8k16.
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688


def fragment_coordinates(k_width):
    if k_width not in (8, 16):
        raise ValueError("only the paired k8/k16 instructions are in scope")
    group, thread = np.arange(32) // 4, np.arange(32) % 4
    a = [
        (group + ((i % 4) // 2) * 8, 2 * thread + i % 2 + (i // 4) * 8)
        for i in range(k_width // 2)
    ]
    b = [(2 * thread + i % 2 + (i // 2) * 8, group) for i in range(k_width // 4)]
    c = [(group + (i // 2) * 8, 2 * thread + i % 2) for i in range(4)]
    return a, b, c


def pack_operands(a, b, k_width):
    """Map BF16 words to per-lane PTX registers without changing their values."""
    ac, bc, _ = fragment_coordinates(k_width)
    if (
        a.ndim != 3
        or b.ndim != 3
        or a.shape[0] != b.shape[0]
        or a.shape[1] != 16
        or b.shape[2] != 8
        or a.shape[2] != b.shape[1]
        or a.shape[2] < 1
    ):
        raise ValueError("expected [tiles,16,K] and [tiles,K,8]")
    for value in (a, b):
        if (
            value.dtype != np.float32
            or not np.isfinite(value).all()
            or not np.array_equal(bf16_round(value), value)
        ):
            raise ValueError("operands must be finite, lossless BF16 in FP32 storage")
    steps = (a.shape[2] + k_width - 1) // k_width
    a = np.pad(a, ((0, 0), (0, 0), (0, steps * k_width - a.shape[2])))
    b = np.pad(b, ((0, 0), (0, steps * k_width - b.shape[1]), (0, 0)))
    abits, bbits = a.view(np.uint32) >> 16, b.view(np.uint32) >> 16
    registers = []
    for step in range(steps):
        af = [abits[:, row, col + step * k_width] for row, col in ac]
        bf = [bbits[:, row + step * k_width, col] for row, col in bc]
        registers.append(
            np.stack(
                [
                    fragment[i] | (fragment[i + 1] << 16)
                    for fragment in (af, bf)
                    for i in range(0, len(fragment), 2)
                ],
                axis=1,
            )
        )
    return np.stack(registers, axis=1)


def pack_accumulator(value):
    if value.ndim != 3 or value.shape[1:] != (16, 8) or value.dtype != np.float32:
        raise ValueError("accumulator must be FP32 [tiles,16,8]")
    return np.stack([value[:, r, c] for r, c in fragment_coordinates(8)[2]], axis=1)


def unpack_accumulator(value):
    if value.ndim != 3 or value.shape[1:] != (4, 32) or value.dtype != np.float32:
        raise ValueError("accumulator register shape/dtype mismatch")
    result = np.empty((value.shape[0], 16, 8), np.float32)
    for i, (r, c) in enumerate(fragment_coordinates(8)[2]):
        result[:, r, c] = value[:, i]
    return result


def mma_assembly(k_width):
    fragment_coordinates(k_width)
    na, nb = k_width // 4, k_width // 8

    def registers(start, count):
        return "{" + ",".join(f"${i}" for i in range(start, start + count)) + "}"

    opcode = f"mma.sync.aligned.m16n8k{k_width}.row.col.f32.bf16.bf16.f32"
    return (
        f"{opcode} {registers(0, 4)}, {registers(4, na)}, "
        f"{registers(4 + na, nb)}, {registers(4 + na + nb, 4)};",
        ",".join(["=f"] * 4 + ["r"] * (na + nb) + ["f"] * 4),
    )


def mma_call(packed, initial, *, k_width):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    asm, constraints = mma_assembly(k_width)
    tiles, steps, registers, lanes = packed.shape
    if (
        registers != 3 * k_width // 8
        or lanes != 32
        or packed.dtype != jnp.uint32
        or initial.shape != (tiles, 4, 32)
        or initial.dtype != jnp.float32
    ):
        raise ValueError("packed MMA operand schema differs")

    def kernel(p_ref, c_ref, out_ref, lane_ref):
        def body(step, carry):
            operands = tuple(p_ref[0, step, i, :] for i in range(registers))
            return tuple(
                pt.elementwise_inline_asm(
                    asm,
                    args=(*operands, *carry),
                    constraints=constraints,
                    pack=1,
                    result_shape_dtypes=[jax.ShapeDtypeStruct((32,), jnp.float32)] * 4,
                )
            )

        carry = tuple(c_ref[0, i, :] for i in range(4))
        carry = jax.lax.fori_loop(0, steps, body, carry)
        for i, value in enumerate(carry):
            out_ref[0, i, :] = value
        lane_ref[0, :] = pt.elementwise_inline_asm(
            "mov.u32 $0, %laneid;",
            args=(p_ref[0, 0, 0, :],),
            constraints="=r,r",
            pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct((32,), jnp.int32)],
        )[0]

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((tiles, 4, 32), jnp.float32),
            jax.ShapeDtypeStruct((tiles, 32), jnp.int32),
        ),
        grid=(tiles,),
        in_specs=(
            pl.BlockSpec((1, steps, registers, 32), lambda i: (i, 0, 0, 0)),
            pl.BlockSpec((1, 4, 32), lambda i: (i, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec((1, 4, 32), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, 32), lambda i: (i, 0)),
        ),
        compiler_params=pt.CompilerParams(num_warps=1, num_stages=1),
    )(packed, initial)


def basis_operands(k=437):
    a = np.zeros((3, 16, k), np.float32)
    b = np.zeros((3, k, 8), np.float32)
    a[0] = np.arange(16)[:, None] % 3 - 1
    b[0] = np.arange(k)[:, None] % 3 - 1 + np.arange(8)[None, :] % 2
    a[1, np.arange(16), np.arange(16) % k] = 1
    b[1] = np.arange(k)[:, None] % 7 + np.arange(8)[None, :]
    a[2, :, -1], b[2, -1, :] = np.arange(16) + 1, np.arange(8) + 1
    initial = (np.arange(3 * 16 * 8).reshape(3, 16, 8) % 5).astype(np.float32)
    return a, b, initial, initial + a @ b


def probe_arms(k8_residue_controls=False):
    arms = [("k8", 8, None), ("k16", 16, None)]
    if k8_residue_controls:
        arms += [
            ("k8_zero_tail_448", 8, "zero_tail_448"),
            ("k8_residue_first_32", 8, "residue_first_32"),
        ]
    return arms


def k8_control_operands(a, b, layout):
    """Insert only zeros; retain all 437 original K terms once, in order.

    CUTLASS v3.8.0 predicated_tile_access_iterator.h:187-220, 471-475,
    514-523 starts at K=0 with residue K%CTA_K, then advances by the residue.
    https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h
    With CTA_K=32 this yields prefix21/zero11/remainder416. The paired
    zero-tail control also executes 56 k8 MMAs, isolating padding placement.
    This public iterator is not proof of the private native cuBLAS CTA layout.
    """
    if (
        a.ndim != 3
        or a.shape[1:] != (16, 437)
        or b.shape != (a.shape[0], 437, 8)
        or a.dtype != np.float32
        or b.dtype != np.float32
    ):
        raise ValueError("k8 controls require original FP32-stored K=437 tiles")
    if layout == "zero_tail_448":
        positions = np.concatenate((np.arange(437), np.full(11, -1)))
    elif layout == "residue_first_32":
        positions = np.concatenate((np.arange(21), np.full(11, -1), np.arange(21, 437)))
    else:
        raise ValueError("only the two frozen k8 padding controls are supported")
    positions = positions.astype(np.int32)
    real = positions >= 0
    if (
        positions.shape != (448,)
        or not np.array_equal(positions[real], np.arange(437))
        or np.count_nonzero(~real) != 11
    ):
        raise ValueError("K layout must retain every original term exactly once")
    aa = np.take(a, np.maximum(positions, 0), axis=2)
    bb = np.take(b, np.maximum(positions, 0), axis=1)
    aa[:, :, ~real], bb[:, ~real, :] = np.float32(0), np.float32(0)
    if (
        not np.array_equal(aa[:, :, real].view(np.uint32), a.view(np.uint32))
        or not np.array_equal(bb[:, real, :].view(np.uint32), b.view(np.uint32))
        or aa[:, :, ~real].view(np.uint32).any()
        or bb[:, ~real, :].view(np.uint32).any()
    ):
        raise ValueError("K control altered original operand bits or padding")
    return aa, bb, positions


def select_tiles(raw, candidate, count=TILE_COUNT):
    """First distinct tiles in native [MSA,token,channel] mismatch order."""
    if raw.shape != candidate.shape or raw.ndim != 4 or raw.shape[0] != 1:
        raise ValueError("native and JAX output shapes differ")
    rows, n, channels = raw.shape[1:]
    if channels != 32:
        raise ValueError("released head has 32 channels")
    if count < 1:
        raise ValueError("tile count must be positive")
    selected = []
    for row in range(rows):
        for index in np.flatnonzero(bf16_round(raw[0, row]) != candidate[0, row]):
            token, channel = divmod(int(index), channels)
            tile = ((row * channels + channel) // 16 * 16, token // 8 * 8)
            if tile not in selected:
                selected.append(tile)
                if len(selected) == count:
                    return selected
    raise ValueError("not enough distinct mismatch tiles for the frozen selection")


def extract_tiles(weights, values, raw, candidate, selected):
    n, channels = values.shape[2:]
    aa, bb, rr, jj, masks = [], [], [], [], []
    for m0, n0 in selected:
        ms, ns = m0 + np.arange(16), n0 + np.arange(8)
        rows, ds = ms // channels, ms % channels
        valid = np.broadcast_to(ns[None, :] < n, (16, 8))
        clipped = np.minimum(ns, n - 1)
        aa.append(values[0, rows, :, ds])
        bb.append(np.where(ns[None, :] < n, weights[0, 0, clipped, :].T, 0))
        rr.append(
            np.where(valid, raw[0, rows[:, None], clipped[None, :], ds[:, None]], 0)
        )
        jj.append(
            np.where(
                valid, candidate[0, rows[:, None], clipped[None, :], ds[:, None]], 0
            )
        )
        masks.append(valid)
    return tuple(np.stack(x) for x in (aa, bb, rr, jj, masks))


def load_tiles(reference, jax_reference):
    report = json.loads((reference / "report.json").read_text())
    control = report.get("raw_fp32_control") or {}
    if (
        report.get("arm") != "native"
        or report.get("passed") is not True
        or control.get("bridge_gate_passed") is not True
        or control.get("capture_complete") is not True
        or report.get("head") != 2
    ):
        raise ValueError("full native raw-FP32 bridge must pass first")
    private = reference / "raw_fp32_control.private.json"
    verify_bound_file(private, control["private_metadata_sha256"])
    metadata = json.loads(private.read_text())
    if (
        metadata["provenance"]["inputs_sha256"] != report["inputs_sha256"]
        or metadata["provenance"]["selected_head"] != report["head"]
        or metadata["provenance"]["source"] != report["source"]
        or metadata["output_sha256"] != control["output_sha256"]
        or metadata["baseline"]["comparison"]["bitwise_equal"] is not True
        or metadata.get("bridge_gate_passed") is not True
        or metadata.get("capture_complete") is not True
    ):
        raise ValueError("native raw-FP32 provenance differs")
    other = json.loads((jax_reference / "report.json").read_text())
    if (
        other.get("arm") != "foldjax"
        or other.get("capture_complete") is not True
        or other.get("inputs_sha256") != report["inputs_sha256"]
        or other.get("head") != report["head"]
    ):
        raise ValueError("JAX reference must share the exact native input archive")
    arrays = []
    for path, digest, schema, names in (
        (
            reference / "inputs.npz",
            report["inputs_sha256"],
            {"weights", "values"},
            ("weights", "values"),
        ),
        (
            reference / "raw_fp32_output.npz",
            control["output_sha256"],
            {"raw_output", "rounded_output"},
            ("raw_output",),
        ),
        (
            jax_reference / "einsum.npz",
            other["arms"]["einsum"]["output_sha256"],
            {"output"},
            ("output",),
        ),
    ):
        verify_bound_file(path, digest)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != schema:
                raise ValueError("unreviewed capture schema")
            arrays.extend(archive[name] for name in names)
    weights, values, raw, candidate = arrays
    if (
        any(x.dtype != np.float32 or not np.isfinite(x).all() for x in arrays)
        or values.ndim != 4
        or values.shape != raw.shape
        or weights.shape != (1, 1, 437, 437)
        or report["input_shapes"]
        != {"weights": list(weights.shape), "values": list(values.shape)}
        or values.shape[2] != 437
    ):
        raise ValueError("full K=437 native arrays or metadata differ")
    selected = select_tiles(raw, candidate)
    return extract_tiles(*arrays, selected), {
        "reference_sha256": sha(reference / "report.json"),
        "jax_reference_sha256": sha(jax_reference / "report.json"),
        "inputs_sha256": report["inputs_sha256"],
        "raw_output_sha256": control["output_sha256"],
        "selection": (
            "first four distinct tiles in native MSA/token/channel mismatch order"
        ),
        "tiles_native_transposed_gemm_mn": selected,
        "orientation": "values[MSA*channel,K] @ weights[token,K].T",
        "full_k": 437,
        "not_new_native_submatrix_reference": True,
    }


def _flatten_inline_asm_results(rule):
    """Unwrap JAX 0.11.1's nested multi-result IR list for this probe only.

    The installed lowering wraps Triton's OpResultList in another list. Single
    results already have the expected representation and must remain untouched.
    No arithmetic, instruction or installed dependency is changed.
    """

    def wrapped(ctx, *args, **kwargs):
        result = rule(ctx, *args, **kwargs)
        if len(ctx.avals_out) > 1 and len(result) == 1:
            result = list(result[0])
        if len(result) != len(ctx.avals_out):
            raise ValueError("inline MMA lowering result arity differs")
        return result

    return wrapped


def run_compiled(packed, initial, k_width, out, label):
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.triton import lowering, primitives

    traces = []
    original = lowering.lower_jaxpr_to_triton_module

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        traces.append(result.module.operation.get_asm(enable_debug_info=True))
        return result

    dump = out / f"{label}.compiler"
    options = {
        "xla_dump_to": str(dump),
        "xla_dump_hlo_as_text": True,
        "xla_gpu_dump_llvmir": True,
        "xla_allow_excess_precision": False,
    }

    def function(p, c):
        return mma_call(p, c, k_width=k_width)

    operands = jnp.asarray(packed), jnp.asarray(pack_accumulator(initial))
    try:
        primitive = primitives.elementwise_inline_asm_p
        rule = lowering.triton_lowering_rules[primitive]
        with (
            patch.object(lowering, "lower_jaxpr_to_triton_module", observe),
            patch.dict(
                lowering.triton_lowering_rules,
                {primitive: _flatten_inline_asm_results(rule)},
            ),
        ):
            executable = (
                jax.jit(function, compiler_options=options).lower(*operands).compile()
            )
    finally:
        # Preserve actual lowering evidence even if backend compilation rejects it.
        for i, value in enumerate(traces):
            with (out / f"{label}.{i}.triton.mlir").open("x") as stream:
                stream.write(value)
    if not traces:
        raise RuntimeError("compiler did not expose a Triton lowering; no fallback")
    with (out / f"{label}.hlo.txt").open("x") as stream:
        stream.write(executable.as_text())
    result, lanes = jax.device_get(executable(*operands))
    if not np.array_equal(lanes, np.broadcast_to(np.arange(32), lanes.shape)):
        raise ValueError("physical lane mapping differs from packed fragment mapping")
    result = unpack_accumulator(result)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite MMA output")
    opcode = mma_assembly(k_width)[0].split()[0]
    if not all(opcode in text for text in traces):
        raise RuntimeError("emitted Triton IR lacks the requested MMA opcode")
    files = [p for p in out.rglob("*") if p.is_file() and p.name.startswith(label)]
    files += [p for p in dump.rglob("*") if p.is_file()]
    files = sorted(set(files))
    ptx = [p for p in files if p.suffix == ".ptx"]
    return result, {
        "probe_only_multi_result_ir_unwrap": True,
        "compiler_options": options,
        "triton_ir_opcode_verified": True,
        "ptx_available": bool(ptx),
        "ptx_mma_opcodes": sorted(
            set(
                op
                for p in ptx
                for op in re.findall(
                    r"mma\.sync\.aligned\.m16n8k\d+\.row\.col\.f32\.bf16\.bf16\.f32",
                    p.read_text(),
                )
            )
        ),
        "sass_verified": False,
        "lane_mapping_equal": True,
        "artifacts": {str(p.relative_to(out)): sha(p) for p in files},
    }


def source_binding(root):
    if (
        Path(inspect.getfile(source_binding)).resolve()
        != root / "bench/boltz_pwa_mma_probe.py"
    ):
        raise ValueError("imported another source snapshot")
    return {
        **averaging_source_binding(root),
        "bench/boltz_pwa_mma_probe.py": sha(root / "bench/boltz_pwa_mma_probe.py"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reference", "jax-reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--k8-residue-controls",
        action="store_true",
        help="add only k8 zero-tail448 and residue-prefix21/zero11 controls",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.source_root = args.source_root.resolve()
    if args.out.resolve().is_relative_to(args.source_root):
        parser.error("private compiler artifacts must be outside the source snapshot")
    source = source_binding(args.source_root)
    (a, b, target, previous, valid), provenance = load_tiles(
        args.reference, args.jax_reference
    )
    import jax
    import jaxlib
    from jax._src.pallas.triton import lowering, primitives

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("probe requires one GPU; no CPU/alternative-kernel fallback")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {
        "source": source,
        "provenance": provenance,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "compiler_source": {
            module.__name__: sha(Path(inspect.getfile(module)))
            for module in (lowering, primitives)
        },
        "not_model_parity_admission": True,
        "native_instruction_identity_proven": False,
        "native_private_cta_identity_proven": False,
        "k8_residue_controls": args.k8_residue_controls,
        "comparison_scope": "four selected 16x8 tiles, original full-native K=437",
        "arms": {},
    }
    for label, k_width, layout in probe_arms(args.k8_residue_controls):
        arm = {}
        report["arms"][label] = arm
        try:
            ba, bb, initial, expected = basis_operands()
            aa, ab = a, b
            if layout is not None:
                ba, bb, basis_positions = k8_control_operands(ba, bb, layout)
                aa, ab, positions = k8_control_operands(a, b, layout)
                if not np.array_equal(basis_positions, positions):
                    raise ValueError("basis and actual K operand layouts differ")
                arm["k_operand_layout"] = {
                    "layout": layout,
                    "original_terms": 437,
                    "executed_k": 448,
                    "mma_calls": 56,
                    "zero_terms": 11,
                    "original_terms_once_in_order": True,
                    "operands_preserved_bitwise": True,
                    "same_basis_and_actual_layout": True,
                    "native_private_cta_identity_proven": False,
                    "positions_sha256": save_array(
                        args.out / f"{label}-k-positions.npz", positions=positions
                    ),
                }
            basis, evidence = run_compiled(
                pack_operands(ba, bb, k_width),
                initial,
                k_width,
                args.out,
                f"{label}-basis",
            )
            arm["basis"] = {
                "comparison": bitwise_comparison(basis, expected),
                "compiler": evidence,
            }
            if not arm["basis"]["comparison"]["bitwise_equal"]:
                raise ValueError("basis/tail/carried-C oracle failed")
            packed = pack_operands(aa, ab, k_width)
            arm["packed_sha256"] = save_array(
                args.out / f"{label}-packed.npz", packed=packed
            )
            output, evidence = run_compiled(
                packed,
                np.zeros_like(target),
                k_width,
                args.out,
                f"{label}-native-tiles",
            )
            arm.update(
                {
                    "status": "complete",
                    "compiler": evidence,
                    "raw_fp32": bitwise_comparison(output[valid], target[valid]),
                    "native_bf16": bitwise_comparison(
                        bf16_round(output)[valid], bf16_round(target)[valid]
                    ),
                    "previous_jax_bf16": bitwise_comparison(
                        bf16_round(output)[valid], previous[valid]
                    ),
                    "output_sha256": save_array(
                        args.out / f"{label}-output.npz",
                        output=output,
                        target=target,
                        previous=previous,
                        valid=valid,
                    ),
                }
            )
        except Exception as error:
            arm.update(
                {
                    "status": "unsupported"
                    if isinstance(error, NotImplementedError)
                    else "failed",
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                    "fallback_used": False,
                    "artifacts": {
                        str(path.relative_to(args.out)): sha(path)
                        for path in args.out.rglob("*")
                        if path.is_file()
                        and path.relative_to(args.out).parts[0].startswith(f"{label}-")
                    },
                }
            )
    if source_binding(args.source_root) != source:
        raise ValueError("benchmark source changed during run")
    report["capture_complete"] = all(
        arm["status"] == "complete" for arm in report["arms"].values()
    )
    save_new(args.out / "report.json", report)
    print(
        json.dumps(
            {
                "capture_complete": report["capture_complete"],
                "arms": {k: v["status"] for k, v in report["arms"].items()},
            }
        )
    )
    if not report["capture_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
