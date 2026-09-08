"""Pinned native triangle attention versus isolated JAX core, operator-only.

The frozen synthetic panel has 17 finite cases and four separate -inf-mask
semantic controls. Full pair437 repeats two row activations; it is a physical
shape control, not a real trunk tensor or a performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import digest, save_new
from bench.openbind_tape_adapter import UPSTREAM_COMMIT
from bench.openbind_triton_linear_probe import LinearKernelCapture
from bench.openbind_triton_norm_probe import (
    metrics,
    save_arrays,
    write_compiler_evidence,
)

PANEL = "openbind-native-triton-attention-v1"
NATIVE_OPERATOR = "openfold3/core/kernels/triton/evoformer.py"
CANDIDATE_OPERATOR = "src/foldjax/models/openfold3/models/native_triton_attention.py"
CANDIDATE_HELPER = "src/foldjax/models/openfold3/models/native_triton_linear.py"
POLICY = {
    "dtype": "float32",
    "float32_matmul_precision": "high",
    "use_exp2": False,
    "dynamic": False,
    "has_pair_bias": True,
    "block_q": 64,
    "block_kv": 16,
    "block_dim": 32,
    "num_warps": 4,
    "num_stages": 1,
}
KEYS = ("query", "key", "value", "additive_mask", "pair_bias")


@dataclass(frozen=True)
class Case:
    name: str
    length: int
    dim: int
    rows: int = 2
    profile: str = "normal"

    @property
    def finite(self):
        return self.profile in {"normal", "finite_masked"}

    @property
    def shape(self):
        return (1, self.rows, self.length, 4, self.dim)

    def identity(self):
        return {**asdict(self), "shape": list(self.shape), "finite_gate": self.finite}


def case_specs():
    cases = [
        Case(f"d{d}-n{n}-normal", n, d)
        for d in (16, 32)
        for n in (17, 31, 32, 33, 63, 64, 65)
    ]
    cases += [
        Case(f"d{d}-n33-{profile}", 33, d, profile=profile)
        for profile in ("finite_masked", "full_inf", "initial_inf")
        for d in (16, 32)
    ]
    return cases + [Case("d32-pair437-normal", 437, 32, rows=437)]


def operand_shapes(case):
    if (
        case.length <= 16
        or case.dim not in (16, 32)
        or case.rows < 1
        or case.profile not in {"normal", "finite_masked", "full_inf", "initial_inf"}
    ):
        raise ValueError("invalid frozen attention case")
    return {
        "query": case.shape,
        "key": case.shape,
        "value": case.shape,
        "additive_mask": (1, case.rows, 1, 1, case.length),
        "pair_bias": (1, 1, 4, case.length, case.length),
    }


def mask_values(case):
    mask = np.zeros((1, case.rows, 1, 1, case.length), np.float32)
    if case.profile == "normal":
        # The two distinct row masks prevent an accidental row-axis broadcast.
        for row in range(case.rows):
            mask[0, row, 0, 0, (np.arange(case.length) + row % 2) % 5 == 2] = -1e9
    elif case.profile == "finite_masked":
        mask.fill(-1e9)
    elif case.profile == "full_inf":
        mask.fill(-np.inf)
    elif case.profile == "initial_inf":
        mask[..., :16] = -np.inf
    return mask


def operands(case):
    shapes = operand_shapes(case)
    rng = np.random.default_rng(10471 + case.length + case.dim)
    arrays = {}
    for key in KEYS[:3]:
        distinct = rng.normal(size=(1, min(case.rows, 2), case.length, 4, case.dim))
        arrays[key] = (
            distinct[:, np.arange(case.rows) % min(case.rows, 2)] * 0.4
        ).astype(np.float32)
    arrays["additive_mask"] = mask_values(case)
    arrays["pair_bias"] = (rng.normal(size=shapes["pair_bias"]) * 0.7).astype(
        np.float32
    )
    return arrays


def validate_operands(arrays, case):
    shapes = operand_shapes(case)
    if set(arrays) != set(shapes):
        raise ValueError("reference operand keys differ")
    for key, value in arrays.items():
        if value.shape != shapes[key] or value.dtype != np.float32:
            raise ValueError(f"reference operand shape/dtype differs: {key}")
        if key != "additive_mask" and not np.isfinite(value).all():
            raise ValueError(f"unexpected nonfinite operand: {key}")
    if not np.array_equal(arrays["additive_mask"], mask_values(case)):
        raise ValueError("additive mask differs from frozen semantic profile")


def compare_outputs(left, right, case):
    if any(x.shape != case.shape or x.dtype != np.float32 for x in (left, right)):
        raise ValueError("attention output shape/dtype differs")
    both_finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    result = (
        metrics(left, right)
        if both_finite
        else {
            "bitwise_equal": left.tobytes() == right.tobytes(),
            "array_equal": bool(np.array_equal(left, right)),
            "strict_1e4_allclose": False,
            "max_absolute_error": None,
            "rmse": None,
            "unequal_values": int(np.count_nonzero(left != right)),
        }
    )
    result.update(
        finite_gate_eligible=case.finite,
        both_finite=both_finite,
        semantic_equal=bool(np.array_equal(left, right, equal_nan=True)),
    )
    result["nonfinite_pattern_equal"] = all(
        np.array_equal(fn(left), fn(right))
        for fn in (np.isnan, np.isposinf, np.isneginf)
    )
    result["nonfinite_counts"] = [
        {
            "nan": int(np.isnan(x).sum()),
            "positive_inf": int(np.isposinf(x).sum()),
            "negative_inf": int(np.isneginf(x).sum()),
        }
        for x in (left, right)
    ]
    if not case.finite:
        result["strict_1e4_allclose"] = None
    return result


def helper_identities():
    root = Path(__file__).resolve().parent
    return {
        name: digest(root / name)
        for name in (
            "openbind_triton_attention_probe.py",
            "openbind_triton_linear_probe.py",
            "openbind_triton_norm_probe.py",
            "boltz_historical_replay.py",
            "openbind_tape_adapter.py",
        )
    }


def source_identity(root, native):
    relative = NATIVE_OPERATOR if native else CANDIDATE_OPERATOR
    commit = None
    if native:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--"])
        untracked = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "openfold3",
            ],
            text=True,
        )
        if (
            commit != UPSTREAM_COMMIT
            or dirty
            or any(p.endswith(".py") for p in untracked.splitlines())
        ):
            raise ValueError("native source must be clean pinned OpenBind")
    paths = sorted((root / ("openfold3" if native else "src")).rglob("*.py"))
    if not paths or any(not p.resolve().is_relative_to(root) for p in paths):
        raise ValueError("source is empty or escapes selected root")
    aggregate = hashlib.sha256()
    for path in paths:
        aggregate.update(str(path.relative_to(root)).encode())
        aggregate.update(bytes.fromhex(digest(path)))
    dependencies = {} if native else {CANDIDATE_HELPER: digest(root / CANDIDATE_HELPER)}
    return {
        "commit": commit,
        "python_source_sha256": aggregate.hexdigest(),
        "operator_path": relative,
        "operator_sha256": digest(root / relative),
        "dependency_sha256": dependencies,
    }


class AttentionKernelCapture(LinearKernelCapture):
    def __getitem__(self, grid):
        def run(*args, **kwargs):
            # Native's grid lambda uses only scalar launch constants; preserve
            # its actual resolved dimensions, not the callable's representation.
            resolved = grid(kwargs) if callable(grid) else grid
            compiled = super(AttentionKernelCapture, self).__getitem__(resolved)(
                *args, **kwargs
            )
            self.evidence["launch_kwargs"] = {
                key: value
                for key, value in kwargs.items()
                if isinstance(value, (bool, int, float, str)) or value is None
            }
            return compiled

        return run


def validate_launch(launch, case):
    metadata, kwargs = launch["metadata"], launch["launch_kwargs"]
    expected = {
        "BLOCK_SIZE_Q": 64,
        "BLOCK_SIZE_KV": 16,
        "BLOCK_DIM": 32,
        "USE_EXP2": False,
        "HAS_PAIR_BIAS": True,
        "HEAD": 4,
        "N_SEQ": case.rows,
        "SEQ_LEN": case.length,
        "DIM": case.dim,
        "softmax_scale": case.dim**-0.5,
        "num_warps": 4,
        "num_stages": 1,
    }
    if any(kwargs.get(key) != value for key, value in expected.items()):
        raise ValueError("native launch policy differs")
    if launch["grid"] != [(case.length + 63) // 64, case.rows * 4, 1]:
        raise ValueError("native launch grid differs")
    if (
        metadata["name"] != "_attn_fwd"
        or metadata["num_warps"] != 4
        or metadata["num_stages"] != 1
        or metadata["default_dot_input_precision"] != "tf32"
        or metadata["enable_fp_fusion"] is not True
    ):
        raise ValueError("native compiler policy differs")


def read_reference_case(root, case, record):
    payload = (root / f"{case.name}.npz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["archive_sha256"]:
        raise ValueError("reference archive changed")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(KEYS) | {"first", "second"}:
            raise ValueError("reference archive schema differs")
        arrays = {key: archive[key] for key in KEYS}
        validate_operands(arrays, case)
        first, second = archive["first"], archive["second"]
    repeat = compare_outputs(first, second, case)
    stable = (
        repeat["bitwise_equal"] and repeat["both_finite"]
        if case.finite
        else repeat["semantic_equal"]
    )
    if repeat != record["native_repeat"] or not stable:
        raise ValueError("reference native repeat differs or finite case is nonfinite")
    return arrays, first


def verify_reference(root, *, expected_manifest_sha=None):
    payload = (root / "manifest.json").read_bytes()
    if (
        expected_manifest_sha is not None
        and hashlib.sha256(payload).hexdigest() != expected_manifest_sha
    ):
        raise ValueError("reference manifest changed")
    manifest = json.loads(payload)
    if (
        manifest["mode"] != "native"
        or manifest["completed"] is not True
        or manifest["panel"] != PANEL
        or manifest["policy"] != POLICY
        or manifest["source"]["commit"] != UPSTREAM_COMMIT
        or manifest["source"]["operator_path"] != NATIVE_OPERATOR
    ):
        raise ValueError("reference native source/policy/completion differs")
    if set(manifest["cases"]) != {case.name for case in case_specs()}:
        raise ValueError("reference case panel incomplete")
    for case in case_specs():
        record = manifest["cases"][case.name]
        if (
            record["status"] != "ok"
            or record["spec"] != case.identity()
            or record["wrapper"] != "EvoformerAttention.apply"
        ):
            raise ValueError("reference case specification/status differs")
        read_reference_case(root, case, record)
        validate_launch(record["launch"], case)
        evidence = record["compiler_evidence"]
        if "ptx" not in evidence or set(evidence) - {"ptx", "ttir", "ttgir"}:
            raise ValueError("reference lacks actual native PTX")
        for kind, sha in evidence.items():
            if digest(root / f"{case.name}.{kind}.txt") != sha:
                raise ValueError("reference compiler artifact changed")
    return manifest


def decode_triton_ir(hlo):
    from jax._src.lib.mlir import ir
    from jax._src.pallas.triton.lowering import _new_ir_context

    encoded = re.findall(r'\bir = "((?:\\.|[^"\\])*)"', hlo)
    if len(encoded) != 1:
        raise ValueError("candidate HLO must expose exactly one Triton IR module")
    payload = bytearray()
    text = encoded[0]
    index = 0
    while index < len(text):
        if text[index] == "\\":
            try:
                payload.append(int(text[index + 1 : index + 3], 16))
            except ValueError as exc:
                raise ValueError("invalid escaped Triton IR byte") from exc
            index += 3
        else:
            payload.extend(text[index].encode())
            index += 1
    with _new_ir_context():
        module = ir.Module.parse(bytes(payload))
        report = inspect_carried_ir(module)
        return str(module), report


def inspect_carried_ir(module):
    from jax._src.lib.mlir import ir
    from jax._src.lib.triton import dialect as tt_dialect

    dots = []

    def visit(op):
        if op.name == "tt.dot":
            dots.append(op)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    visit(module.operation)
    carried = []
    for dot in dots:
        owner = dot.operands[2].owner
        owner = getattr(owner, "operation", owner)
        if (
            isinstance(owner, ir.Operation)
            and owner.name == "tt.elementwise_inline_asm"
        ):
            if "mul.rn.f32" in str(owner) and any(
                isinstance(x.owner, ir.Block) for x in owner.operands
            ):
                carried.append(dot)
    if len(dots) != 2 or len(carried) != 1:
        raise ValueError(
            "candidate IR lacks two dots and one live rescaled carried accumulator"
        )
    if any(
        ir.IntegerAttr(dot.attributes["inputPrecision"]).value
        != int(tt_dialect.InputPrecision.TF32)
        for dot in dots
    ):
        raise ValueError("candidate IR dot input policy differs")
    return {"dot_count": len(dots), "live_rescaled_accumulator_dots": len(carried)}


def call_wrapper(module, arrays, native):
    if native:
        return module.EvoformerAttention.apply(*(arrays[key] for key in KEYS))
    return module.native_triangle_attention(**arrays)


def summarize(records, native):
    field = "native_repeat" if native else "native_comparison"
    finite = [
        r[field]
        for r in records.values()
        if r["status"] == "ok" and r["spec"]["finite_gate"]
    ]
    semantic = [
        r[field]
        for r in records.values()
        if r["status"] == "ok" and not r["spec"]["finite_gate"]
    ]
    return {
        "cases": len(records),
        "execution_errors": sum(r["status"] != "ok" for r in records.values()),
        "finite_cases": len(finite),
        "finite_bitwise": sum(r["bitwise_equal"] and r["both_finite"] for r in finite),
        "finite_strict_pass": sum(r["strict_1e4_allclose"] for r in finite),
        "semantic_cases": len(semantic),
        "semantic_equal": sum(r["semantic_equal"] for r in semantic),
        "semantic_bitwise": sum(r["bitwise_equal"] for r in semantic),
    }


def verify_unchanged(root, native, identity, helpers, reference, reference_sha):
    if source_identity(root, native) != identity:
        raise RuntimeError("source changed during probe")
    if helper_identities() != helpers:
        raise RuntimeError("probe/helper source changed during execution")
    if not native:
        verify_reference(reference, expected_manifest_sha=reference_sha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "candidate"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    native = args.mode == "native"
    if native == (args.reference is not None):
        raise ValueError("candidate requires --reference; native must omit it")
    if any(
        os.environ.get(key) == "1"
        for key in ("OF3_TRITON_EXP2", "OF3_TRITON_DYNAMIC_SHAPES")
    ):
        raise ValueError(
            "probe requires specialized native default, exp2/dynamic disabled"
        )
    root = args.source_root.resolve(strict=True)
    identity, helpers = source_identity(root, native), helper_identities()
    reference_sha = None if native else digest(args.reference / "manifest.json")
    reference = (
        None
        if native
        else verify_reference(args.reference, expected_manifest_sha=reference_sha)
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root if native else root / "src"))
    module = importlib.import_module(
        "openfold3.core.kernels.triton.evoformer"
        if native
        else "foldjax.models.openfold3.models.native_triton_attention"
    )
    if (
        Path(inspect.getfile(module)).resolve()
        != (root / identity["operator_path"]).resolve()
    ):
        raise RuntimeError("operator imported from wrong source")
    if native:
        import torch
        import triton

        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.version.hip
        ):
            raise RuntimeError("requires exactly one NVIDIA CUDA GPU")
        torch.set_float32_matmul_precision("high")
        capture = AttentionKernelCapture(module._attn_fwd)
        module._attn_fwd = capture
        runtime = {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        }
    else:
        import jax
        import jax.numpy as jnp
        import jaxlib

        if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
            raise RuntimeError("requires exactly one GPU")
        jax.config.update("jax_default_matmul_precision", "high")
        runtime = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "device": str(jax.devices()[0]),
        }
    records = {}
    for case in case_specs():
        record = {
            "spec": case.identity(),
            "wrapper": "EvoformerAttention.apply"
            if native
            else "native_triangle_attention",
        }
        arrays = gpu = first = second = expected = executable = None
        try:
            if native:
                arrays = operands(case)
                validate_operands(arrays, case)
                gpu = {
                    key: torch.from_numpy(value).cuda() for key, value in arrays.items()
                }
                with torch.no_grad():
                    first = call_wrapper(module, gpu, True).cpu().numpy().copy()
                    second = call_wrapper(module, gpu, True).cpu().numpy().copy()
                record["archive_sha256"] = save_arrays(
                    args.out_dir / f"{case.name}.npz",
                    **arrays,
                    first=first,
                    second=second,
                )
                record["native_repeat"] = compare_outputs(first, second, case)
                record["launch"] = {
                    key: value
                    for key, value in capture.evidence.items()
                    if key != "assembly"
                }
                record["compiler_evidence"] = write_compiler_evidence(
                    args.out_dir, case.name, capture.evidence["assembly"]
                )
                validate_launch(record["launch"], case)
                if "ptx" not in record["compiler_evidence"]:
                    raise RuntimeError("native wrapper did not expose actual PTX")
            else:
                arrays, expected = read_reference_case(
                    args.reference, case, reference["cases"][case.name]
                )
                gpu = {key: jnp.asarray(value) for key, value in arrays.items()}
                executable = (
                    jax.jit(lambda data: call_wrapper(module, data, False))
                    .lower(gpu)
                    .compile()
                )
                first, second = np.asarray(executable(gpu)), np.asarray(executable(gpu))
                record["archive_sha256"] = save_arrays(
                    args.out_dir / f"{case.name}.npz", first=first, second=second
                )
                record["candidate_repeat"] = compare_outputs(first, second, case)
                record["native_comparison"] = compare_outputs(first, expected, case)
                hlo = executable.as_text()
                record["compiler_evidence"] = write_compiler_evidence(
                    args.out_dir, case.name, {"hlo": hlo}
                )
                if (
                    "__gpu$xla.gpu.triton" not in hlo
                    or "openbind_native_triangle_attention" not in hlo
                ):
                    raise RuntimeError(
                        "candidate HLO lacks explicit native attention Triton call"
                    )
                ir_text, ir_report = decode_triton_ir(hlo)
                record["compiler_evidence"].update(
                    write_compiler_evidence(args.out_dir, case.name, {"ttir": ir_text})
                )
                record["carried_dot_ir"] = ir_report
            record["status"] = "ok"
        except Exception as exc:
            record.update(
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        records[case.name] = record
        save_new(args.out_dir / f"{case.name}.json", record)
        print(
            json.dumps(
                {
                    "case": case.name,
                    "status": record["status"],
                    "comparison": record.get(
                        "native_comparison", record.get("native_repeat")
                    ),
                    "error": record.get("error"),
                }
            ),
            flush=True,
        )
        del arrays, gpu, first, second, expected, executable
    verify_unchanged(root, native, identity, helpers, args.reference, reference_sha)
    summary = summarize(records, native)
    save_new(
        args.out_dir / "manifest.json",
        {
            "mode": args.mode,
            "completed": summary["execution_errors"] == 0,
            "all_cases_attempted": True,
            "panel": PANEL,
            "source": identity,
            "harness_sha256": helpers[Path(__file__).name],
            "helper_source_sha256": helpers,
            "runtime": runtime,
            "policy": POLICY,
            "cases": records,
            "summary": summary,
            "reference_manifest_sha256": reference_sha,
            "reference_runtime": None if native else reference["runtime"],
            "scope": __doc__,
            "scientific_acceptance": None,
            "full_model_admission": None,
            "strict_leaf_tolerance": {"atol": 1e-4, "rtol": 1e-4},
        },
    )
    print(json.dumps({"summary": summary}), flush=True)
    return 1 if summary["execution_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
