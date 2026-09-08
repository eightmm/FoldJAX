"""Pinned native linear wrappers versus the JAX candidate, operator-only.

Run native capture first, then replay the exact archive arrays in candidate
mode. The finite panel uses synthetic operands, including one full pair shape;
it is not a real-trunk, full-model, preprocessing or performance admission.
Candidate-only --operand-rounding=rtz is a labeled counterfactual: truncate
the archived FP32 dot operands on the host, leaving the runtime kernel intact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import io
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import digest, save_new
from bench.openbind_tape_adapter import UPSTREAM_COMMIT
from bench.openbind_triton_norm_probe import (
    KernelCapture,
    metrics,
    save_arrays,
    write_compiler_evidence,
)

PANEL = "openbind-native-triton-linear-v1"
ARMS = ("plain", "sigmoid", "projection", "residual")
NATIVE_OPERATOR = "openfold3/core/model/layers/triangular_multiplicative_update.py"
CANDIDATE_OPERATOR = "src/foldjax/models/openfold3/models/native_triton_linear.py"


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    outputs: int
    arm: str
    profile: str = "normal"

    def identity(self):
        return {**asdict(self), "shape": list(self.shape)}


def case_specs():
    cases = []
    for inputs, outputs in ((128, 128), (64, 64), (64, 128), (128, 64)):
        rows = (1, 7, 9, 127, 128, 129) if inputs == outputs == 128 else (9, 129)
        for count in rows:
            for arm in ARMS:
                cases.append(
                    Case(
                        f"k{inputs}-n{outputs}-m{count}-{arm}",
                        (count, inputs),
                        outputs,
                        arm,
                    )
                )
    cases.append(Case("k128-n128-pair437-plain", (1, 437, 437, 128), 128, "plain"))
    for width in (64, 128):
        cases.append(
            Case(f"k{width}-n{width}-m9-clamp", (9, width), width, "sigmoid", "clamp")
        )
    cases.append(Case("k128-n128-m9-plain-bias", (9, 128), 128, "plain", "bias"))
    return cases


def operand_shapes(case):
    if case.arm not in ARMS or case.profile not in {"normal", "clamp", "bias"}:
        raise ValueError("unsupported synthetic linear case")
    if case.shape[-1] not in (64, 128) or case.outputs not in (64, 128):
        raise ValueError("unsupported synthetic linear dimensions")
    if case.profile == "clamp" and case.arm != "sigmoid":
        raise ValueError("clamp controls require sigmoid")
    shapes = {"x": case.shape, "weight": (case.outputs, case.shape[-1])}
    output = (*case.shape[:-1], case.outputs)
    if case.profile in {"clamp", "bias"}:
        shapes["bias"] = (case.outputs,)
    if case.arm in {"projection", "residual"}:
        shapes["other"] = output
    if case.arm == "projection":
        shapes["mask"] = (*case.shape[:-1], 1)
    if case.arm == "residual":
        shapes["add_tensor"] = output
    return shapes


def operands(case):
    shapes = operand_shapes(case)
    inputs, outputs = case.shape[-1], case.outputs
    rows = math.prod(case.shape[:-1])
    rng = np.random.default_rng(9203 + inputs + outputs)
    # Repeat nine distinct rows to keep the full pair-shaped archive bounded.
    # This controls physical layout/launch size, not real activation coverage.
    indices = np.arange(rows) % min(rows, 9)
    x = rng.normal(size=(min(rows, 9), inputs)).astype(np.float32)
    arrays = {
        "x": x[indices].reshape(case.shape),
        "weight": (rng.normal(size=(outputs, inputs)) / np.sqrt(inputs)).astype(
            np.float32
        ),
    }
    if "bias" in shapes:
        arrays["bias"] = rng.uniform(-0.3, 0.3, outputs).astype(np.float32)
    if case.profile == "clamp":
        arrays["weight"].fill(0)
        boundary = np.array(
            [
                -100,
                -21,
                -20.0001,
                -20,
                -19.9999,
                -1,
                0,
                1,
                19.9999,
                20,
                20.0001,
                21,
                100,
            ],
            np.float32,
        )
        arrays["bias"] = np.resize(boundary, outputs)
    if "other" in shapes:
        values = (
            rng.uniform(0, 1, (min(rows, 9), outputs))
            if case.arm == "projection"
            else rng.normal(size=(min(rows, 9), outputs))
        )
        arrays["other"] = values.astype(np.float32)[indices].reshape(shapes["other"])
    if "mask" in shapes:
        arrays["mask"] = (
            (np.arange(rows) % 3 != 2).astype(np.float32).reshape(shapes["mask"])
        )
    if "add_tensor" in shapes:
        values = rng.normal(size=(min(rows, 9), outputs)).astype(np.float32)
        arrays["add_tensor"] = values[indices].reshape(shapes["add_tensor"])
    return arrays


def validate_operands(arrays, case):
    shapes = operand_shapes(case)
    if set(arrays) != set(shapes):
        raise ValueError("wrong operand keys for frozen native case")
    for key, value in arrays.items():
        if value.shape != shapes[key] or value.dtype != np.float32:
            raise ValueError(f"wrong operand shape/dtype: {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite operand: {key}")


def candidate_operands(arrays, rounding):
    """Control only dot-input rounding; never bias or the fused epilogue."""
    if rounding == "none":
        return arrays
    if rounding != "rtz":
        raise ValueError("unknown operand rounding control")
    controlled = dict(arrays)
    for key in ("x", "weight"):
        value = arrays[key]
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("RTZ control requires finite FP32 dot operands")
        # Clearing the low 13 fraction bits truncates toward zero for either
        # sign. Exact TF32-grid values are unchanged by a subsequent rounding.
        controlled[key] = (value.view(np.uint32) & np.uint32(0xFFFFE000)).view(
            np.float32
        )
    return controlled


def operand_control_metadata(rounding):
    if rounding not in {"none", "rtz"}:
        raise ValueError("unknown operand rounding control")
    return {
        "mode": rounding,
        "operands": [] if rounding == "none" else ["x", "weight"],
        "stage": "none" if rounding == "none" else "host_before_device_transfer",
        "rule": None if rounding == "none" else "float32_uint32_bits & 0xffffe000",
        "runtime_operator_modified": False,
    }


def wrapper_name(case, native):
    prefix = "triton" if native else "native"
    return f"{prefix}_linear" if case.arm == "plain" else f"{prefix}_linear_fused"


def call_wrapper(module, case, arrays, *, native):
    wrapper = getattr(module, wrapper_name(case, native))
    if case.arm == "plain":
        return wrapper(**arrays)
    return wrapper(**arrays, apply_sigmoid=case.arm in {"sigmoid", "residual"})


def helper_identities():
    root = Path(__file__).resolve().parent
    return {
        name: digest(root / name)
        for name in (
            "openbind_triton_linear_probe.py",
            "openbind_triton_norm_probe.py",
            "boltz_historical_replay.py",
            "openbind_tape_adapter.py",
        )
    }


def source_identity(root, native):
    """Hash the actual linear implementation, never the norm probe's target."""
    relative = NATIVE_OPERATOR if native else CANDIDATE_OPERATOR
    if native:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--"])
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
            or diff
            or any(p.endswith(".py") for p in untracked.splitlines())
        ):
            raise ValueError("native source must be clean pinned OpenBind")
    else:
        commit = None
    operator = root / relative
    if not operator.is_file():
        raise FileNotFoundError(operator)
    base = root / ("openfold3" if native else "src")
    aggregate = hashlib.sha256()
    files = sorted(base.rglob("*.py"))
    if not files:
        raise ValueError("selected source has no Python files")
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("source escapes selected root")
        aggregate.update(str(path.relative_to(root)).encode())
        aggregate.update(bytes.fromhex(digest(path)))
    return {
        "commit": commit,
        "python_source_sha256": aggregate.hexdigest(),
        "operator_path": relative,
        "operator_sha256": digest(operator),
    }


class LinearKernelCapture(KernelCapture):
    def __getitem__(self, grid):
        launch = super().__getitem__(grid)

        def run(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            metadata = getattr(compiled, "metadata", None)
            self.evidence["metadata"] = {
                key: (
                    metadata.get(key)
                    if isinstance(metadata, dict)
                    else getattr(metadata, key, None)
                )
                for key in (
                    "name",
                    "num_warps",
                    "num_stages",
                    "default_dot_input_precision",
                    "enable_fp_fusion",
                )
            }
            return compiled

        return run


def read_reference_case(root, case, record):
    # Hash the SAME bytes decoded below, not a separate path read. This binds
    # the consumed operands even if the archive is replaced during the probe.
    payload = (root / f"{case.name}.npz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["archive_sha256"]:
        raise ValueError(f"reference artifact changed: {case.name}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        keys = set(operand_shapes(case))
        if set(archive.files) != keys | {"first", "second"}:
            raise ValueError("reference archive schema differs")
        arrays = {key: archive[key] for key in keys}
        validate_operands(arrays, case)
        first, second = archive["first"], archive["second"]
        shape = (*case.shape[:-1], case.outputs)
        if first.shape != shape or first.dtype != np.float32:
            raise ValueError("reference output shape/dtype differs")
        actual = metrics(first, second)
        if not actual["bitwise_equal"] or actual != record["native_repeat"]:
            raise ValueError("reference native repeat differs")
    return arrays, first


def verify_reference(root, *, expected_manifest_sha=None):
    payload = (root / "manifest.json").read_bytes()
    if (
        expected_manifest_sha is not None
        and hashlib.sha256(payload).hexdigest() != expected_manifest_sha
    ):
        raise ValueError("reference manifest changed during probe")
    manifest = json.loads(payload)
    if manifest["mode"] != "native" or manifest["completed"] is not True:
        raise ValueError("reference is not a completed native capture")
    if (
        manifest["panel"] != PANEL
        or manifest["dtype"] != "float32"
        or manifest["float32_matmul_precision"] != "high"
        or manifest["source"]["commit"] != UPSTREAM_COMMIT
        or manifest["source"]["operator_path"] != NATIVE_OPERATOR
        or manifest.get("operand_control", {}).get("mode", "none") != "none"
    ):
        raise ValueError("reference source/policy differs from pinned linear panel")
    if set(manifest["cases"]) != {case.name for case in case_specs()}:
        raise ValueError("reference case panel is incomplete")
    for case in case_specs():
        record = manifest["cases"][case.name]
        if record["spec"] != case.identity() or record["wrapper"] != wrapper_name(
            case, True
        ):
            raise ValueError("reference case specification/wrapper differs")
        read_reference_case(root, case, record)
        evidence = record["compiler_evidence"]
        if "ptx" not in evidence or set(evidence) - {"ptx", "ttir", "ttgir"}:
            raise ValueError("reference lacks actual native PTX evidence")
        metadata = record["launch"]["metadata"]
        expected_kernel = (
            "linear_kernel" if case.arm == "plain" else "linear_fused_kernel"
        )
        if (
            metadata["name"] != expected_kernel
            or metadata["default_dot_input_precision"] != "tf32"
            or metadata["enable_fp_fusion"] is not True
            or metadata["num_warps"] != 4
            or metadata["num_stages"] != 3
        ):
            raise ValueError("reference native compiler policy differs")
        for kind, sha in evidence.items():
            if digest(root / f"{case.name}.{kind}.txt") != sha:
                raise ValueError("reference compiler artifact changed")
    return manifest


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
    parser.add_argument(
        "--operand-rounding",
        choices=("none", "rtz"),
        default="none",
        help="candidate-only dot operand counterfactual; default leaves arrays intact",
    )
    args = parser.parse_args(argv)
    native = args.mode == "native"
    if native == (args.reference is not None):
        raise ValueError("candidate requires --reference; native must omit it")
    if native and args.operand_rounding != "none":
        raise ValueError("operand rounding control is candidate-only")
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
    if native:
        import torch
        import triton

        module = importlib.import_module(
            "openfold3.core.model.layers.triangular_multiplicative_update"
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("requires exactly one CUDA GPU")
        torch.set_float32_matmul_precision("high")
        captures = {
            name: LinearKernelCapture(getattr(module, name))
            for name in ("linear_kernel", "linear_fused_kernel")
        }
        for name, capture in captures.items():
            setattr(module, name, capture)
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

        module = importlib.import_module(
            "foldjax.models.openfold3.models.native_triton_linear"
        )
        if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
            raise RuntimeError("requires exactly one GPU")
        jax.config.update("jax_default_matmul_precision", "high")
        runtime = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "device": str(jax.devices()[0]),
        }
    if (
        Path(inspect.getfile(module)).resolve()
        != (root / identity["operator_path"]).resolve()
    ):
        raise RuntimeError("operator imported from wrong source")
    records = {}
    for case in case_specs():
        if native:
            arrays = operands(case)
            validate_operands(arrays, case)
            gpu = {key: torch.from_numpy(value).cuda() for key, value in arrays.items()}
            with torch.no_grad():
                first = (
                    call_wrapper(module, case, gpu, native=True).cpu().numpy().copy()
                )
                second = (
                    call_wrapper(module, case, gpu, native=True).cpu().numpy().copy()
                )
            key = "linear_kernel" if case.arm == "plain" else "linear_fused_kernel"
            capture = captures[key].evidence
            compiler = write_compiler_evidence(
                args.out_dir, case.name, capture["assembly"]
            )
            if "ptx" not in compiler:
                raise RuntimeError("native wrapper did not provide compiled PTX")
            record = {
                "native_repeat": metrics(first, second),
                "archive_sha256": save_arrays(
                    args.out_dir / f"{case.name}.npz",
                    **arrays,
                    first=first,
                    second=second,
                ),
                "compiler_evidence": compiler,
                "launch": {
                    key: value for key, value in capture.items() if key != "assembly"
                },
            }
        else:
            arrays, expected = read_reference_case(
                args.reference, case, reference["cases"][case.name]
            )
            arrays = candidate_operands(arrays, args.operand_rounding)
            gpu = {key: jnp.asarray(value) for key, value in arrays.items()}
            executable = (
                jax.jit(lambda data: call_wrapper(module, case, data, native=False))
                .lower(gpu)
                .compile()
            )
            first, second = np.asarray(executable(gpu)), np.asarray(executable(gpu))
            hlo = executable.as_text()
            if (
                "__gpu$xla.gpu.triton" not in hlo
                or "openbind_native_triangle_linear_fused" not in hlo
            ):
                raise RuntimeError(
                    "candidate HLO lacks the explicit Triton linear call"
                )
            record = {
                "candidate_repeat": metrics(first, second),
                "native_comparison": metrics(first, expected),
                "archive_sha256": save_arrays(
                    args.out_dir / f"{case.name}.npz", first=first, second=second
                ),
                "compiler_evidence": write_compiler_evidence(
                    args.out_dir, case.name, {"hlo": hlo}
                ),
                "effective_dot_operand_sha256": {
                    key: hashlib.sha256(arrays[key].tobytes(order="C")).hexdigest()
                    for key in ("x", "weight")
                },
            }
        records[case.name] = {
            "spec": case.identity(),
            "wrapper": wrapper_name(case, native),
            **record,
        }
        print(
            json.dumps(
                {
                    "case": case.name,
                    **{
                        key: value
                        for key, value in record.items()
                        if key.endswith("repeat") or key == "native_comparison"
                    },
                }
            ),
            flush=True,
        )
        # The pair-shaped control should not retain all prior live device arrays.
        del gpu, arrays, first, second
    verify_unchanged(root, native, identity, helpers, args.reference, reference_sha)
    if not native:
        summary = {
            "cases": len(records),
            "bitwise_equal": sum(
                r["native_comparison"]["bitwise_equal"] for r in records.values()
            ),
            "strict_1e4_pass": sum(
                r["native_comparison"]["strict_1e4_allclose"] for r in records.values()
            ),
        }
    else:
        summary = {
            "cases": len(records),
            "native_repeat_bitwise": sum(
                r["native_repeat"]["bitwise_equal"] for r in records.values()
            ),
        }
    save_new(
        args.out_dir / "manifest.json",
        {
            "mode": args.mode,
            "completed": True,
            "panel": PANEL,
            "source": identity,
            "harness_sha256": helpers[Path(__file__).name],
            "helper_source_sha256": helpers,
            "runtime": runtime,
            "dtype": "float32",
            "float32_matmul_precision": "high",
            "operand_control": operand_control_metadata(args.operand_rounding),
            "cases": records,
            "summary": summary,
            "reference_manifest_sha256": reference_sha,
            "reference_runtime": None if native else reference["runtime"],
            "scope": (
                "synthetic operator-only including one real-sized pair shape; "
                "not real activations/model parity/performance"
            ),
            "scientific_acceptance": None,
            "full_model_admission": None,
            "strict_leaf_tolerance": {"atol": 1e-4, "rtol": 1e-4},
        },
    )
    print(json.dumps({"summary": summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
