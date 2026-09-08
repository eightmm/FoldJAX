"""Actual pinned native Triton LayerNorm versus a JAX candidate, operator-only.

Native mode captures two executions of the publisher wrapper on identical
synthetic operands. Candidate mode consumes those exact serialized operands.
Neither mode establishes full-model, real-trunk-input or performance parity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import digest, save_new
from bench.openbind_tape_adapter import UPSTREAM_COMMIT

EPS = 1e-5
PROFILES = ("normal", "below_eps", "at_eps", "above_eps", "large_mean", "constant")


def helper_identities():
    root = Path(__file__).resolve().parent
    return {
        name: digest(root / name)
        for name in (
            "openbind_triton_norm_probe.py",
            "boltz_historical_replay.py",
            "openbind_tape_adapter.py",
        )
    }


def case_specs():
    cases = []
    for width in (64, 128):
        for rows in (7, 8, 9):
            for profile in PROFILES:
                cases.append((f"w{width}-r{rows}-{profile}", (rows, width), profile))
        cases.append((f"w{width}-pair437", (1, 437, 437, width), "normal"))
    return cases


def operands(shape, profile):
    """Fixed balanced profiles distinguish max(var,eps) from var+eps."""
    if profile not in PROFILES or shape[-1] not in (64, 128):
        raise ValueError("unsupported synthetic LayerNorm case")
    width = shape[-1]
    # Repeating 9 distinct rows keeps pair-shaped captures compressible while
    # exercising native multi-program tails. These are not real trunk tensors.
    rng = np.random.default_rng(7301 + width)
    rows = int(np.prod(shape[:-1]))
    values = rng.normal(size=(min(rows, 9), width)).astype(np.float32)
    if profile in {"below_eps", "at_eps", "above_eps"}:
        variance = EPS * {"below_eps": 0.25, "at_eps": 1.0, "above_eps": 4.0}[profile]
        values[:] = np.tile(np.array([-1, 1], np.float32), width // 2)
        values *= np.float32(np.sqrt(variance))
    elif profile == "large_mean":
        values = np.float32(10000) + values * np.float32(0.01)
    elif profile == "constant":
        values.fill(3.0)
    x = values[np.arange(rows) % len(values)].reshape(shape)
    weight = rng.uniform(0.3, 1.7, size=width).astype(np.float32)
    bias = rng.uniform(-0.4, 0.4, size=width).astype(np.float32)
    return {"x": x, "weight": weight, "bias": bias}


def metrics(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("operator output shape/dtype differs")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("operator output is nonfinite")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {
        "array_equal": bool(np.array_equal(left, right)),
        "bitwise_equal": left.tobytes() == right.tobytes(),
        "strict_1e4_allclose": bool(np.allclose(left, right, atol=1e-4, rtol=1e-4)),
        "unequal_values": int(np.count_nonzero(left != right)),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def validate_operands(arrays, shape):
    if set(arrays) != {"x", "weight", "bias"}:
        raise ValueError("incomplete native operand set")
    for name, value in arrays.items():
        expected = shape if name == "x" else (shape[-1],)
        if value.shape != expected or value.dtype != np.float32:
            raise ValueError(f"wrong operand shape/dtype: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite native operand: {name}")


def save_arrays(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return digest(path)


def source_identity(root, native):
    if native:
        relative = "openfold3/core/model/layers/triangular_multiplicative_update.py"
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--"])
        if commit != UPSTREAM_COMMIT or diff:
            raise ValueError("native source must be clean pinned OpenBind")
        base = root / "openfold3"
    else:
        relative = "src/foldjax/models/openfold3/models/native_triton_norm.py"
        base = root / "src"
        commit = None
    if not (root / relative).is_file():
        raise FileNotFoundError(root / relative)
    aggregate = hashlib.sha256()
    for path in sorted(base.rglob("*.py")):
        if not path.resolve().is_relative_to(root):
            raise ValueError("source escapes selected root")
        aggregate.update(str(path.relative_to(root)).encode())
        aggregate.update(bytes.fromhex(digest(path)))
    return {
        "commit": commit,
        "python_source_sha256": aggregate.hexdigest(),
        "operator_path": relative,
        "operator_sha256": digest(root / relative),
    }


def verify_reference(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["mode"] != "native" or manifest["completed"] is not True:
        raise ValueError("reference is not a completed native operator capture")
    if manifest["source"]["commit"] != UPSTREAM_COMMIT or manifest["epsilon"] != EPS:
        raise ValueError("reference source/epsilon differs from pinned policy")
    if list(manifest["cases"]) != [name for name, _, _ in case_specs()]:
        # JSON writer sorts keys, so archive order is not the case order contract.
        if set(manifest["cases"]) != {name for name, _, _ in case_specs()}:
            raise ValueError("reference case panel is incomplete")
    shapes = {name: shape for name, shape, _ in case_specs()}
    for name, record in manifest["cases"].items():
        if digest(root / f"{name}.npz") != record["archive_sha256"]:
            raise ValueError(f"reference artifact changed: {name}")
        if record["native_repeat"]["bitwise_equal"] is not True:
            raise ValueError(f"native repeat differs: {name}")
        with np.load(root / f"{name}.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"x", "weight", "bias", "first", "second"}:
                raise ValueError(f"reference archive schema differs: {name}")
            validate_operands(
                {k: archive[k] for k in ("x", "weight", "bias")}, shapes[name]
            )
            first, second = archive["first"], archive["second"]
            if first.shape != shapes[name] or first.dtype != np.float32:
                raise ValueError(f"reference output shape/dtype differs: {name}")
            actual = metrics(first, second)
            if not actual["bitwise_equal"] or actual != record["native_repeat"]:
                raise ValueError(f"reference repeat metrics disagree: {name}")
        for kind, sha in record.get("compiler_evidence", {}).items():
            if kind not in {"ptx", "ttir", "ttgir", "hlo"}:
                raise ValueError("unknown reference compiler artifact type")
            if digest(root / f"{name}.{kind}.txt") != sha:
                raise ValueError("reference compiler artifact changed")
    return manifest


class KernelCapture:
    """Observe actual compiled native kernel returned by its unmodified launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.evidence = {}

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            self.evidence = {
                "grid": list(grid),
                "launch_kwargs": kwargs,
                "assembly": getattr(compiled, "asm", {}),
            }
            return compiled

        return run


def write_compiler_evidence(out, name, assembly):
    result = {}
    for kind in ("ptx", "ttir", "ttgir", "hlo"):
        text = assembly.get(kind)
        if not isinstance(text, str):
            continue
        path = out / f"{name}.{kind}.txt"
        with path.open("x") as stream:
            stream.write(text)
        result[kind] = digest(path)
    return result


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
    root = args.source_root.resolve(strict=True)
    helpers = helper_identities()
    identity = source_identity(root, native)
    reference_sha = None if native else digest(args.reference / "manifest.json")
    reference = None if native else verify_reference(args.reference)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root if native else root / "src"))
    if native:
        import torch
        import triton

        module = importlib.import_module(
            "openfold3.core.model.layers.triangular_multiplicative_update"
        )
        if not Path(inspect.getfile(module)).resolve().is_relative_to(root):
            raise RuntimeError("native operator imported from wrong source")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("requires exactly one CUDA GPU")
        torch.set_float32_matmul_precision("high")
        capture = KernelCapture(module.layernorm_kernel)
        module.layernorm_kernel = capture
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
            "foldjax.models.openfold3.models.native_triton_norm"
        )
        if not Path(inspect.getfile(module)).resolve().is_relative_to(root / "src"):
            raise RuntimeError("candidate operator imported from wrong source")
        if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
            raise RuntimeError("requires exactly one GPU")
        runtime = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "device": str(jax.devices()[0]),
        }
    records = {}
    for name, shape, profile in case_specs():
        if native:
            arrays = operands(shape, profile)
            validate_operands(arrays, shape)
            gpu = {k: torch.from_numpy(v).cuda() for k, v in arrays.items()}
            with torch.no_grad():
                first = module.triton_layernorm(**gpu, eps=EPS).cpu().numpy().copy()
                second = module.triton_layernorm(**gpu, eps=EPS).cpu().numpy().copy()
            records[name] = {
                "native_repeat": metrics(first, second),
                "archive_sha256": save_arrays(
                    args.out_dir / f"{name}.npz", **arrays, first=first, second=second
                ),
                "compiler_evidence": write_compiler_evidence(
                    args.out_dir, name, capture.evidence["assembly"]
                ),
                "launch": {
                    k: v for k, v in capture.evidence.items() if k != "assembly"
                },
            }
        else:
            with np.load(args.reference / f"{name}.npz", allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in ("x", "weight", "bias")}
                expected = archive["first"]
            validate_operands(arrays, shape)
            gpu = {k: jnp.asarray(v) for k, v in arrays.items()}
            lowered = jax.jit(
                lambda x, weight, bias: module.native_layer_norm(
                    x, weight, bias, eps=EPS
                )
            ).lower(**gpu)
            executable = lowered.compile()
            first = np.asarray(executable(**gpu))
            second = np.asarray(executable(**gpu))
            records[name] = {
                "candidate_repeat": metrics(first, second),
                "native_comparison": metrics(first, expected),
                "archive_sha256": save_arrays(
                    args.out_dir / f"{name}.npz", first=first, second=second
                ),
                "compiler_evidence": write_compiler_evidence(
                    args.out_dir, name, {"hlo": executable.as_text()}
                ),
            }
        print(
            json.dumps(
                {
                    "case": name,
                    **{
                        k: v
                        for k, v in records[name].items()
                        if k.endswith("repeat") or k == "native_comparison"
                    },
                }
            ),
            flush=True,
        )
    if source_identity(root, native) != identity:
        raise RuntimeError("source changed during probe")
    if helper_identities() != helpers:
        raise RuntimeError("probe/helper source changed during execution")
    if not native:
        verify_reference(args.reference)
        if digest(args.reference / "manifest.json") != reference_sha:
            raise RuntimeError("reference manifest changed during probe")
    save_new(
        args.out_dir / "manifest.json",
        {
            "mode": args.mode,
            "completed": True,
            "source": identity,
            "harness_sha256": digest(Path(__file__)),
            "helper_source_sha256": helpers,
            "runtime": runtime,
            "epsilon": EPS,
            "dtype": "float32",
            "cases": records,
            "reference_manifest_sha256": reference_sha,
            "scope": (
                "synthetic operator-only; no model parity or performance admission"
            ),
            "scientific_acceptance": None,
            "full_model_admission": None,
            "strict_leaf_tolerance": {"atol": 1e-4, "rtol": 1e-4},
            "reference_runtime": None if native else reference["runtime"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
