"""Source-isolated FP32 historical replay; completion is not a parity verdict.

The same current coordinate runner imports only the explicitly selected candidate.
Native features are shared, so this does not test independent preprocessing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(root):
    files = sorted((root / "src").rglob("*.py"))
    if not files:
        raise ValueError("candidate has no Python source")
    if any(not p.resolve().is_relative_to(root) for p in files):
        raise ValueError("candidate source escapes selected root")
    return {str(p.relative_to(root)): digest(p) for p in files}


def validate_api(candidate, runner):
    """Fail closed when current runner explicitly calls an absent old option."""
    target = candidate / "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py"
    definitions = {
        node.name: node
        for node in ast.parse(target.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    tree = ast.parse(runner.read_text())
    shared = {
        "chunk_size",
        "matmul_precision",
        "attention_backend",
        "triangle_backend",
        "glu_backend",
    }
    for name in ("boltz2_trunk_forward", "boltz2_sample_forward", "_sample_schedule"):
        if name not in definitions:
            raise ValueError(f"candidate lacks {name}")
        args = definitions[name].args
        accepted = {arg.arg for arg in [*args.args, *args.kwonlyargs]}
        for call in ast.walk(tree):
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Name)
                or call.func.id != name
            ):
                continue
            requested = {kw.arg for kw in call.keywords if kw.arg is not None}
            if any(kw.arg is None for kw in call.keywords):
                requested |= shared
            missing = requested - accepted
            if missing:
                raise ValueError(
                    f"unsupported explicit {name} options: {sorted(missing)}"
                )


def native_hashes(root):
    meta = json.loads((root / "tape.json").read_text())
    if (str(meta["precision"]), meta["kernels"], meta["subsample_msa"]) != (
        "32",
        False,
        False,
    ):
        raise ValueError("requires native FP32, kernels disabled, full MSA")
    if (meta["num_samples"], meta["num_steps"], meta["num_recycles"], meta["seed"]) != (
        5,
        200,
        3,
        101,
    ):
        raise ValueError("requires n5/200/3/seed101")
    return {
        name: digest(root / name)
        for name in (
            "tape.json",
            "tape.npz",
            "features.npz",
            "trunk.npz",
            "coordinate.npz",
        )
    }


CHILD = r"""
import importlib.metadata, inspect, json, os, pathlib, runpy, sys
candidate, runner, runtime_path, *arguments = sys.argv[1:]
import foldjax
import foldjax.models.boltz2.models.trunk_blocks.trunk as trunk
import foldjax.models.boltz2.bridge.native as bridge
for module in (foldjax, trunk, bridge):
    module_path = pathlib.Path(inspect.getfile(module)).resolve()
    if not module_path.is_relative_to(pathlib.Path(candidate) / "src"):
        raise RuntimeError("candidate import escaped selected source")
import jax
runtime = {"python": sys.version, "executable": sys.executable,
           "packages": {}, "devices": [str(d) for d in jax.devices()]}
runtime["execution_environment"] = {
    key: os.environ.get(key) for key in (
        "XLA_FLAGS", "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES",
        "JAX_ENABLE_X64", "JAX_COMPILATION_CACHE_DIR",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "CUBLAS_WORKSPACE_CONFIG",
    )
}
packages = ("jax", "jaxlib", "numpy", "cuequivariance",
            "cuequivariance-jax", "jax-cuda13-plugin")
for package in packages:
    try: runtime["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: runtime["packages"][package] = None
with open(runtime_path, "x") as stream: json.dump(runtime, stream, indent=2)
if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
    raise RuntimeError("replay requires exactly one GPU")
sys.argv = [runner, *arguments]
runpy.run_path(runner, run_name="__main__")
"""


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--native-capture", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    candidate = args.candidate_root.resolve(strict=True)
    native = args.native_capture.resolve(strict=True)
    runner = (
        Path(__file__).resolve().parents[1]
        / "tests/models/boltz2/scripts/parity_matched_tape.py"
    )
    weights = args.weights.resolve(strict=True)
    if not weights.is_file():
        raise ValueError("select one explicit safetensors checkpoint file")
    validate_api(candidate, runner)
    identity = {
        "candidate_source": source_hashes(candidate),
        "native_artifacts": native_hashes(native),
        "runner_sha256": digest(runner),
        "harness_sha256": digest(Path(__file__)),
        "checkpoint_sha256": digest(weights),
        "tape_scope": (
            "native init/churn/augmentation arrays; legacy runner regenerates "
            "sigma schedule and reports its difference, not dynamic sigma injection"
        ),
        "scope": (
            "FP32 core-only, native features, original sample order; "
            "no confidence or performance admission"
        ),
        "settings": {
            "compute_dtype": "float32",
            "triangle_backend": "xla",
            "triangle_multiplication_backend": "cueq",
            "attention_backend": "xla",
            "trunk_atom_attention_backend": "xla",
            "matmul_precision": "highest",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    save_new(args.out_dir / "provenance.json", identity)
    if args.preflight_only:
        return 0
    output = args.out_dir.resolve()
    env = dict(os.environ, PYTHONPATH=str(candidate / "src"), PYTHONNOUSERSITE="1")
    command = [
        str(args.python.absolute()),
        "-c",
        CHILD,
        str(candidate),
        str(runner),
        str(output / "runtime.json"),
        "--tape-dir",
        str(native),
        "--weights",
        str(weights),
        "--out-dir",
        str(output / "legacy"),
        "--compute-dtype",
        "float32",
        "--triangle-backend",
        "xla",
        "--triangle-multiplication-backend",
        "cueq",
        "--attention-backend",
        "xla",
        "--trunk-atom-attention-backend",
        "xla",
        "--trunk-source",
        "jax",
    ]
    with (output / "legacy.log").open("x") as log:
        result = subprocess.run(
            command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if (
        source_hashes(candidate) != identity["candidate_source"]
        or native_hashes(native) != identity["native_artifacts"]
        or digest(runner) != identity["runner_sha256"]
        or digest(weights) != identity["checkpoint_sha256"]
    ):
        raise RuntimeError("bound source or input changed during replay")
    coordinate = output / "legacy/coordinate.npy"
    if result.returncode not in (0, 1) or not coordinate.is_file():
        raise RuntimeError(
            f"replay failed before coordinate export, exit={result.returncode}; "
            "inspect legacy.log"
        )
    value = np.load(coordinate, allow_pickle=False)
    with np.load(native / "coordinate.npz", allow_pickle=False) as reference:
        if value.shape != reference["coordinate"].shape or not np.isfinite(value).all():
            raise ValueError("invalid exported coordinates")
    save_new(
        output / "completion.json",
        {
            "coordinate_exported": True,
            "scientific_acceptance": None,
            "legacy_exit_code_diagnostic_only": result.returncode,
            "coordinate_sha256": digest(coordinate),
            "runtime_sha256": digest(output / "runtime.json"),
            "required_next_step": (
                "whole-system proper Kabsch once; entity RMSD without refit"
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
