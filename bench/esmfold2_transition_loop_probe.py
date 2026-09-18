"""Isolated compile-loop probe for ESMFold2's native-autocast transition.

This is synthetic, loop-only evidence.  It neither runs nor represents the
full folding pipeline and never changes the model implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

BLOCK_ROWS = 64
PREFIX = "folding_trunk.blocks.0.pair_transition"
WEIGHT_SUFFIXES = (
    "norm.weight",
    "norm.bias",
    "ffn.w12.weight",
    "ffn.w3.weight",
)
DEFAULT_SHAPES = ((65, 128), (128, 256), (2096, 128))


def mapped_transition(
    x,
    params,
    prefix: str,
    *,
    residual: bool,
    eps: float = 1e-5,
    original: Callable | None = None,
):
    """Apply the original 64-row body through ``lax.map`` plus its exact tail."""
    import jax
    import jax.numpy as jnp

    if x.shape[1] <= 0:
        raise ValueError("transition row count must be positive")
    if original is None:
        from foldjax.models.esmfold2.models import trunk

        original = trunk._autocast_transition
    full_blocks, tail_rows = divmod(x.shape[1], BLOCK_ROWS)
    if full_blocks == 0:
        return original(x, params, prefix, residual, eps)
    full = x[:, : full_blocks * BLOCK_ROWS]
    blocks = full.reshape((x.shape[0], full_blocks, BLOCK_ROWS, *x.shape[2:]))
    blocks = jnp.swapaxes(blocks, 0, 1)
    mapped = jax.lax.map(
        lambda part: original(part, params, prefix, residual, eps), blocks
    )
    mapped = jnp.swapaxes(mapped, 0, 1).reshape(full.shape)
    if tail_rows == 0:
        return mapped
    tail = original(x[:, full_blocks * BLOCK_ROWS :], params, prefix, residual, eps)
    return jnp.concatenate((mapped, tail), axis=1)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_transition_weights(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only the first folding-trunk pair-transition leaves from safetensors."""
    from safetensors import safe_open

    selected: dict[str, Any] = {}
    digests: list[dict[str, Any]] = []
    with safe_open(str(path), framework="numpy") as handle:
        names = set(handle.keys())
        expected = [f"{PREFIX}.{suffix}" for suffix in WEIGHT_SUFFIXES]
        missing = [name for name in expected if name not in names]
        if missing:
            raise ValueError(f"missing pair-transition weights: {missing}")
        selected_names = sorted(name for name in names if name.startswith(f"{PREFIX}."))
        for name in selected_names:
            value = np.asarray(handle.get_tensor(name))
            selected[name] = value
            digests.append(
                {
                    "key": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": _sha256_bytes(value.tobytes(order="C")),
                }
            )
    return selected, {"prefix": PREFIX, "tensors": digests}


def _memory(analysis) -> dict[str, int | None]:
    return {
        key: getattr(analysis, key, None)
        for key in (
            "temp_size_in_bytes",
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
        )
    }


def _compile(fn, value, params):
    import jax

    traced_at = time.perf_counter()
    lowered = jax.jit(fn).lower(value, params)
    lowering_seconds = time.perf_counter() - traced_at
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    compiled_at = time.perf_counter()
    compiled = lowered.compile()
    return compiled, {
        "lowering_seconds": lowering_seconds,
        "compile_seconds": time.perf_counter() - compiled_at,
        "hlo_sha256": _sha256_bytes(hlo.encode()),
        "hlo_characters": len(hlo),
        "compiler_memory_analysis": _memory(compiled.memory_analysis()),
    }


def _execute(compiled, value, params) -> tuple[Any, float]:
    started = time.perf_counter()
    result = compiled(value, params)
    result.block_until_ready()
    return result, time.perf_counter() - started


def _array_metrics(left, right) -> dict[str, Any]:
    a = np.asarray(left)
    b = np.asarray(right)
    same_shape = a.shape == b.shape
    finite = bool(same_shape and np.isfinite(a).all() and np.isfinite(b).all())
    max_abs = None
    rms = None
    if finite:
        difference = a.astype(np.float32) - b.astype(np.float32)
        max_abs = float(np.max(np.abs(difference)))
        rms = float(np.sqrt(np.mean(difference * difference)))
    return {
        "array_equal": bool(a.dtype == b.dtype and same_shape and np.array_equal(a, b)),
        "bit_identical": bool(
            a.dtype == b.dtype and same_shape and a.tobytes() == b.tobytes()
        ),
        "finite": finite,
        "max_abs": max_abs,
        "rms": rms,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
    }


def _run_shape(
    params, rows: int, columns: int, width: int, residual: bool
) -> dict[str, Any]:
    import jax.numpy as jnp

    from foldjax.models.esmfold2.models import trunk

    rng = np.random.default_rng(101)
    value = jnp.asarray(rng.normal(size=(1, rows, columns, width)), jnp.bfloat16)
    jax_params = {key: jnp.asarray(item) for key, item in params.items()}

    def original(x, runtime_params):
        return trunk._autocast_transition(x, runtime_params, PREFIX, residual, 1e-5)

    def candidate(x, runtime_params):
        return mapped_transition(x, runtime_params, PREFIX, residual=residual)

    compiled = {
        "original": _compile(original, value, jax_params),
        "candidate": _compile(candidate, value, jax_params),
    }
    outputs = {}
    warm = {}
    for name, (executable, _metadata) in compiled.items():
        outputs[name], warm[name] = _execute(executable, value, jax_params)
    timings = {"original": [], "candidate": []}
    measurement_order = []
    for round_index in range(12):
        order = (
            ("original", "candidate")
            if round_index % 2 == 0
            else ("candidate", "original")
        )
        for name in order:
            outputs[name], elapsed = _execute(compiled[name][0], value, jax_params)
            timings[name].append(elapsed)
            measurement_order.append(name)
    return {
        "rows": rows,
        "columns": columns,
        "input_shape": [1, rows, columns, width],
        "synthetic_input": {
            "rng": "numpy.default_rng(101)",
            "dtype": "bfloat16",
            "nonzero": bool(np.any(np.asarray(value))),
        },
        "residual": residual,
        "measurement_rounds": 12,
        "measurement_order": measurement_order,
        "variants": {
            name: {
                **metadata,
                "warm_seconds": warm[name],
                "execution_seconds": timings[name],
                "median_execution_seconds": float(np.median(timings[name])),
            }
            for name, (_executable, metadata) in compiled.items()
        },
        "comparison": _array_metrics(outputs["original"], outputs["candidate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--residual", action="store_true")
    args = parser.parse_args()
    if (args.rows is None) != (args.columns is None):
        raise ValueError("--rows and --columns must be supplied together")
    import jax

    from foldjax.models.esmfold2.models import trunk

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("exactly one GPU is required")
    weights, weight_record = load_transition_weights(args.weights)
    width = int(weights[f"{PREFIX}.norm.weight"].shape[0])
    shapes = ((args.rows, args.columns),) if args.rows is not None else DEFAULT_SHAPES
    if any(rows <= 0 or columns <= 0 for rows, columns in shapes):
        raise ValueError("rows and columns must be positive")
    report = {
        "schema_version": 1,
        "scope": "isolated_esmfold2_transition_loop_synthetic_probe",
        "full_pipeline_claim": "not_assessed",
        "model_optimization_claim": "not_established",
        "weights": weight_record,
        "jax": jax.__version__,
        "device": {
            "repr": str(jax.devices()[0]),
            "platform": jax.devices()[0].platform,
            "device_kind": jax.devices()[0].device_kind,
        },
        "probe_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "trunk_source_sha256": _sha256_bytes(Path(inspect.getfile(trunk)).read_bytes()),
        "shapes": [
            _run_shape(weights, rows, columns, width, args.residual)
            for rows, columns in shapes
        ],
    }
    report["candidate_admissible"] = all(
        item["comparison"]["finite"] and item["comparison"]["bit_identical"]
        for item in report["shapes"]
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["candidate_admissible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
