"""Compare one real ESMFold2 pair-update block with a mapped transition.

This is a synthetic composed-block diagnostic.  It does not run the folding
pipeline, assess structure accuracy, or change model source.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from bench.esmfold2_transition_loop_probe import (
    _array_metrics,
    _memory,
    _sha256_bytes,
    mapped_transition,
)

BLOCK_PREFIX = "folding_trunk.blocks.0"
TRANSITION_PREFIX = f"{BLOCK_PREFIX}.pair_transition"
PAIR_WIDTH = 256
_TRIANGLE_NAMES = ("tri_mul_out", "tri_mul_in")


def _required_weight_names() -> set[str]:
    names = {
        f"{TRANSITION_PREFIX}.norm.weight",
        f"{TRANSITION_PREFIX}.norm.bias",
        f"{TRANSITION_PREFIX}.ffn.w12.weight",
        f"{TRANSITION_PREFIX}.ffn.w3.weight",
    }
    for direction in _TRIANGLE_NAMES:
        engine = f"{BLOCK_PREFIX}.{direction}._engine"
        names.update(
            {
                f"{engine}.norm_start.weight",
                f"{engine}.norm_start.bias",
                f"{engine}.proj_bundle.weight",
                f"{engine}.norm_mix.weight",
                f"{engine}.norm_mix.bias",
                f"{engine}.proj_emit.weight",
                f"{engine}.proj_gate.weight",
            }
        )
    return names


def _optional_weight_names() -> set[str]:
    names = {
        f"{TRANSITION_PREFIX}.ffn.w12.bias",
        f"{TRANSITION_PREFIX}.ffn.w3.bias",
    }
    for direction in _TRIANGLE_NAMES:
        engine = f"{BLOCK_PREFIX}.{direction}._engine"
        names.update(
            {
                f"{engine}.proj_bundle.bias",
                f"{engine}.proj_emit.bias",
                f"{engine}.proj_gate.bias",
            }
        )
    return names


def _shape(value: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if value.shape != expected:
        raise ValueError(f"unexpected shape for {name}: {value.shape} != {expected}")


def _validate_pair_block_weights(selected: dict[str, np.ndarray]) -> None:
    names = set(selected)
    required = _required_weight_names()
    unexpected = names - required - _optional_weight_names()
    missing = required - names
    if missing or unexpected:
        raise ValueError(
            "invalid first pair-block weights: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for direction in _TRIANGLE_NAMES:
        engine = f"{BLOCK_PREFIX}.{direction}._engine"
        norm_start = selected[f"{engine}.norm_start.weight"]
        _shape(norm_start, (PAIR_WIDTH,), f"{engine}.norm_start.weight")
        _shape(
            selected[f"{engine}.norm_start.bias"],
            norm_start.shape,
            f"{engine}.norm_start.bias",
        )
        bundled = selected[f"{engine}.proj_bundle.weight"]
        if bundled.ndim != 2 or bundled.shape[1] != PAIR_WIDTH or bundled.shape[0] % 4:
            raise ValueError(
                f"invalid triangle bundle shape for {engine}: {bundled.shape}"
            )
        hidden = bundled.shape[0] // 4
        _shape(
            selected[f"{engine}.norm_mix.weight"],
            (hidden,),
            f"{engine}.norm_mix.weight",
        )
        _shape(
            selected[f"{engine}.norm_mix.bias"], (hidden,), f"{engine}.norm_mix.bias"
        )
        _shape(
            selected[f"{engine}.proj_emit.weight"],
            (PAIR_WIDTH, hidden),
            f"{engine}.proj_emit.weight",
        )
        _shape(
            selected[f"{engine}.proj_gate.weight"],
            (PAIR_WIDTH, PAIR_WIDTH),
            f"{engine}.proj_gate.weight",
        )
        for projection, output_width in (
            ("proj_bundle", bundled.shape[0]),
            ("proj_emit", PAIR_WIDTH),
            ("proj_gate", PAIR_WIDTH),
        ):
            bias = selected.get(f"{engine}.{projection}.bias")
            if bias is not None:
                _shape(bias, (output_width,), f"{engine}.{projection}.bias")
    _shape(
        selected[f"{TRANSITION_PREFIX}.norm.weight"],
        (PAIR_WIDTH,),
        "transition norm weight",
    )
    _shape(
        selected[f"{TRANSITION_PREFIX}.norm.bias"],
        (PAIR_WIDTH,),
        "transition norm bias",
    )
    w12 = selected[f"{TRANSITION_PREFIX}.ffn.w12.weight"]
    if (
        w12.ndim != 2
        or w12.shape[1] != PAIR_WIDTH
        or w12.shape[0] <= 0
        or w12.shape[0] % 2
    ):
        raise ValueError(f"invalid transition w12 shape: {w12.shape}")
    hidden = w12.shape[0] // 2
    _shape(
        selected[f"{TRANSITION_PREFIX}.ffn.w3.weight"],
        (PAIR_WIDTH, hidden),
        "transition w3",
    )
    for name, width in (("w12", w12.shape[0]), ("w3", PAIR_WIDTH)):
        bias = selected.get(f"{TRANSITION_PREFIX}.ffn.{name}.bias")
        if bias is not None:
            _shape(bias, (width,), f"transition {name} bias")


def load_pair_block_weights(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and bind every safetensors leaf used by the first pair block."""
    from safetensors import safe_open

    selected: dict[str, np.ndarray] = {}
    digests: list[dict[str, Any]] = []
    with safe_open(str(path), framework="numpy") as handle:
        for name in sorted(
            key for key in handle.keys() if key.startswith(f"{BLOCK_PREFIX}.")
        ):
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
    _validate_pair_block_weights(selected)
    return selected, {"prefix": BLOCK_PREFIX, "tensors": digests}


@contextlib.contextmanager
def patched_mapped_transition(trunk: ModuleType) -> Iterator[None]:
    """Temporarily route only the selected transition through the mapped helper."""
    original = trunk._autocast_transition

    def replacement(x, params, prefix, residual, eps):
        if prefix != TRANSITION_PREFIX:
            return original(x, params, prefix, residual, eps)
        return mapped_transition(
            x, params, prefix, residual=residual, eps=eps, original=original
        )

    trunk._autocast_transition = replacement
    try:
        yield
    finally:
        trunk._autocast_transition = original


def pair_block_call(trunk: ModuleType, pair, params, mask):
    """Invoke the unmodified first pair-update block with its native settings."""
    return trunk.pair_update_block(
        pair,
        params,
        BLOCK_PREFIX,
        mask=mask,
        native_autocast=True,
    )


def _compile(fn, pair, params, mask) -> tuple[Any, dict[str, Any]]:
    import jax

    lowered_at = time.perf_counter()
    lowered = jax.jit(fn).lower(pair, params, mask)
    lowering_seconds = time.perf_counter() - lowered_at
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


def _execute(compiled, pair, params, mask) -> tuple[Any, float]:
    started = time.perf_counter()
    result = compiled(pair, params, mask)
    result.block_until_ready()
    return result, time.perf_counter() - started


def validate_input_args(rows: int, input_scale: float) -> None:
    if rows not in (129, 2096):
        raise ValueError("--rows must be 129 or 2096")
    if not math.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("--input-scale must be finite and positive")


def validate_output_path(weights: Path, output: Path) -> None:
    """Refuse the checkpoint itself, including a symlink or hard-link alias."""
    if weights.resolve() == output.resolve():
        raise ValueError("--output must not resolve to --weights")
    try:
        same_file = os.path.samefile(weights, output)
    except FileNotFoundError:
        same_file = False
    if same_file:
        raise ValueError("--output must not alias --weights")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument(
        "--input-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()
    validate_input_args(args.rows, args.input_scale)
    validate_output_path(args.weights, args.output)

    import jax
    import jax.numpy as jnp

    from foldjax.models.esmfold2.models import trunk

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("exactly one GPU is required")
    weights, weight_record = load_pair_block_weights(args.weights)
    dtype = jnp.bfloat16 if args.input_dtype == "bfloat16" else jnp.float32
    pair = jnp.asarray(
        np.random.default_rng(args.seed).normal(
            scale=args.input_scale, size=(1, args.rows, args.rows, PAIR_WIDTH)
        ),
        dtype,
    )
    mask = jnp.ones((1, args.rows, args.rows), dtype=jnp.float32)
    params = {name: jnp.asarray(value) for name, value in weights.items()}

    def original(value, runtime_params, runtime_mask):
        return pair_block_call(trunk, value, runtime_params, runtime_mask)

    original_compiled, original_record = _compile(original, pair, params, mask)
    with patched_mapped_transition(trunk):

        def candidate(value, runtime_params, runtime_mask):
            return pair_block_call(trunk, value, runtime_params, runtime_mask)

        candidate_compiled, candidate_record = _compile(candidate, pair, params, mask)
    outputs = {}
    warm = {}
    outputs["original"], warm["original"] = _execute(
        original_compiled, pair, params, mask
    )
    outputs["candidate"], warm["candidate"] = _execute(
        candidate_compiled, pair, params, mask
    )
    report = {
        "schema_version": 1,
        "scope": "synthetic_esmfold2_first_pair_update_block_composition_probe",
        "full_pipeline_claim": "not_assessed",
        "structure_accuracy_claim": "not_assessed",
        "timing_scope": "diagnostic warm execution only; not a performance claim",
        "candidate_source_change": "none",
        "weights": weight_record,
        "input": {
            "synthetic": True,
            "rng": f"numpy.default_rng({args.seed})",
            "scale": args.input_scale,
            "dtype": args.input_dtype,
            "shape": [1, args.rows, args.rows, PAIR_WIDTH],
            "mask": {
                "shape": [1, args.rows, args.rows],
                "dtype": "float32",
                "value": "ones",
            },
        },
        "jax": jax.__version__,
        "device": {
            "repr": str(jax.devices()[0]),
            "platform": jax.devices()[0].platform,
            "device_kind": jax.devices()[0].device_kind,
        },
        "probe_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "transition_helper_sha256": _sha256_bytes(
            Path(inspect.getfile(mapped_transition)).read_bytes()
        ),
        "trunk_source_sha256": _sha256_bytes(Path(inspect.getfile(trunk)).read_bytes()),
        "variants": {
            "original": {**original_record, "warm_seconds": warm["original"]},
            "candidate": {**candidate_record, "warm_seconds": warm["candidate"]},
        },
        "comparison": _array_metrics(outputs["original"], outputs["candidate"]),
    }
    report["diagnostic_composition_exact"] = bool(
        report["comparison"]["finite"] and report["comparison"]["bit_identical"]
    )
    report["full_model_admission"] = False
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
