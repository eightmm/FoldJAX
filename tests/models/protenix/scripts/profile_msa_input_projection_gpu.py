"""GPU A/B profile for the dense and compact Protenix MSA input projection."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("protenix", "opendde"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--warm-iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _memory_dict(memory: Any) -> dict[str, int | None]:
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {name: getattr(memory, name, None) for name in names}


def _load_projection_weight(path: Path) -> np.ndarray:
    from foldjax.models.protenix.bridge.weights_io import load_native_weights

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        params = load_native_weights(path)
    root = params.pairformer_output
    weight = np.asarray(jax.device_get(root.msa.linear_m.weight)).copy()
    del params, root
    gc.collect()
    return weight


def _timed(
    executable: Any, arguments: tuple[jax.Array, ...]
) -> tuple[jax.Array, float]:
    started = time.perf_counter()
    output = executable(*arguments)
    jax.block_until_ready(output)
    return output, time.perf_counter() - started


def main() -> None:
    args = _parse_args()
    if args.depth <= 0 or args.tokens <= 0 or args.warm_iters <= 0:
        raise ValueError("depth, tokens, and warm-iters must be positive")
    if jax.default_backend() != "gpu":
        backend = jax.default_backend()
        raise RuntimeError(f"GPU profile requires a GPU backend, got {backend}")

    from foldjax.models.protenix.models.primitives.primitives import LinearParams
    from foldjax.models.protenix.models.trunk_blocks.msa import (
        _dense_msa_input_projection,
        _msa_input_projection,
    )

    device = jax.devices("gpu")[0]
    weight_host = _load_projection_weight(args.weights)
    if weight_host.ndim != 2 or weight_host.shape[1] != 34:
        raise ValueError(f"unexpected linear_m weight shape: {weight_host.shape}")
    if not np.isfinite(weight_host).all():
        raise ValueError("released projection weight is not finite")

    rng = np.random.default_rng(args.seed)
    shape = (args.depth, args.tokens)
    msa = jax.device_put(rng.integers(0, 32, size=shape, dtype=np.uint8), device)
    has_deletion = jax.device_put(
        rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.bool_), device
    )
    deletion_value = jax.device_put(
        rng.normal(size=shape).astype(np.float32), device
    ).astype(jnp.bfloat16)
    weight = jax.device_put(weight_host, device).astype(jnp.bfloat16)
    del weight_host

    dense = jax.jit(
        lambda a, b, c, w: _dense_msa_input_projection(
            a,
            b,
            c,
            LinearParams(w, None),
            activation_dtype=jnp.bfloat16,
        )
    )
    compact = jax.jit(
        lambda a, b, c, w: _msa_input_projection(
            a,
            b,
            c,
            LinearParams(w, None),
            activation_dtype=jnp.bfloat16,
        )
    )
    arguments = (msa, has_deletion, deletion_value, weight)

    compiled: dict[str, Any] = {}
    lowered_metrics: dict[str, Any] = {}
    with jax.default_matmul_precision("high"):
        for name, fn in (("dense", dense), ("compact", compact)):
            lower_started = time.perf_counter()
            lowered = fn.lower(*arguments)
            lower_seconds = time.perf_counter() - lower_started
            stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
            compile_started = time.perf_counter()
            executable = lowered.compile()
            compile_seconds = time.perf_counter() - compile_started
            compiled[name] = executable
            lowered_metrics[name] = {
                "lower_seconds": lower_seconds,
                "compile_seconds": compile_seconds,
                "stablehlo_bytes": len(stablehlo.encode()),
                "stablehlo_dot_general": stablehlo.count("stablehlo.dot_general"),
                "stablehlo_gather": stablehlo.count("stablehlo.gather"),
                "memory": _memory_dict(executable.memory_analysis()),
            }

        outputs: dict[str, jax.Array] = {}
        for name in ("dense", "compact"):
            outputs[name], _ = _timed(compiled[name], arguments)

        timings: dict[str, list[float]] = {"dense": [], "compact": []}
        for index in range(args.warm_iters):
            order = ("dense", "compact") if index % 2 == 0 else ("compact", "dense")
            for name in order:
                outputs[name], elapsed = _timed(compiled[name], arguments)
                timings[name].append(elapsed)

    @jax.jit
    def compare(a: jax.Array, b: jax.Array) -> tuple[jax.Array, ...]:
        a_bits = jax.lax.bitcast_convert_type(a, jnp.uint16)
        b_bits = jax.lax.bitcast_convert_type(b, jnp.uint16)
        difference = a.astype(jnp.float32) - b.astype(jnp.float32)
        return (
            jnp.sum(a_bits != b_bits, dtype=jnp.int32),
            jnp.max(jnp.abs(difference)),
            jnp.sqrt(jnp.mean(jnp.square(difference))),
            jnp.all(jnp.isfinite(a)),
            jnp.all(jnp.isfinite(b)),
        )

    comparison = tuple(
        np.asarray(value).item()
        for value in compare(outputs["dense"], outputs["compact"])
    )
    result = {
        "model": args.model,
        "weights": str(args.weights),
        "device": str(device),
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "shape": {
            "depth": args.depth,
            "tokens": args.tokens,
            "channels": int(weight.shape[0]),
            "dtype": "bfloat16",
        },
        "comparison": {
            "bit_mismatch": int(comparison[0]),
            "max_abs": float(comparison[1]),
            "rms": float(comparison[2]),
            "dense_finite": bool(comparison[3]),
            "compact_finite": bool(comparison[4]),
        },
        "lowering": lowered_metrics,
        "warm_seconds": {
            name: {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "samples": values,
            }
            for name, values in timings.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
