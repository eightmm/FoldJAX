"""Profile Boltz-2's scoped BF16 atom attention on one real GPU.

Run one backend per fresh process so XLA executable and allocator state from a
previous arm cannot contaminate latency or peak-memory measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.bridge.native import load_features_npz, load_params
from foldjax.models.boltz2.models.diffusion.diffusion_transformer import (
    _no_proj_qblock,
)
from foldjax.models.boltz2.models.primitives.attention_backend import (
    tokamax_dot_product_attention,
)
from foldjax.models.boltz2.models.trunk_blocks.input_embedder import (
    input_embedder_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    _cast_float_feats,
    _cast_params,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("xla", "tokamax", "triton"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=(2, 32, 128, 256))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _memory_mib() -> dict[str, float]:
    stats = jax.devices()[0].memory_stats() or {}
    return {
        key: float(stats[key]) / 1024**2
        for key in ("bytes_in_use", "peak_bytes_in_use")
        if key in stats
    }


def _time(function, args: tuple[object, ...], *, warmup: int, repeats: int):
    started = time.perf_counter()
    output = jax.block_until_ready(function(*args))
    cold_ms = (time.perf_counter() - started) * 1000
    for _ in range(warmup):
        output = jax.block_until_ready(function(*args))
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = jax.block_until_ready(function(*args))
        timings.append((time.perf_counter() - started) * 1000)
    return output, cold_ms, timings


def _primitive_inputs(windows: int):
    rng = np.random.default_rng(71 + windows)
    q = jnp.asarray(
        rng.standard_normal((windows, 32, 4, 32)), dtype=jnp.bfloat16
    )
    k_np = rng.standard_normal((windows, 128, 4, 32))
    v_np = rng.standard_normal((windows, 128, 4, 32))
    bias_np = rng.standard_normal((windows, 4, 32, 128)) * 0.1
    mask_np = np.ones((windows, 128), dtype=bool)
    mask_np[0] = False
    k_np[0] = np.nan
    v_np[0] = 1e4
    bias_np[0] = np.nan
    if windows > 1:
        mask_np[1:, 96:] = False
        k_np[1:, 96:] = np.nan
        # Masked values still participate in the final `probability @ value`
        # contraction as exact zero multipliers. Keep this adversarial but
        # finite: the historical XLA route also yields NaN for 0 * Inf, so
        # demanding otherwise would test a new contract rather than parity.
        v_np[1:, 96:] = 1e4
        bias_np[1:, :, :, 96:] = np.nan
    k = jnp.asarray(k_np, dtype=jnp.bfloat16)
    v = jnp.asarray(v_np, dtype=jnp.bfloat16)
    bias = jnp.asarray(bias_np, dtype=jnp.bfloat16)
    mask = jnp.asarray(mask_np)
    return q, k, v, bias, mask


def _primitive(mode: str, q, k, v, bias, mask):
    if mode == "xla":
        return _no_proj_qblock(
            q,
            k,
            v,
            bias,
            mask[:, None, None, :],
            jnp.sqrt(jnp.asarray(q.shape[-1], dtype=jnp.float32)),
        )
    return tokamax_dot_product_attention(
        q,
        k,
        v,
        bias,
        mask,
        scale=float(q.shape[-1]) ** -0.5,
        backend=mode,
    )


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    # NumPy NPZ does not preserve ml_dtypes.bfloat16 metadata.  The benchmark
    # validates the device result dtype before this point and stores the exact
    # BF16 values losslessly as float32 for cross-process comparison.
    actual = actual.astype(np.float32)
    expected = expected.astype(np.float32)
    finite = np.isfinite(actual) & np.isfinite(expected)
    delta = np.abs(actual - expected)
    if np.any(finite):
        left = actual[finite].astype(np.float64)
        right = expected[finite].astype(np.float64)
        correlation = float(np.corrcoef(left, right)[0, 1])
        denominator = float(np.sqrt(np.mean(right**2)))
        nrmse = float(np.sqrt(np.mean((left - right) ** 2)) / denominator)
    else:
        correlation = float("nan")
        nrmse = float("nan")
    return {
        "exact": bool(np.array_equal(actual, expected, equal_nan=True)),
        "max_abs": float(np.nanmax(delta)),
        "correlation": correlation,
        "nrmse": nrmse,
        "finite_pattern_equal": bool(
            np.array_equal(np.isfinite(actual), np.isfinite(expected))
        ),
    }


def main() -> None:
    args = _args()
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError(f"GPU required, got {device}")
    capability = getattr(device, "compute_capability", None)
    if capability is not None and float(capability) < 8.0:
        raise RuntimeError(f"Triton requires compute capability >=8.0: {capability}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_arrays: dict[str, np.ndarray] = {}
    primitive_results: dict[str, object] = {}
    for windows in args.windows:
        inputs = _primitive_inputs(windows)
        runner = jax.jit(lambda *values: _primitive(args.mode, *values))
        output, cold_ms, timings = _time(
            runner, inputs, warmup=args.warmup, repeats=args.repeats
        )
        if output.dtype != jnp.bfloat16:
            raise AssertionError(
                f"primitive output dtype is {output.dtype}, expected bfloat16"
            )
        output_np = np.asarray(output, dtype=np.float32)
        if not np.isfinite(output_np).all():
            raise AssertionError(f"nonfinite primitive output for windows={windows}")
        if np.any(output_np[0] != 0):
            raise AssertionError(
                f"empty window is not exact zero for windows={windows}"
            )
        name = f"primitive_w{windows}"
        output_arrays[name] = output_np
        result: dict[str, object] = {
            "cold_ms": cold_ms,
            "warm_median_ms": statistics.median(timings),
            "warm_times_ms": timings,
            "finite": True,
            "empty_exact_zero": True,
        }
        if args.reference_dir is not None:
            with np.load(args.reference_dir / "arrays.npz", allow_pickle=False) as ref:
                result["vs_xla"] = _comparison(output_np, np.asarray(ref[name]))
        primitive_results[str(windows)] = result

    params = load_params(args.weights)
    features = load_features_npz(args.features)
    trunk_params = _cast_params(params["trunk"], jnp.bfloat16)
    trunk_features = _cast_float_feats(features, jnp.bfloat16)
    input_runner = jax.jit(
        lambda p, f: input_embedder_forward(
            p, f, attention_backend=args.mode
        )
    )
    input_output, input_cold_ms, input_timings = _time(
        input_runner,
        (trunk_params["input_embedder"], trunk_features),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    if input_output.dtype != jnp.bfloat16:
        raise AssertionError(
            f"input-embedder output dtype is {input_output.dtype}, expected bfloat16"
        )
    input_np = np.asarray(input_output, dtype=np.float32)
    if not np.isfinite(input_np).all():
        raise AssertionError("nonfinite real input-embedder output")
    output_arrays["input_embedder"] = input_np
    input_result: dict[str, object] = {
        "shape": list(input_np.shape),
        "cold_ms": input_cold_ms,
        "warm_median_ms": statistics.median(input_timings),
        "warm_times_ms": input_timings,
        "finite": True,
    }
    if args.reference_dir is not None:
        with np.load(args.reference_dir / "arrays.npz", allow_pickle=False) as ref:
            input_result["vs_xla"] = _comparison(
                input_np, np.asarray(ref["input_embedder"])
            )

    np.savez(args.output.parent / "arrays.npz", **output_arrays)
    report = {
        "mode": args.mode,
        "device": str(device),
        "compute_capability": capability,
        "dtype": "bfloat16",
        "primitive": primitive_results,
        "input_embedder": input_result,
        "memory_mib": _memory_mib(),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
