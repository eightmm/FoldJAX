"""Profile separately compiled Boltz-JAX trunk and diffusion sampler."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_warm_predict import _load_features, _memory_stats

from foldjax.models.boltz2.bridge.native import load_params
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    _cast_float_feats,
    _cast_params,
    boltz2_sample_forward,
    boltz2_trunk_forward,
)


def _timed(call, iters: int) -> tuple[object, float, list[float]]:
    start = time.perf_counter()
    out = call(0)
    cold_ms = (time.perf_counter() - start) * 1000
    times = []
    for seed in range(1, iters + 1):
        start = time.perf_counter()
        out = call(seed)
        times.append((time.perf_counter() - start) * 1000)
    return out, cold_ms, times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/native_weights/boltz2_conf")
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--triangle-backend",
        choices=("xla", "pallas", "tokamax", "cueq"),
        default="cueq",
    )
    parser.add_argument("--glu-backend", choices=("xla", "tokamax"), default="xla")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--compile-cache", type=Path, default=Path(".cache/jax_compilation")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")
    args.compile_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compile_cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    feats_np, record_id = _load_features(args.features)
    feats = {key: jnp.asarray(value) for key, value in feats_np.items()}
    params = load_params(args.weights)
    trunk_params = _cast_params(params["trunk"], jnp.bfloat16)
    trunk_feats = _cast_float_feats(feats, jnp.bfloat16)

    trunk_fn = jax.jit(
        partial(
            boltz2_trunk_forward,
            recycling_steps=args.recycling,
            use_scan=True,
            subsample_msa=True,
            num_subsampled_msa=1024,
            triangle_backend=args.triangle_backend,
            chunk_size=args.chunk_size,
            glu_backend=args.glu_backend,
        )
    )

    def trunk_call(_seed: int):
        return jax.block_until_ready(trunk_fn(trunk_params, trunk_feats))

    trunk, trunk_cold_ms, trunk_times = _timed(trunk_call, args.iters)

    sampler_fn = jax.jit(
        partial(
            boltz2_sample_forward,
            recycling_steps=args.recycling,
            num_sampling_steps=args.steps,
            augmentation=False,
            multiplicity=args.multiplicity,
            compute_dtype=jnp.bfloat16,
            use_scan=True,
            triangle_backend=args.triangle_backend,
            glu_backend=args.glu_backend,
        )
    )

    def sampler_call(seed: int):
        out = sampler_fn(params, feats, jax.random.PRNGKey(seed), trunk=trunk)
        return jax.block_until_ready(out)

    sampler_out, sampler_cold_ms, sampler_times = _timed(sampler_call, args.iters)
    trunk_median = statistics.median(trunk_times)
    sampler_median = statistics.median(sampler_times)
    payload = {
        "record_id": record_id,
        "features": str(args.features),
        "dtype": "bfloat16-trunk-fp32-diffusion",
        "steps": args.steps,
        "recycling": args.recycling,
        "multiplicity": args.multiplicity,
        "triangle_backend": args.triangle_backend,
        "chunk_size": args.chunk_size,
        "glu_backend": args.glu_backend,
        "n_tokens": int(feats_np["token_pad_mask"].shape[1]),
        "n_atoms": int(feats_np["atom_pad_mask"].shape[1]),
        "trunk": {
            "cold_ms": trunk_cold_ms,
            "warm_median_ms": trunk_median,
            "warm_times_ms": trunk_times,
        },
        "sampler": {
            "cold_ms": sampler_cold_ms,
            "warm_median_ms": sampler_median,
            "warm_times_ms": sampler_times,
        },
        "split_warm_sum_ms": trunk_median + sampler_median,
        "memory_mib": _memory_stats(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(
        args.output.with_suffix(".coords.npy"),
        np.asarray(sampler_out["sample_atom_coords"]),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
