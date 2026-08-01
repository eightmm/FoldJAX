"""Benchmark cached, compiled Boltz-JAX inference in a clean process."""

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

from foldjax.models.boltz2.bridge.native import load_params
from foldjax.models.boltz2.models.predict import boltz2_predict


def _load_features(path: Path) -> tuple[dict[str, np.ndarray], str]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}, path.stem

    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    feats = {
        key: value.detach().cpu().numpy()
        for key, value in obj.items()
        if not key.startswith("_") and torch.is_tensor(value)
    }
    return feats, str(obj.get("_record_id", path.stem))


def _memory_stats() -> dict[str, float]:
    stats = jax.devices()[0].memory_stats() or {}
    return {
        key: float(stats[key]) / 1024**2
        for key in ("bytes_in_use", "peak_bytes_in_use")
        if key in stats
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/native_weights/boltz2_conf")
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument(
        "--triangle-backend",
        choices=("xla", "pallas", "tokamax", "cueq"),
        default="cueq",
    )
    parser.add_argument("--glu-backend", choices=("xla", "tokamax"), default="xla")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument("--confidence-sequentially", action="store_true")
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
    compute_dtype = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }[args.dtype]

    predict = jax.jit(
        partial(
            boltz2_predict,
            recycling_steps=args.recycling,
            num_sampling_steps=args.steps,
            augmentation=False,
            run_confidence=not args.sampling_only,
            run_distogram=not args.sampling_only,
            run_bfactor=not args.sampling_only,
            multiplicity=args.multiplicity,
            compute_dtype=compute_dtype,
            use_scan=True,
            return_pair_chains_iptm=False,
            triangle_backend=args.triangle_backend,
            glu_backend=args.glu_backend,
            confidence_sequentially=args.confidence_sequentially,
            recompute_nonpolymer_frames=bool(np.any(feats_np["mol_type"] == 3)),
        )
    )

    def call(seed: int):
        return jax.block_until_ready(predict(params, feats, jax.random.PRNGKey(seed)))

    start = time.perf_counter()
    out = call(0)
    cold_ms = (time.perf_counter() - start) * 1000
    for seed in range(1, args.warmup):
        out = call(seed)

    times_ms = []
    for seed in range(args.warmup, args.warmup + args.iters):
        start = time.perf_counter()
        out = call(seed)
        times_ms.append((time.perf_counter() - start) * 1000)

    coords = np.asarray(out["sample_atom_coords"])
    payload = {
        "record_id": record_id,
        "features": str(args.features),
        "dtype": args.dtype,
        "steps": args.steps,
        "recycling": args.recycling,
        "multiplicity": args.multiplicity,
        "triangle_backend": args.triangle_backend,
        "glu_backend": args.glu_backend,
        "sampling_only": args.sampling_only,
        "confidence_sequentially": args.confidence_sequentially,
        "n_tokens": int(feats_np["token_pad_mask"].shape[1]),
        "n_atoms": int(feats_np["atom_pad_mask"].shape[1]),
        "cold_ms": cold_ms,
        "warm_median_ms": statistics.median(times_ms),
        "warm_mean_ms": statistics.mean(times_ms),
        "warm_times_ms": times_ms,
        "memory_mib": _memory_stats(),
        "coords_finite": bool(np.isfinite(coords).all()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(args.output.with_suffix(".coords.npy"), coords)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
