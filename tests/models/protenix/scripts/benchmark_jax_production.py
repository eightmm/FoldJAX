"""Benchmark Protenix JAX on the upstream production inference workload."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("outputs/compile_cache"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--warm-iters", type=int, default=3)
    parser.add_argument("--diffusion-scan", action="store_true")
    parser.add_argument("--sampler-scan", dest="sampler_scan", action="store_true")
    parser.add_argument("--no-sampler-scan", dest="sampler_scan", action="store_false")
    parser.set_defaults(sampler_scan=True)
    parser.add_argument("--denoiser-jit", action="store_true")
    parser.add_argument(
        "--diffusion-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa"),
        default="xla_jit",
    )
    parser.add_argument(
        "--trunk-triangle-attention-backend",
        choices=("xla", "xla_jit", "tokamax", "cueq", "cueq_jit"),
        default="cueq_jit",
    )
    parser.add_argument("--bf16-trunk", dest="bf16_trunk", action="store_true")
    parser.add_argument("--fp32-trunk", dest="bf16_trunk", action="store_false")
    parser.set_defaults(bf16_trunk=True)
    parser.add_argument(
        "--confidence-scan", dest="no_confidence_scan", action="store_false"
    )
    parser.add_argument(
        "--no-confidence-scan", dest="no_confidence_scan", action="store_true"
    )
    parser.set_defaults(no_confidence_scan=True)
    parser.add_argument(
        "--diffusion-efficient-fusion",
        dest="no_diffusion_efficient_fusion",
        action="store_false",
    )
    parser.add_argument(
        "--no-diffusion-efficient-fusion",
        dest="no_diffusion_efficient_fusion",
        action="store_true",
    )
    parser.set_defaults(no_diffusion_efficient_fusion=True)
    parser.add_argument("--full-depth-msa", dest="full_depth_msa", action="store_true")
    parser.add_argument(
        "--sample-msa-per-cycle", dest="full_depth_msa", action="store_false"
    )
    parser.set_defaults(full_depth_msa=True)
    parser.add_argument("--msa-row-bucket", type=int, default=64)
    parser.add_argument("--max-msa-padding-rows", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warm_iters <= 0:
        raise ValueError("warm_iters must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    from foldjax.models.protenix.bridge.weights_io import load_native_weights
    from foldjax.models.protenix.chunking import resolve_chunk_config
    from foldjax.models.protenix.data.static_io import load_static_feature_npz
    from foldjax.models.protenix.models.model import cast_trunk_params
    from foldjax.models.protenix.models.predict import protenix_predict_static
    from foldjax.models.protenix.models.trunk_blocks.msa import (
        pad_msa_features_to_bucket,
        sample_msa_cycle_features,
    )

    features = load_static_feature_npz(args.features)
    materialized_msa_rows = int(features["msa"].shape[-2])
    if args.msa_row_bucket > 0:
        features = pad_msa_features_to_bucket(
            features,
            bucket_size=args.msa_row_bucket,
            max_padding_rows=args.max_msa_padding_rows,
        )
    params = load_native_weights(args.weights)
    trunk_dtype = None
    if args.bf16_trunk:
        trunk_dtype = jnp.bfloat16
        params = cast_trunk_params(params, trunk_dtype)
    cycle_msa_features = None
    if not args.full_depth_msa:
        sampled = sample_msa_cycle_features(
            features,
            num_recycles=args.cycles,
            seed=args.seed,
        )
        cycle_msa_features = sampled or None
    n_token = int(features["restype"].shape[-2])
    chunks = resolve_chunk_config(
        n_token=n_token,
        num_samples=args.samples,
        policy="auto",
    )

    def run() -> dict:
        output = protenix_predict_static(
            params,
            features,
            key=jax.random.PRNGKey(args.seed),
            num_samples=args.samples,
            num_sampling_steps=args.steps,
            recycling_steps=args.cycles,
            use_pairformer_scan=False,
            use_confidence_scan=not args.no_confidence_scan,
            use_diffusion_scan=args.diffusion_scan,
            use_sampler_scan=args.sampler_scan,
            use_denoiser_jit=args.denoiser_jit,
            use_diffusion_efficient_fusion=(not args.no_diffusion_efficient_fusion),
            diffusion_attention_backend=args.diffusion_attention_backend,
            trunk_single_attention_backend="xla_jit",
            trunk_triangle_attention_backend=args.trunk_triangle_attention_backend,
            run_confidence=True,
            run_confidence_scores=True,
            triangle_mul_chunk_size=chunks.triangle_mul_chunk_size,
            triangle_att_q_chunk_size=chunks.triangle_att_q_chunk_size,
            single_att_q_chunk_size=chunks.single_att_q_chunk_size,
            token_q_chunk_size=chunks.token_q_chunk_size,
            diffusion_chunk_size=chunks.diffusion_chunk_size,
            matmul_precision="default",
            trunk_dtype=trunk_dtype,
            cycle_msa_features=cycle_msa_features,
        )
        return jax.block_until_ready(output)

    started = time.perf_counter()
    run()
    cold_seconds = time.perf_counter() - started
    warm_samples = []
    for _ in range(args.warm_iters):
        started = time.perf_counter()
        output = run()
        warm_samples.append(time.perf_counter() - started)
    warm_seconds = statistics.median(warm_samples)
    memory = jax.devices()[0].memory_stats() or {}
    metrics = {
        "backend": "jax",
        "contract": "upstream_production_workload",
        "checkpoint": "protenix_base_default_v1.0.0",
        "dtype": "fp32",
        "trunk_dtype": "bf16" if args.bf16_trunk else "fp32",
        "matmul_precision": "default_tf32_allowed",
        "persistent_compile_cache": str(args.cache.resolve()),
        "diffusion_attention_backend": args.diffusion_attention_backend,
        "trunk_attention_backends": args.trunk_triangle_attention_backend,
        "diffusion_scan": args.diffusion_scan,
        "sampler_scan": args.sampler_scan,
        "denoiser_jit": args.denoiser_jit,
        "diffusion_shared_vars_cache": True,
        "diffusion_efficient_fusion": not args.no_diffusion_efficient_fusion,
        "confidence": True,
        "confidence_scan": not args.no_confidence_scan,
        "seed": args.seed,
        "tokens": n_token,
        "atoms": int(features["atom_to_token_idx"].shape[-1]),
        "msa_rows_materialized": materialized_msa_rows,
        "msa_rows_executed": int(features["msa"].shape[-2]),
        "msa_row_bucket": args.msa_row_bucket,
        "msa_sampling": "full_depth" if args.full_depth_msa else "per_cycle_random",
        "msa_cycle_depths": (
            None
            if cycle_msa_features is None
            else [int(cycle["msa"].shape[-2]) for cycle in cycle_msa_features]
        ),
        "cycles": args.cycles,
        "num_steps": args.steps,
        "samples": args.samples,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "warm_seconds_samples": warm_samples,
        "peak_vram_gb": memory.get("peak_bytes_in_use", 0) / 1e9,
        "coordinate_checksum": float(output["coordinate"].sum()),
        "coordinate_shape": list(output["coordinate"].shape),
    }
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
