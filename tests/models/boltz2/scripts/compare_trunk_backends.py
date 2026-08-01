#!/usr/bin/env python3
"""Compare deterministic Boltz-JAX trunk outputs across triangle backends."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_warm_predict import _load_features

from foldjax.models.boltz2.bridge.native import load_params
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    _cast_float_feats,
    _cast_params,
    boltz2_trunk_forward,
)


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate = candidate.astype(np.float64)
    reference = reference.astype(np.float64)
    delta = candidate - reference
    rmse = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(reference * reference)))
    flat_candidate = candidate.reshape(-1)
    flat_reference = reference.reshape(-1)
    cosine = float(
        np.dot(flat_candidate, flat_reference)
        / (np.linalg.norm(flat_candidate) * np.linalg.norm(flat_reference))
    )
    return {
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, np.finfo(np.float64).tiny),
        "max_abs": float(np.max(np.abs(delta))),
        "cosine_similarity": cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/native_weights/boltz2_conf")
    )
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument(
        "--candidate", choices=("pallas", "tokamax", "cueq"), default="pallas"
    )
    parser.add_argument(
        "--reference-triangle",
        choices=("xla", "pallas", "tokamax", "cueq"),
        default="xla",
    )
    parser.add_argument("--reference-glu", choices=("xla", "tokamax"), default="xla")
    parser.add_argument("--candidate-glu", choices=("xla", "tokamax"), default="xla")
    parser.add_argument(
        "--reference-triangle-multiplication",
        choices=("xla", "cueq"),
        default="xla",
    )
    parser.add_argument(
        "--candidate-triangle-multiplication",
        choices=("xla", "cueq"),
        default="xla",
    )
    parser.add_argument("--torch-reference", type=Path)
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
    params = _cast_params(params["trunk"], jnp.bfloat16)
    feats = _cast_float_feats(feats, jnp.bfloat16)

    def make_fn(triangle_backend: str, glu_backend: str):
        return jax.jit(
            partial(
                boltz2_trunk_forward,
                recycling_steps=args.recycling,
                use_scan=True,
                subsample_msa=True,
                num_subsampled_msa=1024,
                triangle_backend=triangle_backend,
                chunk_size=128,
                glu_backend=glu_backend,
            )
        )

    os.environ["BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"] = (
        args.reference_triangle_multiplication
    )
    reference = jax.block_until_ready(
        make_fn(args.reference_triangle, args.reference_glu)(params, feats)
    )
    os.environ["BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"] = (
        args.candidate_triangle_multiplication
    )
    candidate = jax.block_until_ready(
        make_fn(args.candidate, args.candidate_glu)(params, feats)
    )
    payload = {
        "record_id": record_id,
        "features": str(args.features),
        "reference": {
            "triangle_backend": args.reference_triangle,
            "triangle_multiplication_backend": (
                args.reference_triangle_multiplication
            ),
            "glu_backend": args.reference_glu,
        },
        "candidate": {
            "triangle_backend": args.candidate,
            "triangle_multiplication_backend": (
                args.candidate_triangle_multiplication
            ),
            "glu_backend": args.candidate_glu,
        },
        "metrics": {
            key: _metrics(np.asarray(candidate[key]), np.asarray(reference[key]))
            for key in reference
        },
    }
    if args.torch_reference is not None:
        torch_reference = np.load(args.torch_reference)
        payload["torch_metrics"] = {
            name: {
                "reference": _metrics(
                    np.asarray(reference[name]), torch_reference[name]
                ),
                "candidate": _metrics(
                    np.asarray(candidate[name]), torch_reference[name]
                ),
            }
            for name in ("s", "z")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
