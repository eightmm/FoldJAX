"""Benchmark the complete native Chai-JAX inference path.

The first iteration records cold compilation/execution. Later iterations use the
same mapped parameters and shapes, so they measure the executable-cache path.
Set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` to obtain meaningful allocator peak
figures rather than JAX's default reservation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np

import foldjax.models.chai.inference as inference
from foldjax.models.chai.models.pairformer import _triangle_attention_backend

INSTRUMENTED_FUNCTIONS = (
    "_embed_and_initialize",
    "_run_staged_trunk",
    "_compiled_diffusion_step",
    "_compiled_confidence_initialize",
    "_run_staged_confidence_block",
    "_compiled_confidence_project",
)


def _ready(value: Any) -> None:
    jax.block_until_ready(value)


def _memory_snapshot(memory: Mapping[str, int]) -> dict[str, int]:
    return {
        key: value for key, value in memory.items() if "byte" in key or "limit" in key
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--conformers", type=Path, required=True)
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--compile-cache", type=Path)
    parser.add_argument(
        "--use-esm-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include native ESM2 preprocessing in the benchmark",
    )
    parser.add_argument("--esm-model-path", type=Path)
    parser.add_argument("--esm-cache-directory", type=Path)
    args = parser.parse_args()

    config = inference.InferenceConfig(
        num_trunk_recycles=args.recycles,
        num_diffusion_timesteps=args.timesteps,
        num_diffusion_samples=args.samples,
        seed=args.seed,
        use_esm_embeddings=args.use_esm_embeddings,
        esm_model_path=args.esm_model_path,
        esm_cache_directory=(
            args.esm_cache_directory
            if args.esm_cache_directory is not None
            else Path.home() / ".cache/foldjax/chai/esm_embeddings"
        ),
        compilation_cache_dir=args.compile_cache,
    )
    prepared, assets = inference.prepare_inference(
        args.fasta,
        bundle_path=args.bundle,
        conformer_path=args.conformers,
        config=config,
    )
    components = inference.map_model_components(assets.bundle)
    phase_seconds: dict[str, list[float]] = defaultdict(list)
    phase_fingerprints: dict[str, list[list[float]]] = defaultdict(list)
    phase_memory: dict[str, list[dict[str, int | None]]] = defaultdict(list)

    originals = {name: getattr(inference, name) for name in INSTRUMENTED_FUNCTIONS}

    def instrument(name: str):
        original = originals[name]

        def measured(*positional: Any, **keywords: Any) -> Any:
            start = time.perf_counter()
            result = original(*positional, **keywords)
            _ready(result)
            phase_seconds[name].append(time.perf_counter() - start)
            memory = jax.devices()[0].memory_stats() or {}
            phase_memory[name].append(_memory_snapshot(memory))
            phase_fingerprints[name].append(
                [
                    float(np.asarray(leaf).reshape(-1)[0])
                    for leaf in jax.tree.leaves(result)
                    if np.asarray(leaf).size
                ]
            )
            return result

        return measured

    for name in originals:
        setattr(inference, name, instrument(name))

    iterations = []
    coordinate_fingerprints = []
    try:
        for _ in range(args.iterations):
            phase_starts = {name: len(values) for name, values in phase_seconds.items()}
            start = time.perf_counter()
            prediction = inference.execute_prepared_inference(
                prepared, components, config
            )
            _ready(prediction)
            elapsed = time.perf_counter() - start
            coordinate_fingerprints.append(float(prediction.atom_coords.reshape(-1)[0]))
            phase_totals = {
                name: sum(values[phase_starts.get(name, 0) :])
                for name, values in phase_seconds.items()
            }
            iterations.append({"seconds": elapsed, "phases": phase_totals})
    finally:
        for name, original in originals.items():
            setattr(inference, name, original)

    memory = jax.devices()[0].memory_stats() or {}
    print(
        json.dumps(
            {
                "device": str(jax.devices()[0]),
                "triangle_attention_backend": _triangle_attention_backend(),
                "model_size": prepared.model_size,
                "msa_shape": list(prepared.padded_inputs["msa_tokens"].shape),
                "config": {
                    "recycles": args.recycles,
                    "timesteps": args.timesteps,
                    "samples": args.samples,
                    "seed": args.seed,
                },
                "iterations": iterations,
                "coordinate_fingerprints": coordinate_fingerprints,
                "phase_fingerprints": phase_fingerprints,
                "phase_memory": phase_memory,
                "memory": {
                    key: value
                    for key, value in memory.items()
                    if "byte" in key or "limit" in key
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
