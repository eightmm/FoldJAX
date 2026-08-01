#!/usr/bin/env python3
"""Benchmark the complete cached JAX denoiser against Chai TorchScript."""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from foldjax.models.chai.models.diffusion_denoiser import (
    full_diffusion_denoiser,
    map_full_diffusion_denoiser,
)


def _default_component() -> Path | None:
    asset_dir = os.environ.get("CHAI_JAX_OFFICIAL_ASSET_DIR")
    return Path(asset_dir).expanduser() / "diffusion_module.pt" if asset_dir else None


def _inputs(tokens: int = 256, samples: int = 1) -> tuple[np.ndarray, ...]:
    if tokens != 256:
        raise ValueError("the official benchmark currently targets forward_256")
    rng = np.random.default_rng(20260714)
    atoms = 23 * tokens
    query = 32
    keys = 128
    blocks = atoms // query
    block_h = np.arange(atoms, dtype=np.int64).reshape(blocks, query)
    starts = block_h[:, :1] + (query - keys) // 2
    block_w = (starts + np.arange(keys, dtype=np.int64)) % atoms
    atom_token = (np.arange(atoms, dtype=np.int64) // 23)[None]

    def normal(shape: tuple[int, ...], scale: float = 0.1) -> np.ndarray:
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    return (
        normal((1, tokens, 384)),
        normal((1, tokens, tokens, 256), 0.02),
        normal((1, tokens, 384)),
        normal((1, tokens, tokens, 256), 0.02),
        normal((1, atoms, 128)),
        np.zeros((1, blocks, query, keys, 16), np.float32),
        np.ones((1, atoms), np.bool_),
        np.ones((1, blocks, query, keys), np.bool_),
        np.ones((1, tokens), np.bool_),
        block_h,
        block_w,
        normal((1, samples, atoms, 3)),
        np.full((1, samples), 1.5, np.float32),
        atom_token,
    )


def _sync(value):
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), value)


def _device_peak_bytes(device) -> int | None:
    stats = device.memory_stats()
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", type=Path, default=_default_component())
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument(
        "--bf16-component",
        action="append",
        choices=(
            "all",
            "conditioning",
            "atom_encoder",
            "transformer",
            "atom_decoder",
            "top_level",
        ),
        default=[],
    )
    parser.add_argument("--max-nrmse", type=float, default=0.005)
    args = parser.parse_args()

    if args.component is None:
        parser.error(
            "--component is required unless CHAI_JAX_OFFICIAL_ASSET_DIR is set"
        )
    if not args.component.is_file():
        parser.error(f"official diffusion component is unavailable: {args.component}")

    numpy_inputs = _inputs(samples=args.samples)
    module = torch.jit.load(str(args.component), map_location="cuda").eval()
    state = {
        key: value.detach().cpu().numpy() for key, value in module.state_dict().items()
    }
    torch_inputs = tuple(torch.from_numpy(value).cuda() for value in numpy_inputs)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        module.forward_256(*torch_inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            expected = module.forward_256(*torch_inputs)
        torch.cuda.synchronize()
    torch_seconds = (time.perf_counter() - start) / args.repeats
    torch_peak = torch.cuda.max_memory_allocated()
    expected_numpy = expected.float().cpu().numpy()
    del expected, module, torch_inputs
    gc.collect()
    torch.cuda.empty_cache()

    params = map_full_diffusion_denoiser(state)
    if "all" in args.bf16_component:
        params = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    else:
        replacements = {
            name: jax.tree.map(
                lambda value: value.astype(jnp.bfloat16), getattr(params, name)
            )
            for name in args.bf16_component
        }
        params = params._replace(**replacements)
    jax_inputs = tuple(jnp.asarray(value) for value in numpy_inputs)
    compiled = jax.jit(
        full_diffusion_denoiser,
        static_argnames=("query_chunk_size",),
    )
    device = jax.devices()[0]
    before_peak = _device_peak_bytes(device)
    start = time.perf_counter()
    actual = _sync(
        compiled(
            *jax_inputs,
            params,
            query_chunk_size=args.query_chunk_size,
        )
    )
    cold_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(args.repeats):
        actual = _sync(
            compiled(
                *jax_inputs,
                params,
                query_chunk_size=args.query_chunk_size,
            )
        )
    jax_seconds = (time.perf_counter() - start) / args.repeats
    after_peak = _device_peak_bytes(device)

    actual_numpy = np.asarray(actual)
    difference = actual_numpy - expected_numpy
    nrmse = float(
        np.sqrt(np.mean(difference**2))
        / max(np.sqrt(np.mean(expected_numpy**2)), 1e-12)
    )
    correlation = float(np.corrcoef(actual_numpy.ravel(), expected_numpy.ravel())[0, 1])
    print(f"torch_warm_seconds={torch_seconds:.6f}")
    print(f"jax_cold_seconds={cold_seconds:.6f}")
    print(f"jax_warm_seconds={jax_seconds:.6f}")
    print(f"speedup={torch_seconds / jax_seconds:.3f}")
    print(f"torch_peak_bytes={torch_peak}")
    print(f"jax_peak_bytes_before={before_peak}")
    print(f"jax_peak_bytes_after={after_peak}")
    print(f"nrmse={nrmse:.8f}")
    print(f"correlation={correlation:.8f}")
    print(f"max_abs={np.max(np.abs(difference)):.8f}")
    if not np.isfinite(nrmse) or nrmse > args.max_nrmse:
        raise SystemExit(
            f"diffusion parity failed: nrmse={nrmse:.8f} > {args.max_nrmse}"
        )


if __name__ == "__main__":
    main()
