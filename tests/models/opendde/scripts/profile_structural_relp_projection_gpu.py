"""GPU A/B probe for OpenDDE structural relative-position projection.

This is deliberately a leaf-level experiment: it answers whether XLA already
fuses away the historical ``[N, N, 139]`` one-hot before production code is
changed.  The direct arm selects the four non-zero projection columns from the
released 139-channel layout.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.primitives.primitives import LinearParams
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    RelativePositionParams,
    relative_position_encoding,
    relative_position_encoding_from_features,
    relative_position_features,
)


def _features(n_token: int) -> dict[str, jax.Array]:
    index = jnp.arange(n_token, dtype=jnp.int32)
    chain_width = max(1, n_token // 4)
    asym_id = index // chain_width
    return {
        "asym_id": asym_id,
        "residue_index": index,
        "entity_id": asym_id % 2,
        "sym_id": asym_id % 3,
        "token_index": index,
    }


def _dense(features: dict[str, jax.Array], weight: jax.Array) -> jax.Array:
    params = RelativePositionParams(LinearParams(weight=weight, bias=None))
    return relative_position_encoding(relative_position_features(features), params)


def _direct(features: dict[str, jax.Array], weight: jax.Array) -> jax.Array:
    params = RelativePositionParams(LinearParams(weight=weight, bias=None))
    return relative_position_encoding_from_features(features, params)


def _memory_dict(analysis: object) -> dict[str, int]:
    return {
        name: int(getattr(analysis, name))
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "generated_code_size_in_bytes",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1886)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    rng = np.random.default_rng(20260831)
    weight = jnp.asarray(
        rng.normal(size=(args.channels, 139)).astype(np.float32) / 16.0
    )
    features = _features(args.tokens)
    results: dict[str, object] = {
        "device": jax.devices()[0].device_kind,
        "tokens": args.tokens,
        "channels": args.channels,
    }
    for name, fn in (("dense", _dense), ("direct", _direct)):
        started = time.perf_counter()
        compiled = jax.jit(fn).lower(features, weight).compile()
        record: dict[str, object] = {
            "compile_seconds": time.perf_counter() - started,
            "memory": _memory_dict(compiled.memory_analysis()),
        }
        if args.execute:
            started = time.perf_counter()
            output = compiled(features, weight)
            output.block_until_ready()
            record["first_execute_seconds"] = time.perf_counter() - started
            record["checksum"] = float(jnp.sum(output[:1, :1]))
            del output
        results[name] = record
        jax.clear_caches()
    if args.execute:
        max_abs_error = jax.jit(
            lambda feature_values, projection: jnp.max(
                jnp.abs(
                    _dense(feature_values, projection)
                    - _direct(feature_values, projection)
                )
            )
        )(features, weight)
        results["parity_max_abs_error"] = float(max_abs_error)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
