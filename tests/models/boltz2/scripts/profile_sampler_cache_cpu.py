"""Compile-only CPU probe for Boltz atom sampler cache multiplicity.

This models the released 1UBQ atom-cache shapes without loading checkpoints.
It reports logical cache bytes plus XLA's compiled argument/temporary sizes for
the former expanded cache and the compact broadcast-at-use cache.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.diffusion.atom import (
    _broadcast_window_s_terms,
)

LAYERS = 3
WINDOWS = 608 // 32
WINDOW_WIDTH = 32
CHANNELS = 128
TERMS_PER_LAYER = 6
BRANCHES = 2
SAMPLING_STEPS = 200
DTYPE = jnp.float32


def _cache_shapes(*, multiplicity: int, compact: bool):
    cache_windows = WINDOWS if compact else multiplicity * WINDOWS
    shape = (LAYERS, cache_windows, WINDOW_WIDTH, CHANNELS)
    return tuple(
        tuple(
            jax.ShapeDtypeStruct(shape, DTYPE) for _ in range(TERMS_PER_LAYER)
        )
        for _ in range(BRANCHES)
    )


def _logical_bytes(tree) -> int:
    return sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(tree))


def _compile(*, multiplicity: int, compact: bool):
    cache = _cache_shapes(multiplicity=multiplicity, compact=compact)
    activation = jax.ShapeDtypeStruct(
        (multiplicity * WINDOWS, WINDOW_WIDTH, CHANNELS),
        DTYPE,
    )

    def consume(cache_terms, a):
        def apply_branch(a_c, branch):
            def layer(a_l, layer_terms):
                if compact:
                    layer_terms = _broadcast_window_s_terms(
                        layer_terms,
                        multiplicity=multiplicity,
                        num_windows=WINDOWS,
                    )
                scale_a, bias_a, gate_a, scale_b, bias_b, gate_b = layer_terms
                a_l = jax.nn.sigmoid(scale_a) * a_l + bias_a
                a_l = a_l + jax.nn.sigmoid(gate_a) * jnp.tanh(a_l)
                a_l = jax.nn.sigmoid(scale_b) * a_l + bias_b
                a_l = a_l + jax.nn.sigmoid(gate_b) * jnp.tanh(a_l)
                return a_l, None

            a_c, _ = jax.lax.scan(layer, a_c, branch)
            return a_c

        def sample_step(a_c, _):
            for branch in cache_terms:
                a_c = apply_branch(a_c, branch)
            return a_c, None

        a, _ = jax.lax.scan(sample_step, a, None, length=SAMPLING_STEPS)
        return a

    lowered = jax.jit(consume).lower(cache, activation)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiled = lowered.compile()
    memory = compiled.memory_analysis()
    return {
        "cache_bytes": _logical_bytes(cache),
        "cache_mib": _logical_bytes(cache) / 2**20,
        "compiled_argument_bytes": memory.argument_size_in_bytes,
        "compiled_temporary_bytes": memory.temp_size_in_bytes,
        "compiled_output_bytes": memory.output_size_in_bytes,
        "stablehlo_broadcast_ops": stablehlo.count("stablehlo.broadcast_in_dim"),
        "stablehlo_chars": len(stablehlo),
    }


def main() -> None:
    result = {}
    for multiplicity in (1, 5):
        result[f"m{multiplicity}"] = {
            "expanded": _compile(multiplicity=multiplicity, compact=False),
            "compact": _compile(multiplicity=multiplicity, compact=True),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
