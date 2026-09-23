"""Compiled-program fingerprints for Protenix's two 2-D triangle-attention entries.

This port reaches the 2-D Fold-CP ring through two modules: the CP dispatcher
(`triangle/triangle_attention_cp.py`, which every Pairformer imports) and the
serial module's own context-parallel branch (`triangle/triangle.py`,
`_triangle_attention_ring_2d`, reached only by a direct call of that module's
`triangle_attention` under a 2-D mesh). Both read `triangle_attention_grid`,
and with it omitted or `ring` both must compile the program they compiled
before the option existed.

Run it against two source trees with the same interpreter, devices and flags
and compare the output line by line:

    JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    PYTHONPATH=<tree>/src python <this file> [--grid ring|gather]

`--grid ring` must print the lines the omitted option prints; `--grid gather`
prints the gather programs, which are expected to differ on the mesh arms and
not on the serial ones. A tree without the scope can still run the default,
which is how the lines are compared against the parent commit.

The hash is over the HLO text with source metadata stripped, as in
`tests/models/boltz2/scripts/cp_ring_tile_fingerprints.py`; `temp_size_in_bytes`
is printed beside it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import re

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import context_parallel
from foldjax.models.protenix.models.primitives.attention import AttentionParams
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.triangle import triangle as serial_triangle
from foldjax.models.protenix.models.triangle import (
    triangle_attention_cp as cp_triangle,
)
from foldjax.models.protenix.models.triangle.triangle import TriangleAttentionParams

DEVICES = 4
# Divisible by the 2x2 grid's side, so no arm measures a padded remainder.
TOKENS, CZ, HEADS = 12, 8, 2


def _fingerprint(text: str) -> str:
    text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
    text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def _report(name: str, jitted, *args) -> None:
    lowered = jitted.lower(*args)
    text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    temp = lowered.compile().memory_analysis().temp_size_in_bytes
    print(f"{name} {_fingerprint(text)[:32]} temp={temp}")


def _grid_scope(grid: str | None):
    if grid is None:
        return contextlib.nullcontext()
    from foldjax.models._cp_attention import triangle_attention_grid_scope

    return triangle_attention_grid_scope(grid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        choices=("ring", "gather"),
        default=None,
        help="enter triangle_attention_grid_scope with this value (default: "
        "no scope, the omitted option)",
    )
    args = parser.parse_args(argv)
    rng = np.random.default_rng(20260923)

    def arr(*shape, scale=0.5):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    def linear(out_channels, in_channels):
        return LinearParams(weight=arr(out_channels, in_channels), bias=None)

    params = TriangleAttentionParams(
        layer_norm=LayerNormParams(weight=arr(CZ) * 0.1 + 1.0, bias=arr(CZ) * 0.1),
        linear=linear(HEADS, CZ),
        attention=AttentionParams(
            linear_q=linear(CZ, CZ),
            linear_k=linear(CZ, CZ),
            linear_v=linear(CZ, CZ),
            linear_o=linear(CZ, CZ),
            linear_g=linear(CZ, CZ),
        ),
    )
    z = arr(TOKENS, TOKENS, CZ)
    keep = rng.random(TOKENS) > 0.2
    keep[0] = True
    pair_mask = jnp.asarray(keep[:, None] & keep[None, :]).astype(jnp.float32)

    assert jax.device_count() == DEVICES, jax.devices()
    entries = (
        ("serial_module", serial_triangle.triangle_attention),
        ("cp_module", cp_triangle.triangle_attention),
    )
    with context_parallel(DEVICES, layout="2d"), _grid_scope(args.grid):
        # A fresh closure per arm: `jax.jit` keys its cache on the callable.
        for name, entry in entries:
            for direction in (True, False):
                def one(x, mask, params, entry=entry, direction=direction):
                    return entry(
                        x, mask, params, num_heads=HEADS, starting=direction,
                        q_chunk_size=2,
                    )

                _report(
                    f"{name}_ring_{'start' if direction else 'end'}",
                    jax.jit(one),
                    z,
                    pair_mask,
                    params,
                )

    # The serial program both entries run with no mesh: what a change to the
    # 2-D dispatch must not touch. `xla` explicitly -- the serial default is
    # the cuEquivariance kernel, which this host cannot load.
    with _grid_scope(args.grid):
        for name, entry in entries:
            for direction in (True, False):
                def one(x, mask, params, entry=entry, direction=direction):
                    return entry(
                        x, mask, params, num_heads=HEADS, starting=direction,
                        attention_backend="xla",
                    )

                _report(
                    f"{name}_serial_{'start' if direction else 'end'}",
                    jax.jit(one),
                    z,
                    pair_mask,
                    params,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
