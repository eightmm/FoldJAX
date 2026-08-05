"""Time and size one triangle multiplication against its row-block size.

The blocked contraction reads the whole of `b` once per output block, so its
memory traffic is `ceil(n / block) * n^2 * c` -- cubic in tokens once the block
size stops growing with them. `_row_block` caps every block at 64 rows, which
makes 24 passes over `b` at 1,531 tokens where the chunk policy asked for 3.

That cap was chosen on peak memory alone; the note beside it records 9,809 ->
9,652 -> 9,628 MiB at 256 -> 128 -> 64 rows and says nothing about time. This
measures the other axis, so the trade can be made on both numbers instead of
one.

    python -m bench.triangle_block_sweep --tokens 1531 --blocks 64,128,256,512,0
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1531)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument(
        "--blocks",
        default="64,128,256,512,0",
        help="row blocks to sweep; 0 means no blocking",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.models.triangle import triangle as tri

    dtype = jnp.dtype(args.dtype)
    n, c = args.tokens, args.channels
    key = jax.random.PRNGKey(0)
    z_key, b_key = jax.random.split(key)
    z_norm = jax.random.normal(z_key, (1, n, n, c), dtype=dtype)
    b = jax.random.normal(b_key, (1, n, n, c), dtype=dtype)
    mask = jnp.ones((1, n, n, 1), dtype=dtype)
    weight = jax.random.normal(b_key, (c, c), dtype=dtype) / c**0.5

    def project_a(z_slice, mask_slice):
        return mask_slice * jnp.einsum("...d,de->...e", z_slice, weight)

    def run(block: int | None):
        return tri._triangle_contract(project_a, z_norm, mask, b, "outgoing", block)

    rows = []
    for token in args.blocks.split(","):
        block = int(token) or None
        compiled = jax.jit(run, static_argnums=0)
        # Compile and warm before timing: a cold call reports XLA, not the
        # kernel, and the whole point here is the kernel.
        jax.block_until_ready(compiled(block))
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            jax.block_until_ready(compiled(block))
            samples.append(time.perf_counter() - start)
        analysis = compiled.lower(block).compile().memory_analysis()
        passes = 1 if block is None else -(-n // block)
        rows.append(
            {
                "block": block,
                "passes_over_b": passes,
                "best_ms": round(min(samples) * 1e3, 2),
                "median_ms": round(sorted(samples)[len(samples) // 2] * 1e3, 2),
                "temp_mib": round(analysis.temp_size_in_bytes / 2**20, 1),
                "output_mib": round(analysis.output_size_in_bytes / 2**20, 1),
            }
        )
        print(json.dumps(rows[-1]))

    baseline = next((row for row in rows if row["block"] == 64), None)
    print()
    print(f"tokens={n} channels={c} dtype={dtype.name}")
    header = f"| {'block':>6} | {'passes':>6} | {'ms':>8} |"
    print(header + f" {'vs 64':>7} | {'temp MiB':>9} |")
    print("|" + "---|" * 5)
    for row in rows:
        speedup = (
            f"{baseline['best_ms'] / row['best_ms']:.2f}x"
            if baseline and row["best_ms"]
            else "-"
        )
        label = row["block"] if row["block"] is not None else "none"
        print(
            f"| {label:>6} | {row['passes_over_b']:>6} | {row['best_ms']:>8.2f} "
            f"| {speedup:>7} | {row['temp_mib']:>9.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
