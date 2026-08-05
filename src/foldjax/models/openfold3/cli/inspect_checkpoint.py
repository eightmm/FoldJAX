"""Inspect an OpenFold3 checkpoint without assuming its layout.

Answers the questions a port has to settle before mapping anything: what the
top-level structure is, what the real dimensions are, how many blocks each stack
has, and which triangular-multiplication layout the weights use.

    openfold3-jax-inspect-checkpoint weights.safetensors
    openfold3-jax-inspect-checkpoint weights.pt --depth 3 --grep pairformer
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from foldjax.models.openfold3.bridge.checkpoint import (
    describe,
    detect_fused_tri_mul,
    iter_shapes,
    load_checkpoint,
)


def count_blocks(state: dict[str, np.ndarray], root: str) -> int | None:
    """Return the number of ``root.N`` blocks, or ``None`` if ``root`` is absent."""
    indices = set()
    for key in state:
        if f"{root}." in key:
            tail = key.split(f"{root}.", 1)[1].split(".", 1)[0]
            if tail.isdigit():
                indices.add(int(tail))
    return len(indices) or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openfold3-jax-inspect-checkpoint",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--depth", type=int, default=2, help="prefix depth for the summary"
    )
    parser.add_argument(
        "--grep", default=None, help="print key/shape pairs containing this substring"
    )
    parser.add_argument(
        "--limit", type=int, default=40, help="max key/shape lines to print"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state = load_checkpoint(args.checkpoint)

    total = sum(int(np.size(value)) for value in state.values())
    print(f"tensors: {len(state)}   parameters: {total:,}")

    layout = detect_fused_tri_mul(state)
    label = {True: "fused", False: "unfused", None: "not found"}[layout]
    print(f"triangular multiplication: {label} (both layouts are supported)")

    print(f"\nstructure (depth {args.depth}):")
    for prefix, count in describe(state, depth=args.depth).items():
        print(f"  {prefix:<48} {count:>6} tensors")

    print("\nblock counts:")
    roots = (
        "pairformer_stack.blocks",
        "msa_module.blocks",
        "template_pair_stack.blocks",
        "diffusion_transformer.blocks",
        "atom_transformer.blocks",
    )
    for root in roots:
        count = count_blocks(state, root)
        if count is not None:
            print(f"  {root:<48} {count:>6}")

    if args.grep:
        print(f"\nkeys matching {args.grep!r}:")
        for index, (key, shape) in enumerate(iter_shapes(state, args.grep)):
            if index >= args.limit:
                print(f"  ... ({args.limit} shown; raise --limit for more)")
                break
            print(f"  {key:<64} {shape}")
    return 0


def entrypoint() -> None:
    raise SystemExit(main())

if __name__ == "__main__":  # ``python -m foldjax.models.openfold3.cli.<name>``
    entrypoint()
