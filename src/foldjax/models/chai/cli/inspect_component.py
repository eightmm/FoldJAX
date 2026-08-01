"""CLI: inspect a Chai-1 TorchScript component's extractable weights.

Usage:
    chai-jax-inspect-component --component path/to/trunk.pt --limit 80
"""

from __future__ import annotations

import argparse

from foldjax.models.chai.bridge.component_io import (
    load_component_state_dict,
    summarize_state_dict,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect Chai-1 component weights")
    ap.add_argument("--component", required=True, help="path to a Chai .pt component")
    ap.add_argument("--limit", type=int, default=80, help="max rows to print")
    args = ap.parse_args()

    state = load_component_state_dict(args.component)
    rows = summarize_state_dict(state)
    print(f"component: {args.component}")
    print(f"named tensors: {len(rows)}")
    for name, shape, dtype in rows[: args.limit]:
        print(f"  {name:60s} {str(shape):20s} {dtype}")
    if len(rows) > args.limit:
        print(f"  ... ({len(rows) - args.limit} more)")


if __name__ == "__main__":
    main()
