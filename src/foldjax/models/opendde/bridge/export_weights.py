"""Export a trusted official OpenDDE checkpoint to native JAX weights."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from foldjax.models.opendde.bridge.weights_io import (
    load_torch_checkpoint,
    save_native_weights,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    compression = parser.add_mutually_exclusive_group()
    compression.add_argument("--compress", dest="compress", action="store_true")
    compression.add_argument("--no-compress", dest="compress", action="store_false")
    parser.set_defaults(compress=True)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    params = load_torch_checkpoint(args.checkpoint)
    save_native_weights(args.out, params, compress=args.compress)
    print(f"wrote native weights: {args.out}")


if __name__ == "__main__":
    main()
