"""One body for the ports' weight-export console scripts.

`protenix-jax-export-weights` and `opendde-jax-export-weights` are published
separately and each prints its own module docstring as `--help`, so every port
keeps its own `main` and its own docstring; only the body moves here.

The loader and the writer arrive as arguments rather than being imported here,
because the port suites monkeypatch those two names on the port module. Passing
them from inside `main` keeps the lookup at call time, which is what makes the
patch land.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol


class _NativeWriter(Protocol):
    def __call__(self, out: Path, params: Any, *, compress: bool) -> Any: ...


def run_weight_export(
    argv: Sequence[str] | None,
    *,
    description: str | None,
    load: Callable[[Path], Any],
    save: _NativeWriter,
) -> None:
    """Convert one trusted upstream checkpoint into native JAX weights."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    compression = parser.add_mutually_exclusive_group()
    compression.add_argument("--compress", dest="compress", action="store_true")
    compression.add_argument("--no-compress", dest="compress", action="store_false")
    parser.set_defaults(compress=True)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    params = load(args.checkpoint)
    save(args.out, params, compress=args.compress)
    print(f"wrote native weights: {args.out}")
