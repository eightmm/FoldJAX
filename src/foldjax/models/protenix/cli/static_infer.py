"""Deprecated alias for standalone Protenix JAX prediction."""

from __future__ import annotations

from collections.abc import Sequence

from foldjax.models.protenix.cli.predict import main as _predict_main


def main(argv: Sequence[str] | None = None) -> None:
    _predict_main(argv)


if __name__ == "__main__":
    main()

