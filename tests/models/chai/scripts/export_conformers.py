#!/usr/bin/env python3
"""Convert Chai's antipickle conformer asset to a Torch-free native archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chai-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--asset-version", default="conformers_v1")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))

    from foldjax.models.chai.bridge.conformer_export import export_native_conformers

    count = export_native_conformers(
        args.chai_source,
        args.source,
        args.destination,
        asset_version=args.asset_version,
    )
    print(
        f"exported {count} conformers to "
        f"{args.destination.resolve()}"
    )


if __name__ == "__main__":
    main()
