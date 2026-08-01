"""Export Chai's official ESM2 TorchScript asset to native JAX shards."""

from __future__ import annotations

import argparse

from foldjax.models.chai.bridge.esm2_io import export_native_esm2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the pinned Chai ESM2 model for Torch-free JAX inference"
    )
    parser.add_argument("--source", required=True, help="official traced ESM2 .pt")
    parser.add_argument("--output", required=True, help="new native bundle directory")
    args = parser.parse_args()
    manifest = export_native_esm2(args.source, args.output)
    print(
        f"exported {manifest['model']['layers']} ESM2 layers to {args.output} "
        f"(source sha256={manifest['source_sha256']})"
    )


if __name__ == "__main__":
    main()
