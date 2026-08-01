"""Installed commands for building all native Chai-JAX assets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from foldjax.models.chai.bridge.bundle_io import (
    COMPONENT_TENSOR_COUNTS,
    export_native_bundle,
)
from foldjax.models.chai.bridge.conformer_export import export_native_conformers
from foldjax.models.chai.bridge.esm2_io import export_native_esm2


def add_assets_parser(subparsers: argparse._SubParsersAction) -> None:
    assets = subparsers.add_parser(
        "assets", help="Export official Chai assets for Torch-free inference."
    )
    commands = assets.add_subparsers(dest="asset_command", required=True)

    bundle = commands.add_parser(
        "bundle", help="Export the strict six-component native model bundle."
    )
    _add_bundle_arguments(bundle)

    conformers = commands.add_parser(
        "conformers", help="Export Chai's native conformer archive."
    )
    _add_conformer_arguments(conformers)

    esm2 = commands.add_parser(
        "esm2", help="Export the pinned native ESM2 model bundle."
    )
    _add_esm2_arguments(esm2)


def _add_bundle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--components-directory",
        required=True,
        type=Path,
        help="directory containing exactly the six official component .pt files",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--chai-source", required=True, help="source repository identity for manifest"
    )
    parser.add_argument(
        "--chai-release", required=True, help="source release or commit for manifest"
    )


def _add_conformer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chai-source-directory",
        type=Path,
        help="optional Chai checkout; decode with Chai's own antipickle "
        "adapters instead of the built-in reader, to cross-check it",
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset-version", default="conformers_v1")


def _add_esm2_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source", required=True, type=Path, help="official pinned ESM2 .pt file"
    )
    parser.add_argument("--output", required=True, type=Path)


def run_assets(args: argparse.Namespace) -> int:
    if args.asset_command == "bundle":
        sources = {
            component: args.components_directory / f"{component}.pt"
            for component in COMPONENT_TENSOR_COUNTS
        }
        missing = [name for name, path in sources.items() if not path.is_file()]
        if missing:
            raise ValueError(f"missing official component files: {missing}")
        manifest = export_native_bundle(
            sources,
            args.output,
            chai_source=args.chai_source,
            chai_release=args.chai_release,
        )
        print(
            f"exported {len(manifest['components'])} components to "
            f"{args.output.resolve()}"
        )
        return 0
    if args.asset_command == "conformers":
        count = export_native_conformers(
            args.source,
            args.output,
            chai_source_directory=args.chai_source_directory,
            asset_version=args.asset_version,
        )
        print(f"exported {count} conformers to {args.output.resolve()}")
        return 0
    if args.asset_command == "esm2":
        manifest = export_native_esm2(args.source, args.output)
        print(
            f"exported {manifest['model']['layers']} ESM2 layers to "
            f"{args.output.resolve()} "
            f"(source sha256={manifest['source_sha256']})"
        )
        return 0
    raise ValueError(f"unknown asset command: {args.asset_command}")


def _standalone_main(
    name: str,
    description: str,
    add_arguments,
    asset_command: str,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog=name, description=description)
    add_arguments(parser)
    args = parser.parse_args(argv)
    args.asset_command = asset_command
    return run_assets(args)


def bundle_main(argv: Sequence[str] | None = None) -> int:
    return _standalone_main(
        "chai-jax-export-bundle",
        "Export the strict six-component native Chai-JAX model bundle.",
        _add_bundle_arguments,
        "bundle",
        argv,
    )


def conformers_main(argv: Sequence[str] | None = None) -> int:
    return _standalone_main(
        "chai-jax-export-conformers",
        "Export Chai's conformers for Torch-free Chai-JAX inference.",
        _add_conformer_arguments,
        "conformers",
        argv,
    )
