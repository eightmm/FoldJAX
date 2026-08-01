"""Convert a Chai TorchScript component to the native Chai-JAX format."""

from __future__ import annotations

import argparse

from foldjax.models.chai.bridge.component_io import convert_component_to_native


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Chai TorchScript component for Torch-free JAX loading"
    )
    parser.add_argument("--component", required=True, help="path to a Chai .pt file")
    parser.add_argument("--output", required=True, help="output .npz path")
    args = parser.parse_args()

    tensor_count = convert_component_to_native(args.component, args.output)
    print(f"exported {tensor_count} tensors to {args.output}")


if __name__ == "__main__":
    main()
