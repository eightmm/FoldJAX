"""Export a trusted upstream Protenix checkpoint to native JAX weights."""

from __future__ import annotations

from collections.abc import Sequence

from foldjax.models._export_cli import run_weight_export
from foldjax.models.protenix.bridge.torch_mapping import load_torch_checkpoint
from foldjax.models.protenix.bridge.weights_io import save_native_weights


def main(argv: Sequence[str] | None = None) -> None:
    run_weight_export(
        argv,
        description=__doc__,
        load=load_torch_checkpoint,
        save=save_native_weights,
    )


if __name__ == "__main__":
    main()
