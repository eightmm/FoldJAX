"""Export a trusted official OpenDDE checkpoint to native JAX weights."""

from __future__ import annotations

from collections.abc import Sequence

from foldjax.models._export_cli import run_weight_export
from foldjax.models.opendde.bridge.weights_io import (
    load_torch_checkpoint,
    save_native_weights,
)


def main(argv: Sequence[str] | None = None) -> None:
    run_weight_export(
        argv,
        description=__doc__,
        load=load_torch_checkpoint,
        save=save_native_weights,
    )


if __name__ == "__main__":
    main()
