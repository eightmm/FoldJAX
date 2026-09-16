"""Stand in for the native prediction module the argv ports import.

The Protenix and OpenDDE adapters resolve two modules by name at prediction
time -- `models/<port>/cli/predict.py`, for the private weight-session ABI, and
`models/<port>/runner.py`, whose `run_prediction` executes the run -- and tests
replace `import_module` wholesale, so one object answers both names.

The double is written once, against the configuration: `run(config, **keywords)`
is what the adapters call. `main` is kept beside it and answers the same body,
by parsing the rendered command with the port's own parser the way `main`
itself does. So a double does not have to be written twice to cover both the
ordinary path and Protenix's `cli_args` escape hatch, which still goes through
that parser, and a test that asserts on the rendered command can keep doing so
-- `result.raw["argv"]` is the same tuple the adapter renders.

Not named `test_*`, so pytest does not collect it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests._parser_capture import capture_parser


def native_module(
    port: str,
    run: Callable[..., list[Path]],
    *,
    loader: Callable[[Path, str], Any] | None = None,
) -> SimpleNamespace:
    """One module double for `port`, driving `run` from either entry point.

    ``loader`` is the private prepared-parameter loader. Supplied only when a
    test exercises weight-session reuse, because the sentinel beside it is what
    the adapter reads to decide whether to negotiate at all: a double without
    one stands for a module that never offered the capability.
    """

    runner = import_module(f"foldjax.models.{port}.runner")
    cli = import_module(f"foldjax.models.{port}.cli.predict")

    def main(argv: Sequence[str], **keywords: Any) -> list[Path]:
        args = capture_parser(cli.main).parse_args(list(argv))
        args.max_msa_depth = runner._resolve_msa_depth(args.max_msa_depth)
        return run(runner.PredictionConfig(**vars(args)), **keywords)

    attributes: dict[str, Any] = {
        "PredictionConfig": runner.PredictionConfig,
        "run_prediction": run,
        "main": main,
    }
    if loader is not None:
        attributes["PREPARED_PARAMS_LOADER_API"] = True
        attributes["_load_prepared_params"] = loader
    return SimpleNamespace(**attributes)
