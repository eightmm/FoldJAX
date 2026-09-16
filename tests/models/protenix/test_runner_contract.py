"""The seam between the Protenix parser and the runner it hands a config to.

`main` builds a `PredictionConfig` from `vars(args)`, so the two sides have to
agree on the field set exactly -- a flag the parser produces and the config has
no field for is a `TypeError` there, and the assertions here say which of the
two moved. They exist because the runner is meant to be callable without argv:
a caller that constructs the config itself gets no parser to fill the gaps.
"""

from __future__ import annotations

from pathlib import Path

from foldjax.models.protenix import runner
from foldjax.models.protenix.cli import predict as predict_cli
from tests._parser_capture import capture_parser


def _parser_dests() -> list[str]:
    """The parser's destinations, deduplicated, in declaration order."""
    parser = capture_parser(predict_cli.main)
    seen: list[str] = []
    for action in parser._actions:
        if action.dest == "help" or action.dest in seen:
            continue
        seen.append(action.dest)
    return seen


def test_the_config_fields_are_the_parsers_own_destinations() -> None:
    assert list(runner.PredictionConfig._fields) == _parser_dests()


def test_a_parsed_namespace_builds_the_config_with_no_translation(
    tmp_path: Path,
) -> None:
    """`main`'s own construction, asserted without running a prediction."""
    parser = capture_parser(predict_cli.main)
    args = parser.parse_args(
        [
            "--input-json",
            str(tmp_path / "job.json"),
            "--weights",
            str(tmp_path / "weights.jax"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    args.max_msa_depth = runner._resolve_msa_depth(args.max_msa_depth)
    config = runner.PredictionConfig(**vars(args))

    assert config.input_json == tmp_path / "job.json"
    assert config.max_msa_depth == runner._DEFAULT_MSA_DEPTH
    assert config.trunk_dtype == "bf16"


def test_the_cli_still_carries_the_prepared_parameter_contract() -> None:
    """The backend reads the sentinel and the loader off the module it imports.

    It imports `cli.predict`; the contract is defined on the runner. Identity
    rather than truthiness, so a second implementation on the CLI fails here.
    """
    assert predict_cli.PREPARED_PARAMS_LOADER_API is True
    assert predict_cli._load_prepared_params is runner._load_prepared_params
