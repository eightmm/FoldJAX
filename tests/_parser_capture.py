"""Get at a CLI's constructed `argparse` parser without running the CLI.

None of the port entry points expose a parser factory: `main` builds the parser,
parses immediately, and then runs a model. Tests that need to assert on the
parser itself -- which flags exist, their types, defaults and help -- intercept
`parse_args`, keep the bound parser and stop there, the way
`tests/models/opendde/test_cache_profile.py` reads released defaults.

Not named `test_*`, so pytest does not collect it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import patch

import pytest


class ParserCapturedError(Exception):
    """Raised once the parser exists, to stop before any work is done."""


def capture_parser(
    main: Callable[..., Any], argv: Sequence[str] = ()
) -> argparse.ArgumentParser:
    """Return the parser `main` builds, without letting `main` proceed."""
    captured: dict[str, argparse.ArgumentParser] = {}

    def capture(
        parser: argparse.ArgumentParser,
        _args: object = None,
        _namespace: object = None,
    ) -> None:
        captured["parser"] = parser
        raise ParserCapturedError

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(ParserCapturedError):
            main(list(argv))
    return captured["parser"]


def declared_flags(parser: argparse.ArgumentParser) -> set[str]:
    """Every option string the parser accepts."""
    return {flag for action in parser._actions for flag in action.option_strings}


def flag_order(parser: argparse.ArgumentParser) -> list[str]:
    """Option strings in declaration order, which is the order `--help` prints."""
    return [flag for action in parser._actions for flag in action.option_strings]
