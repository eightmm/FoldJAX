"""Pin the weight-export console scripts the Protenix and OpenDDE ports share.

``protenix-jax-export-weights`` and ``opendde-jax-export-weights`` are both
published in ``pyproject.toml`` and are how a user converts a trusted upstream
checkpoint. Their ``main`` bodies are byte-identical apart from the module
docstring and where the two callables come from, and the docstring is what
``--help`` prints, so it has to stay per-port.

The per-port suites already cover ``--no-compress``. These add the parts a
shared implementation could silently change: the parser description, the flag
declarations, the ``set_defaults(compress=True)`` line, and the explicit
``--compress`` spelling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from foldjax.models.opendde.bridge import export_weights as opendde_export
from foldjax.models.protenix.bridge import export_weights as protenix_export

_MODULES: tuple[ModuleType, ...] = (protenix_export, opendde_export)
_IDS: tuple[str, ...] = ("protenix", "opendde")

_EXPECTED_DESCRIPTIONS: dict[str, str] = {
    "protenix": (
        "Export a trusted upstream Protenix checkpoint to native JAX weights."
    ),
    "opendde": "Export a trusted official OpenDDE checkpoint to native JAX weights.",
}


class _ParserCapturedError(Exception):
    """Raised once the parser exists, to stop before any file is read."""


def _capture_parser(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> argparse.ArgumentParser:
    captured: dict[str, argparse.ArgumentParser] = {}

    def capture(
        parser: argparse.ArgumentParser,
        _args: object = None,
        _namespace: object = None,
    ) -> None:
        captured["parser"] = parser
        raise _ParserCapturedError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_ParserCapturedError):
        module.main([])
    return captured["parser"]


def _run(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> dict[str, Any]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"trusted fixture")
    output = tmp_path / "native.jax"
    params = object()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(module, "load_torch_checkpoint", lambda path: params)

    def save(path: Path, value: object, *, compress: bool) -> None:
        seen.update(path=path, value=value, compress=compress)

    monkeypatch.setattr(module, "save_native_weights", save)
    module.main(["--checkpoint", str(checkpoint), "--out", str(output), *extra])
    seen["expected"] = {"path": output, "value": params}
    return seen


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_export_parser_keeps_the_port_docstring_as_its_description(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, request: Any
) -> None:
    parser = _capture_parser(module, monkeypatch)
    port = request.node.callspec.id
    assert parser.description == module.__doc__
    assert parser.description == _EXPECTED_DESCRIPTIONS[port]


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_export_parser_declares_the_same_flags_in_the_same_order(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = _capture_parser(module, monkeypatch)
    specs = [
        (
            tuple(action.option_strings),
            action.dest,
            action.type,
            action.required,
            action.default,
            type(action).__name__,
        )
        for action in parser._actions
    ]
    assert specs == [
        (("-h", "--help"), "help", None, False, argparse.SUPPRESS, "_HelpAction"),
        (("--checkpoint",), "checkpoint", Path, True, None, "_StoreAction"),
        (("--out",), "out", Path, True, None, "_StoreAction"),
        (("--compress",), "compress", None, False, True, "_StoreTrueAction"),
        (("--no-compress",), "compress", None, False, True, "_StoreFalseAction"),
    ]


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_export_compresses_by_default(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    seen = _run(module, monkeypatch, tmp_path)
    assert seen["path"] == seen["expected"]["path"]
    assert seen["value"] is seen["expected"]["value"]
    assert seen["compress"] is True
    written = seen["expected"]["path"]
    assert f"wrote native weights: {written}" in capsys.readouterr().out


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
@pytest.mark.parametrize(
    ("flag", "compress"), [("--compress", True), ("--no-compress", False)]
)
def test_export_honours_the_explicit_compression_switch(
    module: ModuleType,
    flag: str,
    compress: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert _run(module, monkeypatch, tmp_path, flag)["compress"] is compress


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_export_rejects_the_two_compression_switches_together(
    module: ModuleType, tmp_path: Path, capsys: Any
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        module.main(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--out",
                str(tmp_path / "native.jax"),
                "--compress",
                "--no-compress",
            ]
        )
    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_export_reports_a_missing_checkpoint_before_loading_it(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(path: Path) -> Any:
        raise AssertionError(f"must not read {path}")

    monkeypatch.setattr(module, "load_torch_checkpoint", refuse)
    missing = tmp_path / "missing.pt"
    with pytest.raises(SystemExit) as exit_info:
        module.main(
            ["--checkpoint", str(missing), "--out", str(tmp_path / "native.jax")]
        )
    assert str(exit_info.value) == f"missing checkpoint: {missing}"
