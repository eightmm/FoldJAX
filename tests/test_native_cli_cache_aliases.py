"""Native Protenix and OpenDDE predict CLIs accept FoldJAX's cache flag aliases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foldjax.models.opendde import runner as opendde_runner
from foldjax.models.opendde.cli import predict as opendde_predict
from foldjax.models.protenix import runner as protenix_runner
from foldjax.models.protenix.cli import predict as protenix_predict
from tests._parser_capture import capture_parser


def _required_argv(tmp_path: Path) -> list[str]:
    return [
        "--input-json",
        str(tmp_path / "job.json"),
        "--weights",
        str(tmp_path / "weights.jax"),
        "--out",
        str(tmp_path / "out"),
    ]


def _protenix_main_config(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> protenix_runner.PredictionConfig:
    captured: list[protenix_runner.PredictionConfig] = []

    def fake_run(
        config: protenix_runner.PredictionConfig, **_kwargs: Any
    ) -> list[Path]:
        captured.append(config)
        return []

    monkeypatch.setattr(protenix_predict, "run_prediction", fake_run)
    protenix_predict.main(argv)
    assert len(captured) == 1
    return captured[0]


def _opendde_main_config(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> opendde_runner.PredictionConfig:
    captured: list[opendde_runner.PredictionConfig] = []

    def fake_run(
        config: opendde_runner.PredictionConfig, **_kwargs: Any
    ) -> list[Path]:
        captured.append(config)
        return []

    monkeypatch.setattr(opendde_predict, "run_prediction", fake_run)
    opendde_predict.main(argv)
    assert len(captured) == 1
    return captured[0]


def test_protenix_cache_dir_alias_matches_compile_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _required_argv(tmp_path)
    parser = protenix_predict.build_parser()

    omitted_ns = parser.parse_args(base)
    assert omitted_ns.compile_cache == Path("outputs/compile_cache")
    assert omitted_ns.no_compile_cache is False
    assert not hasattr(omitted_ns, "cache_dir")
    assert not hasattr(omitted_ns, "no_cache")

    omitted_cfg = protenix_predict.parse_config(base)
    assert omitted_cfg.compile_cache == Path("outputs/compile_cache")
    assert omitted_cfg.no_compile_cache is False

    target = tmp_path / "custom-cache"
    old_argv = [*base, "--compile-cache", str(target)]
    new_argv = [*base, "--cache-dir", str(target)]
    equals_argv = [*base, f"--cache-dir={target}"]

    old_ns = parser.parse_args(old_argv)
    new_ns = parser.parse_args(new_argv)
    equals_ns = parser.parse_args(equals_argv)
    assert new_ns == old_ns
    assert equals_ns == old_ns
    assert new_ns.compile_cache == target

    old_cfg = _protenix_main_config(monkeypatch, old_argv)
    new_cfg = _protenix_main_config(monkeypatch, new_argv)
    assert new_cfg == old_cfg
    assert new_cfg == protenix_predict.parse_config(new_argv)
    assert new_cfg.compile_cache == target
    assert new_cfg.no_compile_cache is False


def test_protenix_no_cache_alias_matches_no_compile_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _required_argv(tmp_path)
    parser = protenix_predict.build_parser()
    target = tmp_path / "disabled-cache"

    old_ns = parser.parse_args([*base, "--no-compile-cache"])
    new_ns = parser.parse_args([*base, "--no-cache"])
    assert new_ns == old_ns
    assert new_ns.no_compile_cache is True
    assert new_ns.compile_cache == Path("outputs/compile_cache")

    old_combo = [*base, "--compile-cache", str(target), "--no-compile-cache"]
    new_combo = [*base, "--cache-dir", str(target), "--no-cache"]
    assert parser.parse_args(new_combo) == parser.parse_args(old_combo)

    old_cfg = _protenix_main_config(monkeypatch, old_combo)
    new_cfg = _protenix_main_config(monkeypatch, new_combo)
    assert new_cfg == old_cfg
    assert new_cfg.compile_cache == target
    assert new_cfg.no_compile_cache is True


def test_opendde_cache_dir_alias_matches_compile_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _required_argv(tmp_path)
    parser = capture_parser(opendde_predict.main)

    omitted_ns = parser.parse_args(base)
    assert omitted_ns.compile_cache is None
    assert not hasattr(omitted_ns, "cache_dir")

    omitted_cfg = _opendde_main_config(monkeypatch, base)
    assert omitted_cfg.compile_cache is None

    target = tmp_path / "custom-cache"
    old_argv = [*base, "--compile-cache", str(target)]
    new_argv = [*base, "--cache-dir", str(target)]
    equals_argv = [*base, f"--cache-dir={target}"]

    old_ns = parser.parse_args(old_argv)
    new_ns = parser.parse_args(new_argv)
    equals_ns = parser.parse_args(equals_argv)
    assert new_ns == old_ns
    assert equals_ns == old_ns
    assert new_ns.compile_cache == target

    old_cfg = _opendde_main_config(monkeypatch, old_argv)
    new_cfg = _opendde_main_config(monkeypatch, new_argv)
    assert new_cfg == old_cfg
    assert new_cfg.compile_cache == target


def test_repeated_cache_flags_keep_last_wins_behaviour_on_both_clis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _required_argv(tmp_path)
    first = tmp_path / "first-cache"
    second = tmp_path / "second-cache"

    sequences = (
        ("--compile-cache", str(first), "--compile-cache", str(second)),
        ("--cache-dir", str(first), "--cache-dir", str(second)),
        ("--compile-cache", str(first), "--cache-dir", str(second)),
        ("--cache-dir", str(first), "--compile-cache", str(second)),
    )
    expected_protenix = protenix_predict.parse_config(
        [*base, "--compile-cache", str(second)]
    )
    expected_opendde = _opendde_main_config(
        monkeypatch, [*base, "--compile-cache", str(second)]
    )
    for flags in sequences:
        assert protenix_predict.parse_config([*base, *flags]) == expected_protenix
        assert _opendde_main_config(monkeypatch, [*base, *flags]) == expected_opendde

    for switch_pair in (
        ("--no-compile-cache", "--no-cache"),
        ("--no-cache", "--no-compile-cache"),
        ("--no-cache", "--no-cache"),
    ):
        cfg = protenix_predict.parse_config([*base, *switch_pair])
        assert cfg.no_compile_cache is True


def test_cache_alias_edges_and_malformed_inputs(tmp_path: Path) -> None:
    base = _required_argv(tmp_path)
    protenix_parser = protenix_predict.build_parser()
    opendde_parser = capture_parser(opendde_predict.main)

    assert protenix_parser.parse_args(
        [*base, "--cache-dir", ""]
    ) == protenix_parser.parse_args([*base, "--compile-cache", ""])
    assert opendde_parser.parse_args(
        [*base, "--cache-dir", ""]
    ) == opendde_parser.parse_args([*base, "--compile-cache", ""])

    for parser in (protenix_parser, opendde_parser):
        with pytest.raises(SystemExit) as missing_arg:
            parser.parse_args([*base, "--cache-dir"])
        assert missing_arg.value.code == 2

    with pytest.raises(SystemExit) as unexpected_val:
        protenix_parser.parse_args([*base, "--no-cache=1"])
    assert unexpected_val.value.code == 2

    with pytest.raises(SystemExit) as unknown_switch:
        opendde_parser.parse_args([*base, "--no-cache"])
    assert unknown_switch.value.code == 2
