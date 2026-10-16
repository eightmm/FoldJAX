"""Discovery commands say what a run will use before it runs."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

from foldjax.cli import _parser, main
from foldjax.registry import normalize_model_name


def test_home_path_offers_every_location(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    for name in ("msa", "templates"):
        assert main(["home", "--path", name]) == 0
        assert capsys.readouterr().out.strip() == str(tmp_path / name)


def test_boltz_2_is_an_alias_like_af3() -> None:
    assert normalize_model_name("boltz-2") == "boltz2"
    assert normalize_model_name("af3") == "alphafold3"


@pytest.mark.parametrize(
    "argv",
    [
        ["capabilities", "--help"],
        ["runtime", "status", "--help"],
        ["weights", "path", "--help"],
    ],
)
def test_model_flags_name_the_models(argv: list[str], capsys) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(argv)
    text = " ".join(capsys.readouterr().out.split())
    assert "boltz2 (boltz, boltz-2, boltz-jax)" in text


def test_capabilities_show_sampling_defaults(capsys) -> None:
    assert main(["capabilities", "--model", "boltz2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sampling_defaults"]["num_steps"] == 200
    assert set(payload["sampling_defaults"]) == set(payload["sampling"])


def test_plan_prints_the_effective_sampling_and_where_it_comes_from(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"w")
    assert (
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--sequence",
                "MKTAYIAKQRQISFVK",
                "--msa",
                "single",
                "--num-samples",
                "2",
                "--weights",
                str(weights),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["sampling"]["num_samples"] == 2
    assert plan["sampling_source"]["num_samples"] == "request"
    assert plan["sampling"]["num_steps"] == 200
    assert plan["sampling_source"]["num_steps"] == "default"
    # The featurizer's own cap, `const.max_msa_seqs`, which an omitted depth runs.
    assert plan["sampling"]["max_msa_depth"] == 16384
    assert plan["sampling_source"]["max_msa_depth"] == "default"
    assert "not_checked" not in plan

    assert (
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--sequence",
                "MKTAYIAKQRQISFVK",
                "--msa",
                "single",
                "--pad-msa",
                "1",
                "--weights",
                str(weights),
            ]
        )
        == 0
    )
    assert "padding.msa" in json.loads(capsys.readouterr().out)["not_checked"][0]


def test_plan_with_padding_shows_the_token_bucket(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The padding block is the request (null where unpinned); the bucket the
    run would pad to is what a reader of `plan --padding` wants to know."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"w")
    argv = [
        "plan", "--model", "boltz2", "--sequence", "MKTAYIAKQRQISFVK" * 10,
        "--msa", "single", "--weights", str(weights), "--padding",
    ]
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["padding"]["tokens"] is None
    assert plan["padding_estimate"] == {"tokens": 160, "token_bucket": 256}

    assert main([*argv[:-1], "--pad-tokens", "512"]) == 0
    assert json.loads(capsys.readouterr().out)["padding_estimate"] == {
        "tokens": 160,
        "token_bucket": 512,
    }
    assert main(argv[:-1]) == 0
    assert "padding_estimate" not in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize(
    "n_tokens,padding_args,token_bucket",
    [
        (767, (), None),
        (768, (), None),
        (896, (), None),
        (700, ("--padding",), 768),
        (700, ("--pad-tokens", "750"), 750),
        (700, ("--pad-tokens", "800"), 800),
        (9000, ("--padding", "--padding-overflow", "exact"), 9000),
        (9000, ("--padding",), None),
    ],
)
def test_esmfold2_plan_reports_capacity_risk_in_every_output_mode(
    tmp_path: Path, monkeypatch, capsys, n_tokens: int, json_output: bool,
    padding_args: tuple[str, ...], token_bucket: int | None,
) -> None:
    from foldjax import memory_policy

    def no_device_read(**kwargs):
        raise AssertionError("planning tried to initialize the GPU")

    monkeypatch.setattr(memory_policy, "device_memory_budget", no_device_read)
    store = tmp_path / "store"
    monkeypatch.setenv("FOLDJAX_HOME", str(store))
    weights = tmp_path / "weights"
    weights.mkdir()
    argv = [
        "plan", "--model", "esmfold2", "--sequence", "M" * n_tokens,
        "--weights", str(weights), "--option", "no_language_model=true",
        *padding_args,
    ]
    if json_output:
        argv.append("--json")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert main(argv) == 0

    plan = json.loads(capsys.readouterr().out)
    expected = memory_policy.capacity_warning(
        "esmfold2", max(n_tokens, token_bucket or 0)
    )
    if padding_args:
        assert plan["padding_estimate"]["token_bucket"] == token_bucket
    if json_output:
        assert plan["slurm"].get("warning") == expected
        assert plan["slurm"]["tokens_estimate"] == n_tokens
        if expected is not None:
            assert plan["slurm"]["reason"].count(expected) == 1
        assert not caught
    else:
        assert "slurm" not in plan
        assert [str(warning.message) for warning in caught] == (
            [] if expected is None else [expected]
        )
    assert not store.exists()


def test_esmfold2_batch_plan_warns_once_for_the_same_capacity_risk(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    weights = tmp_path / "weights"
    weights.mkdir()
    inputs = []
    for n_tokens in (768, 896):
        source = tmp_path / f"job-{n_tokens}.json"
        source.write_text(json.dumps({
            "entities": [{"type": "protein", "id": "A", "sequence": "M" * n_tokens}]
        }))
        inputs.append(str(source))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert main([
            "plan", "--model", "esmfold2", "--input", *inputs,
            "--weights", str(weights), "--option", "no_language_model=true",
        ]) == 0

    assert len(json.loads(capsys.readouterr().out)) == 2
    assert len(caught) == 1
    assert "24 GB cards with 22-23.5 GiB" in str(caught[0].message)


def _console_scripts() -> dict[str, str]:
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


@pytest.mark.parametrize("script", sorted(_console_scripts()))
def test_every_console_script_prints_its_help(script: str, monkeypatch, capsys) -> None:
    """argparse formats help lazily, so a stray ``%`` only fails on ``--help``."""
    import importlib

    module_name, _, attribute = _console_scripts()[script].partition(":")
    entry = getattr(importlib.import_module(module_name), attribute)
    monkeypatch.setattr("sys.argv", [script, "--help"])
    with pytest.raises(SystemExit) as stopped:
        entry()
    assert stopped.value.code in (0, None)
    assert "usage:" in capsys.readouterr().out


def test_python_dash_m_foldjax_runs_the_cli() -> None:
    import subprocess
    import sys

    import foldjax

    environment = dict(os.environ)
    source = str(Path(foldjax.__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-m", "foldjax", "--help"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("usage: foldjax")
