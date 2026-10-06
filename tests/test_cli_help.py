"""Discovery commands say what a run will use before it runs."""

from __future__ import annotations

import json
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
