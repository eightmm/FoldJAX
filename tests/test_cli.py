import json
from pathlib import Path

import pytest

from foldjax.cli import _options, entrypoint, main


def test_models_cli_lists_backends(capsys) -> None:
    assert main(["models"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "alphafold3",
        "boltz2",
        "esmfold2",
        "opendde",
        "openfold3",
        "protenix",
    ]


def test_setup_fetches_what_is_public_and_instructs_for_the_rest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """One command, and an honest account of what it could not do.

    The two models whose publishers release parameters on request cannot be
    fetched by anyone, so `setup` must not fail on them -- it reports the step
    the user has to take. Its exit status is about the public downloads only.
    """
    from foldjax import assets

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    monkeypatch.delenv("PROTENIX_TEMPLATE_MMCIF_DIR", raising=False)
    fetched = []
    monkeypatch.setattr(
        assets, "fetch", lambda name, **kw: fetched.append(name) or tmp_path / name
    )

    assert main(["setup"]) == 0
    out = capsys.readouterr().out
    assert fetched == ["boltz2", "opendde", "protenix"], "only the public ones"
    assert "alphafold3  manual" in out and "Request the parameters" in out
    assert "openfold3   manual" in out and "huggingface.co/OpenFold" in out
    # The two modalities that need no weights still need an answer.
    assert "msa" in out and "ColabFold" in out
    assert "PROTENIX_TEMPLATE_MMCIF_DIR" in out


def test_setup_exit_status_follows_the_public_downloads(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))

    def explode(name, **kwargs):
        raise RuntimeError(f"{name} download failed")

    monkeypatch.setattr(assets, "fetch", explode)
    assert main(["setup"]) == 1
    assert "download failed" in capsys.readouterr().err


def test_capabilities_cli_reports_one_backend(capsys) -> None:
    """Asked by an alias, so the CLI's name resolution is covered too."""
    assert main(["capabilities", "--model", "protenix-jax"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "protenix"
    assert "foldjax" in payload["input_formats"]


def test_predict_cli_prints_result_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    seen = {}

    def fake_predict(request):
        from foldjax.schema import PredictionResult

        seen["request"] = request
        return PredictionResult(model="protenix", output_dir=request.output_dir)

    monkeypatch.setattr("foldjax.cli.predict", fake_predict)
    code = main(
        [
            "predict",
            "--model",
            "protenix",
            "--input",
            str(input_path),
            "--weights",
            str(weights),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "3",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--option",
            "n_step=20",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["model"] == "protenix"
    assert seen["request"].seed == 3
    assert seen["request"].cache_dir == tmp_path / "cache"
    assert seen["request"].options == {"n_step": 20}


def test_cli_options_preserve_json_types() -> None:
    assert _options(
        ["steps=20", "enabled=true", "buckets=[256,512]", "dtype=bf16"]
    ) == {
        "steps": 20,
        "enabled": True,
        "buckets": [256, 512],
        "dtype": "bf16",
    }


def test_cli_options_require_key_value_pairs() -> None:
    with pytest.raises(ValueError, match="must be KEY=VALUE"):
        _options(["steps"])


def test_entrypoint_exits_with_the_command_status() -> None:
    with pytest.raises(SystemExit) as exit_info:
        entrypoint()
    assert exit_info.value.code == 2  # argparse: a subcommand is required


def test_user_errors_are_one_clean_line_not_a_traceback(capsys, monkeypatch) -> None:
    """A bad request is the user's problem; a traceback would imply ours."""
    from foldjax import cli

    monkeypatch.setattr(
        cli,
        "main",
        lambda argv=None: (_ for _ in ()).throw(ValueError("no such model 'x'")),
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.entrypoint()

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "foldjax: no such model 'x'"
    assert "Traceback" not in captured.err


def test_unexpected_failures_keep_their_traceback(monkeypatch) -> None:
    from foldjax import cli

    def fail(argv=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "main", fail)
    with pytest.raises(RuntimeError, match="boom"):
        cli.entrypoint()


def test_a_successful_run_still_exits_zero(monkeypatch) -> None:
    from foldjax import cli

    monkeypatch.setattr(cli, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as exit_info:
        cli.entrypoint()
    assert exit_info.value.code == 0
