"""What the command line writes where: results on stdout, everything else on stderr."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

from foldjax import cli
from foldjax.backends.base import Backend
from foldjax.registry import backend_override
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)


class _ChattyBackend(Backend):
    """Prints from Python and from the file-descriptor level, as native code does."""

    name = "boltz2"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model="boltz2", input_formats=("native",))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        print("Found explicit empty MSA for some proteins")
        os.write(1, b"native runner chatter\n")
        warnings.warn("a backend warning", UserWarning, stacklevel=1)
        return PredictionResult(
            model=self.name,
            samples=(
                PredictionSample(
                    seed=request.seed,
                    coordinates=((0.0, 0.0, 0.0),),
                    scores={"iptm": 0.5},
                ),
            ),
            output_dir=request.output_dir,
        )


def test_predict_stdout_is_only_the_json_result(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    job = tmp_path / "job.yaml"
    job.write_text("version: 1\nsequences: []\n")
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"w")

    with backend_override("boltz2", _ChattyBackend):
        code = cli.main(
            [
                "predict",
                "--model",
                "boltz2",
                "--input",
                str(job),
                "--weights",
                str(weights),
                "--output-dir",
                str(tmp_path / "out"),
                "--seed",
                "3",
                "--no-cache",
                "--quiet",
            ]
        )

    captured = capfd.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["model"] == "boltz2"
    assert "Found explicit empty MSA" in captured.err
    assert "native runner chatter" in captured.err
    # The descriptor is pointed back at stdout afterwards.
    os.write(1, b"after\n")
    assert capfd.readouterr().out == "after\n"


def test_cli_warnings_are_one_line_and_printed_once(capsys) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        cli._format_warnings()
        for _ in range(3):
            warnings.warn("boltz2 at 16 tokens: no peak estimate", RuntimeWarning)
        warnings.warn("something else", UserWarning)

    err = capsys.readouterr().err.splitlines()
    assert err == [
        "foldjax: warning: boltz2 at 16 tokens: no peak estimate",
        "foldjax: warning: something else",
    ]


def test_the_entrypoint_installs_the_format_and_main_does_not(
    monkeypatch,
) -> None:
    with warnings.catch_warnings():
        before = warnings.showwarning
        monkeypatch.setattr(cli, "main", lambda: 0)
        with pytest.raises(SystemExit):
            cli.entrypoint()
        assert warnings.showwarning is not before
