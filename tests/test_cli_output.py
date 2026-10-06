"""What the command line writes where: results on stdout, everything else on stderr."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

import foldjax
from foldjax import cli, results
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


class _TwoSampleBackend(Backend):
    """Two structures per seed; seed 2's second one ranks first."""

    name = "boltz2"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model="boltz2", input_formats=("native",))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        samples = []
        for index in range(2):
            path = Path(request.output_dir) / f"native-{request.seed}-{index}.cif"
            path.write_text("data_mock\n_entry.id mock\n#\n", encoding="utf-8")
            score = 0.9 if (request.seed, index) == (2, 1) else 0.1 * (index + 1)
            samples.append(
                PredictionSample(
                    seed=request.seed,
                    structure_path=path,
                    scores={"confidence_score": score},
                )
            )
        return PredictionResult(
            model=self.name, samples=tuple(samples), output_dir=request.output_dir
        )


def _two_seed_run(tmp_path: Path) -> Path:
    job = tmp_path / "job.yaml"
    job.write_text("version: 1\nsequences: []\n")
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"w")
    output = tmp_path / "out"
    with backend_override("boltz2", _TwoSampleBackend):
        foldjax.predict(
            PredictionRequest(
                model="boltz2",
                input=job,
                weights=weights,
                output_dir=output,
                seeds=(1, 2),
                use_compile_cache=False,
            )
        )
    return output


def test_merged_seeds_keep_their_own_sample_numbers(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    output = _two_seed_run(tmp_path)
    manifest = json.loads((output / "foldjax_run.json").read_text())
    assert [
        (sample["seed"], sample["metadata"]["sample"]) for sample in manifest["samples"]
    ] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert (manifest["best"]["seed"], manifest["best"]["sample"]) == (2, 1)

    rows = results.results_table(results.load_results(output))
    assert [(row["seed"], row["sample"]) for row in rows] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert [row["best_within_model"] for row in rows] == [False, False, False, True]

    assert cli.main(["show", str(output)]) == 0
    table = capsys.readouterr().out
    assert "best      seed 2 / sample 01" in table
    best_rows = [line for line in table.splitlines() if "<- best" in line]
    assert len(best_rows) == 1 and best_rows[0].split()[:2] == ["2", "1"]
    assert "input     job.yaml" in table


def test_a_merged_manifest_from_before_per_seed_numbers_still_reads(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Old merged manifests counted samples across seeds, best included."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    output = _two_seed_run(tmp_path)
    path = output / "foldjax_run.json"
    manifest = json.loads(path.read_text())
    for sample in manifest["samples"]:
        sample["metadata"].pop("sample")
    manifest["best"]["sample"] = 3
    path.write_text(json.dumps(manifest))

    rows = results.results_table(results.load_results(output))
    assert [row["sample"] for row in rows] == [0, 1, 0, 1]
    assert [row["best_within_model"] for row in rows] == [False, False, False, True]
    assert cli.main(["show", str(output)]) == 0
    assert "best      seed 2 / sample 01" in capsys.readouterr().out


def test_show_lists_failures_and_warns_on_zero_structures(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    output = _two_seed_run(tmp_path)
    (output / "foldjax_failures.json").write_text(
        json.dumps(
            [
                {
                    "model": "protenix",
                    "input": str(tmp_path / "other.yaml"),
                    "seed": 5,
                    "output_dir": str(output / "x"),
                    "error": "planned failure",
                    "error_type": "MemoryError",
                }
            ]
        )
    )
    assert cli.main(["show", str(output)]) == 0
    text = capsys.readouterr().out
    assert "failed    1 run(s)" in text
    assert "protenix · other.yaml seed 5: planned failure" in text

    empty = tmp_path / "empty"
    empty.mkdir()
    manifest = json.loads((output / "foldjax_run.json").read_text())
    manifest["samples"] = []
    manifest["best"] = None
    (empty / "foldjax_run.json").write_text(json.dumps(manifest))
    with pytest.warns(UserWarning, match="found 0 structures"):
        assert cli.main(["show", str(empty)]) == 0
    with pytest.warns(UserWarning, match="found 0 structures"):
        assert cli.main(["show", str(empty), "--format", "csv"]) == 0
    with pytest.warns(UserWarning, match="compare found 0 structures"):
        assert cli.main(["compare", str(empty), "--out", str(tmp_path / "c")]) == 0
