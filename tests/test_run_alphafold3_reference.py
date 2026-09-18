"""CPU contracts for the AlphaFold 3 common-JAX reference boundary."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from bench import run_alphafold3_reference as runner


def test_native_input_rejects_common_entities_dialect(tmp_path):
    common = tmp_path / "common.json"
    common.write_text('{"entities": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="sequences dialect"):
        runner._native_input(common)


def test_native_input_requires_supplied_protein_msa_and_templates(tmp_path):
    native = tmp_path / "native.json"
    unpaired = tmp_path / "unpaired.a3m"
    unpaired.write_text(">query\nACDE\n", encoding="utf-8")
    native.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "ACDE",
                            "unpairedMsaPath": unpaired.name,
                            "pairedMsa": "",
                            "templates": [],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert runner._native_input(native, require_provided=True) == native

    native.write_text(
        '{"sequences": [{"protein": {"id": "A", "sequence": "ACDE"}}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unpairedMsa"):
        runner._native_input(native, require_provided=True)

    native.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "ACDE",
                            "unpairedMsa": "",
                            "unpairedMsaPath": unpaired.name,
                            "pairedMsa": "",
                            "templates": [],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both unpairedMsa and unpairedMsaPath"):
        runner._native_input(native, require_provided=True)


def test_native_argv_uses_publisher_sampling_flags_and_explicit_bootstrap(tmp_path):
    args = SimpleNamespace(
        python=tmp_path / "python",
        source=tmp_path / "source",
        native_input=tmp_path / "native.json",
        output_dir=tmp_path / "output",
        weights=tmp_path / "weights",
        cache_dir=tmp_path / "cache",
        common_runtime=tmp_path / "runtime",
        num_samples=2,
        num_recycles=3,
    )

    argv = runner.native_argv(args)

    assert argv[:3] == [str(args.python), "-c", runner._BOOTSTRAP]
    assert "--run_data_pipeline=true" in argv
    assert "--run_inference=true" in argv
    assert "--num_diffusion_samples=2" in argv
    assert "--num_recycles=3" in argv
    assert f"--buckets={runner._BUCKETS}" in argv


def test_native_argv_skips_database_search_only_when_requested(tmp_path):
    args = SimpleNamespace(
        python=tmp_path / "python",
        source=tmp_path / "source",
        native_input=tmp_path / "native.json",
        output_dir=tmp_path / "output",
        weights=tmp_path / "weights",
        cache_dir=tmp_path / "cache",
        common_runtime=None,
        num_samples=2,
        num_recycles=3,
        skip_database_search=True,
    )

    assert "--run_data_pipeline=false" in runner.native_argv(args)


def test_samples_preserve_one_canonical_structure_per_ranking_row(tmp_path):
    output = tmp_path / "output"
    sample = output / "job" / "seed-101_sample-0"
    sample.mkdir(parents=True)
    (sample / "model.cif").write_text("data", encoding="utf-8")
    (output / "job" / "job_ranking_scores.csv").write_text(
        "seed,sample,ranking_score\n101,0,0.75\n", encoding="utf-8"
    )

    assert runner._samples(output) == [
        {
            "seed": 101,
            "sample": 0,
            "structure_path": "job/seed-101_sample-0/model.cif",
            "scores": {"ranking_score": 0.75},
        }
    ]


def _configure_main(monkeypatch, tmp_path, *, peak=True, returncode=0):
    source = tmp_path / "source"
    (source / "src/alphafold3").mkdir(parents=True)
    (source / "src/alphafold3/__init__.py").write_text("", encoding="utf-8")
    (source / "run_alphafold.py").write_text("", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    native = tmp_path / "native.json"
    native.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "ACDE",
                            "unpairedMsa": ">query\nACDE\n",
                            "pairedMsa": ">query\nACDE\n",
                            "templates": [],
                        }
                    }
                ],
                "modelSeeds": [101],
            }
        ),
        encoding="utf-8",
    )
    common = tmp_path / "common.json"
    common.write_text('{"entities": []}', encoding="utf-8")
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    for name, value in {
        "_parameter_paths": lambda _weights: {"parameter.0": weights / "af3.bin"},
        "_assets": lambda *_args, **_kwargs: {},
        "artifact_identity": lambda **_kwargs: {"artifact": "fixed"},
        "source_identity": lambda *_args: {"source": "fixed"},
        "runtime_identity": lambda: {"runtime": "parent-fixed"},
        "upstream_runtime_versions": lambda *_args, **_kwargs: {
            "runtime": "child-fixed"
        },
        "device_identity": lambda _environment: {"device": "fixed"},
        "execution_identity": lambda *_args, **_kwargs: {"execution": "fixed"},
        "require_unchanged": lambda *_args: None,
    }.items():
        monkeypatch.setattr(runner, name, value)

    def fake_child(_argv, environment, _timeout):
        sample = output / "job" / "seed-101_sample-0"
        sample.mkdir(parents=True)
        (sample / "model.cif").write_text("data", encoding="utf-8")
        (output / "job" / "job_ranking_scores.csv").write_text(
            "seed,sample,ranking_score\n101,0,0.75\n", encoding="utf-8"
        )
        if peak:
            runner.Path(environment["BENCH_PEAK_FILE"]).write_text(
                "1048576", encoding="utf-8"
            )
        return returncode, "child stdout", "child stderr", 1.5, None

    monkeypatch.setattr(runner, "run_cli_child", fake_child)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--python",
            str(tmp_path / "python"),
            "--source",
            str(source),
            "--weights",
            str(weights),
            "--native-input",
            str(native),
            "--common-job",
            str(common),
            "--case",
            "3OG2",
            "--length",
            "42",
            "--output-dir",
            str(output),
            "--cache-dir",
            str(cache),
            "--num-samples",
            "1",
            "--num-recycles",
            "3",
        ],
    )
    return output


def test_main_records_common_runtime_reference_and_complete_child_logs(
    monkeypatch, tmp_path, capsys
):
    output = _configure_main(monkeypatch, tmp_path)

    assert runner.main() == 0

    record = json.loads(capsys.readouterr().out)
    assert record["comparison_scope"] == (
        "common_jax_runtime_reference_not_independent_port_speedup"
    )
    assert record["identity"]["options"]["data_pipeline"] == (
        "publisher_database_search_pipeline"
    )
    assert record["schedule"] == {
        "num_recycles": 3,
        "num_samples": 1,
        "num_steps": 200,
    }
    assert record["samples"][0]["structure_path"] == "job/seed-101_sample-0/model.cif"
    assert (output / "cli_stdout.txt").read_text(encoding="utf-8") == "child stdout"
    assert (output / "cli_stderr.txt").read_text(encoding="utf-8") == "child stderr"


def test_main_keeps_missing_peak_as_a_failed_record(monkeypatch, tmp_path, capsys):
    _configure_main(monkeypatch, tmp_path, peak=False)

    assert runner.main() == 1

    record = json.loads(capsys.readouterr().out)
    assert record["peak_mib"] is None
    assert record["reason"] == "AlphaFold 3 produced no JAX peak observer result"
