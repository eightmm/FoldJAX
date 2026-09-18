"""CPU contracts for the public-CLI benchmark process boundary."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from bench import run_foldjax_cli as runner


def _summary(samples: int = 1) -> dict:
    return {
        "model": "opendde",
        "samples": [
            {
                "seed": 101,
                "structure_path": f"sample-{index}.cif",
                "scores": {"iptm": 0.6 + index / 100},
            }
            for index in range(samples)
        ],
    }


def test_terminal_prediction_summary_allows_diagnostic_prefix_for_five_samples():
    parsed = runner.parse_stdout_summary(
        "wrote: predictions\n" + json.dumps(_summary(5)) + "\n"
    )

    assert parsed["stdout_summary_framing"] == "terminal_line_json"
    assert parsed["stdout_summary_prefix_present"] is True
    assert parsed["samples"] == _summary(5)["samples"]


def test_terminal_prediction_summary_allows_one_pretty_printed_list():
    parsed = runner.parse_stdout_summary(
        "wrote: predictions\n" + json.dumps([_summary(), _summary()], indent=2)
    )

    assert parsed["stdout_summary_framing"] == "terminal_line_json"
    assert len(parsed["samples"]) == 2


@pytest.mark.parametrize(
    "stdout",
    (
        'wrote: predictions\n{"model":"opendde","samples":[',
        ('wrote: predictions\n{"model":"opendde","samples":[{"scores":{"iptm":NaN}}]}'),
        (
            "wrote: predictions\n"
            '{"model":"opendde","samples":[{"scores":{"iptm":1e309}}]}'
        ),
        '{"model":"opendde","samples":[]}',
        json.dumps(_summary()) + "\ntrailing diagnostics",
        json.dumps(_summary()) + "\n" + json.dumps(_summary()),
    ),
)
def test_terminal_prediction_summary_rejects_invalid_or_ambiguous_stdout(stdout):
    parsed = runner.parse_stdout_summary(stdout)

    assert parsed == {
        "samples": [],
        "stdout_summary_framing": "rejected",
        "stdout_summary_prefix_present": False,
    }


def test_child_boundary_passes_only_constructed_environment(monkeypatch):
    calls = {}

    def fake_run(argv, **kwargs):
        calls.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0, stdout='{"scores": {}}', stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    code, stdout, stderr, elapsed, error = runner.run_cli_child(["cmd"], {"X": "1"}, 9)

    assert (code, stdout, stderr, error) == (0, '{"scores": {}}', "", None)
    assert elapsed >= 0
    assert calls["argv"] == ["cmd"]
    assert calls["env"] == {"X": "1"}
    assert calls["timeout"] == 9


def test_missing_peak_is_none_and_failed_child_is_retained(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=4, stdout="", stderr="bad"
        ),
    )

    code, _stdout, stderr, _elapsed, error = runner.run_cli_child(["cmd"], {}, 1)

    assert runner._read_peak(tmp_path / "missing") is None
    assert (code, stderr, error) == (4, "bad", None)


def test_timeout_preserves_complete_stdout_and_stderr(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise runner.subprocess.TimeoutExpired(
            "foldjax", 1, output=b"partial stdout", stderr=b"partial stderr"
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    code, stdout, stderr, _elapsed, error = runner.run_cli_child(["cmd"], {}, 1)

    assert (code, stdout, stderr) == (124, "partial stdout", "partial stderr")
    assert error == "FoldJAX CLI exceeded 1s timeout"


def _configure_main(monkeypatch, tmp_path, *, returncode=0, write_peak=True):
    case = SimpleNamespace(name="case", length=3, job=tmp_path / "input.json")
    case.job.write_text("{}", encoding="utf-8")
    weights = tmp_path / "weights.safetensors"
    weights.write_text("weights", encoding="utf-8")
    output_dir = tmp_path / "result"
    schedule = {"num_samples": 2, "num_steps": 3, "num_recycles": 4}
    monkeypatch.setitem(
        sys.modules,
        "bench.spec",
        SimpleNamespace(SCHEDULE=schedule, SEED=101, cases=lambda: [case]),
    )
    monkeypatch.setitem(
        sys.modules, "foldjax.cli", SimpleNamespace(_options=lambda _values: {})
    )
    for name, value in {
        "foldjax_checkpoint_paths": lambda *_args, **_kwargs: {"weights": weights},
        "foldjax_implicit_asset_paths": lambda *_args, **_kwargs: {},
        "artifact_identity": lambda **_kwargs: {"artifact": "fixed"},
        "source_identity": lambda *_args, **_kwargs: {"source": "fixed"},
        "runtime_identity": lambda: {"runtime": "fixed"},
        "device_identity": lambda _environment: {"device": "fixed"},
        "execution_identity": lambda *_args, **_kwargs: {"execution": "fixed"},
        "benchmark_identity": lambda **_kwargs: {"identity": "fixed"},
        "require_unchanged": lambda *_args, **_kwargs: None,
    }.items():
        monkeypatch.setattr(runner, name, value)

    def fake_child(_argv, environment, _timeout):
        output_dir.mkdir(exist_ok=True)
        (output_dir / "sample-0.cif").write_text("one", encoding="utf-8")
        (output_dir / "sample-1.cif").write_text("two", encoding="utf-8")
        (output_dir / "foldjax_run.json").write_text("{}", encoding="utf-8")
        if write_peak:
            runner.Path(environment["BENCH_PEAK_FILE"]).write_text(
                "1048576", encoding="utf-8"
            )
        summary = {
            "model": "boltz2",
            "samples": [
                {
                    "seed": 101,
                    "structure_path": "sample-0.cif",
                    "scores": {"iptm": 0.6, "ptm": 0.5},
                },
                {
                    "seed": 102,
                    "structure_path": "sample-1.cif",
                    "scores": {"iptm": 0.7, "ptm": 0.8},
                },
            ],
        }
        return returncode, json.dumps(summary), "child stderr", 1.25, None

    monkeypatch.setattr(runner, "run_cli_child", fake_child)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_foldjax_cli.py",
            "--model",
            "boltz2",
            "--case",
            case.name,
            "--weights",
            str(weights),
            "--output-dir",
            str(output_dir),
        ],
    )
    return case, weights, output_dir


def test_main_records_nested_prediction_samples_and_full_child_logs(
    monkeypatch, tmp_path, capsys
):
    _case, _weights, output_dir = _configure_main(monkeypatch, tmp_path)

    assert runner.main() == 0

    record = json.loads(capsys.readouterr().out)
    assert record["timing_scope"] == "cli_subprocess"
    assert record["peak_mib"] == 1.0
    assert record["stdout_summary_framing"] == "pure_json"
    assert record["stdout_summary_prefix_present"] is False
    assert record["samples"] == [
        {
            "scores": {"iptm": 0.6, "ptm": 0.5},
            "seed": 101,
            "structure_path": "sample-0.cif",
        },
        {
            "scores": {"iptm": 0.7, "ptm": 0.8},
            "seed": 102,
            "structure_path": "sample-1.cif",
        },
    ]
    assert (output_dir / "cli_stdout.txt").read_text(encoding="utf-8")
    assert (output_dir / "cli_stderr.txt").read_text(encoding="utf-8") == "child stderr"


def test_main_retains_failed_child_record(monkeypatch, tmp_path, capsys):
    _case, _weights, _output_dir = _configure_main(monkeypatch, tmp_path, returncode=4)

    assert runner.main() == 1

    record = json.loads(capsys.readouterr().out)
    assert record["failed"] is True
    assert record["reason"] == "FoldJAX CLI exited 4"
    assert record["returncode"] == 4
    assert record["stderr_tail"] == "child stderr"


def test_main_marks_missing_peak_as_failed(monkeypatch, tmp_path, capsys):
    _case, _weights, _output_dir = _configure_main(
        monkeypatch, tmp_path, write_peak=False
    )

    assert runner.main() == 1

    record = json.loads(capsys.readouterr().out)
    assert record["reason"] == "FoldJAX CLI produced no peak observer result"
    assert record["peak_mib"] is None


def test_main_refuses_stale_output_dir(monkeypatch, tmp_path):
    _case, _weights, output_dir = _configure_main(monkeypatch, tmp_path)
    output_dir.mkdir()
    (output_dir / "old.cif").write_text("stale", encoding="utf-8")

    with pytest.raises(SystemExit):
        runner.main()


@pytest.mark.parametrize("alias", ("input", "weights"))
def test_main_refuses_json_output_alias_to_input_or_weights(
    monkeypatch, tmp_path, alias
):
    case, weights, _output_dir = _configure_main(monkeypatch, tmp_path)
    target = case.job if alias == "input" else weights
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--json-out", str(target)])

    with pytest.raises(SystemExit):
        runner.main()



def test_main_counts_canonical_af3_samples_not_top_rank_alias(
    monkeypatch, tmp_path, capsys
):
    _case, _weights, output_dir = _configure_main(monkeypatch, tmp_path)
    original_child = runner.run_cli_child

    def child_with_af3_alias(argv, environment, timeout):
        returncode, stdout, stderr, elapsed, error = original_child(
            argv, environment, timeout
        )
        summary = json.loads(stdout)
        for index, sample in enumerate(summary["samples"]):
            source = output_dir / f"sample-{index}.cif"
            structure = (
                output_dir
                / f"seed-101_sample-{index:02d}"
                / f"case_seed-101_sample-{index:02d}.cif"
            )
            structure.parent.mkdir()
            source.replace(structure)
            sample["structure_path"] = str(structure)
        (output_dir / "case_model.cif").write_text("top-ranked alias")
        return returncode, json.dumps(summary), stderr, elapsed, error

    monkeypatch.setattr(runner, "run_cli_child", child_with_af3_alias)

    assert runner.main() == 0

    record = json.loads(capsys.readouterr().out)
    assert len(record["samples"]) == 2
    assert len(list(output_dir.rglob("*.cif"))) == 3
