"""Exercise submission examples with argv-recording scheduler/CLI stubs only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "slurm"


@pytest.fixture
def shell(tmp_path: Path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "calls.jsonl"
    stub = f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
path = Path(os.environ['STUB_LOG'])
name = Path(sys.argv[0]).name
old = path.read_text().splitlines() if path.exists() else []
with path.open('a') as handle:
    row = {{'tool': name, 'args': sys.argv[1:], 'cwd': os.getcwd()}}
    handle.write(json.dumps(row) + '\\n')
if name == 'sbatch':
    count = sum(json.loads(line)['tool'] == 'sbatch' for line in old)
    print(str(701 + count) + ';example-cluster')
elif os.environ.get('FAIL_PLAN') and sys.argv[1:2] == ['plan']:
    sys.exit(3)
"""
    for name in ("sbatch", "foldjax"):
        target = binaries / name
        target.write_text(stub)
        target.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
        "STUB_LOG": str(log),
        "PARTITIONS_GPU": "example-gpu",
        "GRES": "gpu:1",
        "CPUS_PER_GPU": "4",
        "PARTITION_CPU": "example-cpu",
        "MODULE": "",
        "FIXED_MSA_DEPTH": "",
        "FAIL_PLAN": "",
    }

    def run(script: str, *args: str, extra_env=None):
        return subprocess.run(
            ["bash", str(SCRIPTS / script), *args],
            cwd=tmp_path,
            env={**env, **(extra_env or {})},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def calls():
        return (
            [json.loads(line) for line in log.read_text().splitlines()]
            if log.exists()
            else []
        )

    return run, calls, tmp_path


def test_auto_prefetch_dependencies_arrays_and_literal_passthrough(shell):
    run, calls, tmp = shell
    first = str(tmp / "first job.json")
    second = str(tmp / "$(touch INJECTED).json")
    output = str(tmp / "results with spaces")
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz-2",
        "-m",
        "af3",
        "-o",
        output,
        "--msa",
        "auto",
        "--gpu",
        "example_type",
        "--time",
        "01:20:00",
        "--shards",
        "3",
        "--fixed-msa",
        "1536",
        "--",
        "--input",
        first,
        second,
        "--msa-pairing",
        "complete",
        "--seed",
        "7",
    )
    assert completed.returncode == 0, completed.stderr
    rows = calls()
    assert [row["tool"] for row in rows] == [
        "foldjax",
        "foldjax",
        "sbatch",
        "sbatch",
        "sbatch",
    ]
    assert all(row["args"][0] == "plan" for row in rows[:2])
    cpu, boltz, af3 = [row["args"] for row in rows[2:]]
    assert "--partition=example-cpu" in cpu
    assert "--cpus-per-task=2" in cpu and "--mem=8G" in cpu
    script = cpu.index(str(SCRIPTS / "msa.sbatch"))
    assert cpu[script + 1 :] == [
        "--model",
        "boltz2",
        "alphafold3",
        "--msa-pairing",
        "complete",
        "--",
        first,
        second,
    ]
    for gpu in (boltz, af3):
        for option in (
            "--dependency=afterok:701",
            "--kill-on-invalid-dep=yes",
            "--array=0-2",
            "--gres=gpu:example_type:1",
            "--time=01:20:00",
            "--resume",
            "--keep-going",
        ):
            assert option in gpu
        assert gpu[gpu.index("--shard") + 1] == "auto"
        assert (
            gpu[-7:]
            == ["--input", first, second, "--msa-pairing", "complete", "--seed", "7"][
                -7:
            ]
        )
    assert boltz[boltz.index("--fixed-msa") + 1] == "1536"
    assert "--fixed-msa" not in af3
    assert boltz[boltz.index("--output-dir") + 1] == output + "/boltz2"
    assert af3[af3.index("--output-dir") + 1] == output + "/alphafold3"
    assert not (tmp / "INJECTED").exists()


@pytest.mark.parametrize(
    "model", ["boltz2", "protenix", "openfold3", "opendde", "esmfold2", "alphafold3"]
)
@pytest.mark.parametrize("depth", ["", "2048"])
def test_fixed_msa_has_no_hidden_default_and_only_reaches_supported_models(
    shell, model, depth
):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        model,
        "-o",
        str(tmp / "out"),
        "--",
        "--input",
        "job.json",
        extra_env={"FIXED_MSA_DEPTH": depth},
    )
    assert completed.returncode == 0, completed.stderr
    rows = calls()
    assert len(rows) == 2
    for row in rows:
        assert ("--fixed-msa" in row["args"]) == (
            bool(depth) and model in {"boltz2", "protenix", "openfold3", "opendde"}
        )
    assert not any("--dependency" in arg for arg in rows[-1]["args"])


@pytest.mark.parametrize("depth", ["2048", "unselected"])
def test_exact_and_dry_run_make_no_scheduler_cli_or_filesystem_changes(shell, depth):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz2",
        "-o",
        str(tmp / "out"),
        "--exact",
        "--dry-run",
        "--",
        "--input",
        "input with spaces.json",
        extra_env={"FIXED_MSA_DEPTH": depth},
    )
    assert completed.returncode == 0, completed.stderr
    assert "--fixed-msa" not in completed.stdout
    assert "--resume" in completed.stdout and "--keep-going" in completed.stdout
    assert not calls() and not (tmp / "out").exists()


def test_an_explicit_single_shard_is_an_array(shell):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz2",
        "-o",
        str(tmp / "out"),
        "--shards",
        "1",
        "--",
        "--input",
        "job.json",
    )
    assert completed.returncode == 0, completed.stderr
    args = calls()[-1]["args"]
    assert "--array=0-0" in args
    assert args[args.index("--shard") + 1] == "auto"


def test_duplicate_model_aliases_cannot_race_one_output_directory(shell):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz2",
        "-m",
        "boltz",
        "-o",
        str(tmp / "out"),
        "--",
        "--input",
        "job.json",
    )
    assert completed.returncode == 2
    assert "duplicate model" in completed.stderr
    assert not calls()


def test_failed_preflight_prevents_all_submissions(shell):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz2",
        "-o",
        str(tmp / "out"),
        "--",
        "--input",
        "job.json",
        extra_env={"FAIL_PLAN": "1"},
    )
    assert completed.returncode == 3
    assert [row["tool"] for row in calls()] == ["foldjax"]
    assert not (tmp / "out").exists()


@pytest.mark.parametrize(
    "options",
    [
        ("--msa", "bad"),
        ("--shards", "0"),
        ("--fixed-msa", "0"),
        ("--msa", "auto"),
        ("--exact", "--fixed-msa", "1024"),
    ],
)
def test_invalid_wrapper_options_submit_nothing(shell, options):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz2",
        "-o",
        str(tmp / "out"),
        *options,
        "--",
        "--sequence",
        "ACDE",
    )
    assert completed.returncode == 2
    assert not calls()


@pytest.mark.parametrize(
    "script,args,prefix",
    [
        (
            "msa.sbatch",
            ("--model", "boltz2", "--", "job with spaces.json"),
            ("msa", "prefetch"),
        ),
        (
            "predict.sbatch",
            ("--model", "boltz2", "--input", "job with spaces.json"),
            ("predict",),
        ),
        (
            "warm.sbatch",
            ("--model", "boltz2", "--fixed-msa", "1024"),
            ("cache", "warm", "--token-buckets", "auto", "--msa", "single"),
        ),
        (
            "warm.sbatch",
            ("--verify-only", "--", "--model", "boltz2", "--input", "job.json"),
            ("predict",),
        ),
    ],
)
def test_batch_scripts_execute_only_the_expected_cli_with_literal_arguments(
    shell, script, args, prefix
):
    run, calls, tmp = shell
    completed = run(
        script, *args, extra_env={"SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(tmp)}
    )
    assert completed.returncode == 0, completed.stderr
    raw_args = args[2:] if args[:1] == ("--verify-only",) else args
    assert calls() == [
        {"tool": "foldjax", "args": [*prefix, *raw_args], "cwd": str(tmp)}
    ]


@pytest.mark.parametrize("script", ["msa.sbatch", "predict.sbatch", "warm.sbatch"])
def test_batch_scripts_refuse_running_outside_slurm(shell, script):
    run, calls, _ = shell
    completed = run(script, extra_env={"SLURM_JOB_ID": ""})
    assert completed.returncode != 0
    assert not calls()


def test_submission_scripts_have_valid_bash_syntax():
    for name in ("foldjax-submit", "msa.sbatch", "predict.sbatch", "warm.sbatch"):
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPTS / name)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
