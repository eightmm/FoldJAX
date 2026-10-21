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
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GPU_PARTITION_")
        },
        "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
        "STUB_LOG": str(log),
        "PARTITIONS_GPU": "example-gpu",
        "GRES": "gpu:1",
        "CPUS_PER_GPU": "4",
        "PARTITION_CPU": "example-cpu",
        "MODULE": "",
        "DEFAULT_MSA": "single",
        "FIXED_MSA_DEPTH": "",
        "OPENDDE_PADDING_PROFILE": "",
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


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("profile_source", ["flag", "environment"])
def test_compact_profile_reaches_only_opendde_in_warm_and_verify(
    shell, verify, profile_source
):
    from foldjax.cli import _parser

    run, calls, tmp = shell
    extra = {"SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(tmp)}
    flags = []
    if profile_source == "flag":
        flags = ["--opendde-padding-profile", "compact"]
    else:
        extra["OPENDDE_PADDING_PROFILE"] = "compact"
    completed = run(
        "warm.sbatch",
        *(("--verify-only",) if verify else ()),
        "--model", "opendde", "boltz2", "esmfold2", *flags,
        "--input", "job.json", extra_env=extra,
    )
    assert completed.returncode == 0, completed.stderr
    for row in calls():
        parsed = _parser().parse_args(row["args"])
        assert parsed.opendde_padding_profile == (
            "compact" if parsed.model == ["opendde"] else None
        )


def test_compact_submission_preset_reaches_plan_and_gpu_job(shell):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit", "-m", "opendde", "-m", "esmfold2",
        "-o", str(tmp / "out"), "--opendde-padding-profile", "compact",
        "--", "--input", "job.json",
    )
    assert completed.returncode == 0, completed.stderr
    rows = calls()
    assert len(rows) == 4
    for row in rows:
        assert ("--opendde-padding-profile" in row["args"]) is (
            "opendde" in row["args"]
        )


def test_exact_clears_optional_compact_environment_preset(shell):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit", "-m", "opendde", "-o", str(tmp / "out"),
        "--exact", "--", "--input", "job.json",
        extra_env={"OPENDDE_PADDING_PROFILE": "compact"},
    )
    assert completed.returncode == 0, completed.stderr
    assert all("--opendde-padding-profile" not in row["args"] for row in calls())


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
    assert "--padding" in af3
    for gpu in (boltz, af3):
        assert gpu[gpu.index("--output-dir") + 1] == output
        assert "--batch-output" in gpu
    assert all("--batch-output" in row["args"] for row in rows[:2])
    assert not (tmp / "INJECTED").exists()


@pytest.mark.parametrize(
    "model", ["boltz2", "protenix", "openfold3", "opendde", "esmfold2", "alphafold3"]
)
@pytest.mark.parametrize("depth", ["", "1536", "2048"])
def test_wrapper_defaults_to_2048_and_only_reaches_supported_models(
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
            model in {"boltz2", "protenix", "openfold3", "opendde"}
        )
        if "--fixed-msa" in row["args"]:
            assert row["args"][row["args"].index("--fixed-msa") + 1] == (
                depth or "2048"
            )
        assert ("--padding" in row["args"]) == (model in {"esmfold2", "alphafold3"})
    assert not any("--dependency" in arg for arg in rows[-1]["args"])


@pytest.mark.parametrize("depth", ["2048", "unselected"])
@pytest.mark.parametrize("model", ["boltz2", "alphafold3", "esmfold2"])
def test_exact_and_dry_run_make_no_scheduler_cli_or_filesystem_changes(
    shell, depth, model
):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        model,
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
    assert "--padding" not in completed.stdout
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
            ("predict", "--msa", "single"),
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
    if script == "warm.sbatch":
        raw_args = (
            ["--model", "boltz2", "--batch-output", "--fixed-msa", "2048",
             "--input", "job.json"]
            if args[:1] == ("--verify-only",)
            else ["--model", "boltz2", "--batch-output", "--fixed-msa", "1024"]
        )
    else:
        raw_args = args
    assert calls() == [
        {"tool": "foldjax", "args": [*prefix, *raw_args], "cwd": str(tmp)}
    ]


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize(
    "options,depth",
    [((), "2048"), (("--fixed-msa", "1536"), "1536"), (("--exact",), None)],
)
def test_warm_and_verify_scope_each_model_preset_and_preserve_literal_arguments(
    shell, verify, options, depth
):
    from foldjax.cli import _parser

    run, calls, tmp = shell
    source = "job $(touch INJECTED).json"
    extra = ("--verify-only",) if verify else ()
    completed = run(
        "warm.sbatch", *extra, "--model", "boltz2", "af3", "esmfold2",
        *options, "--", "--input", source, "--output-dir", "out with spaces",
        extra_env={"SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(tmp)},
    )
    assert completed.returncode == 0, completed.stderr
    rows = calls()
    assert len(rows) == 3
    for model, row in zip(("boltz2", "alphafold3", "esmfold2"), rows, strict=True):
        argv = row["args"]
        parsed = _parser().parse_args(argv)
        assert parsed.model == [model]
        assert parsed.fixed_msa == (int(depth) if model == "boltz2" and depth else None)
        assert parsed.padding is (model != "boltz2" and depth is not None)
        assert parsed.batch_output
        assert argv[-4:] == ["--input", source, "--output-dir", "out with spaces"]
        assert argv[:1] == (["predict"] if verify else ["cache"])
    assert not (tmp / "INJECTED").exists()
    if not verify:
        assert "Warming finished" in completed.stderr


def test_warm_mixed_models_reject_one_explicit_weights_path_before_running(shell):
    run, calls, tmp = shell
    completed = run(
        "warm.sbatch", "--model", "boltz2", "af3", "--weights", "one-model.jax",
        extra_env={"SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(tmp)},
    )
    assert completed.returncode == 2
    assert "cannot be shared" in completed.stderr
    assert not calls()


def test_default_warm_sweep_routes_mixed_model_presets_to_valid_cli_calls(shell):
    from foldjax.cli import _parser

    run, calls, tmp = shell
    completed = run(
        "warm.sbatch", "--model", "boltz2", "af3", "esmfold2",
        extra_env={"SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(tmp)},
    )
    assert completed.returncode == 0, completed.stderr
    rows = calls()
    assert len(rows) == 3
    for row in rows:
        parsed = _parser().parse_args(row["args"])
        assert parsed.token_buckets == "auto" and parsed.msa == "single"
        assert (parsed.fixed_msa == 2048) is (parsed.model == ["boltz2"])
        assert parsed.padding is (parsed.model != ["boltz2"])


@pytest.mark.parametrize(
    "args",
    [(), ("--model", "unknown"), ("--model", "af3", "alphafold3"),
     ("--model", "boltz2", "--fixed-msa", "0"),
     ("--model", "boltz2", "--exact", "--fixed-msa", "2048"),
     ("--model", "boltz2", "--exact")],
)
def test_invalid_warm_script_options_run_nothing(shell, args):
    run, calls, tmp = shell
    completed = run(
        "warm.sbatch", *args,
        extra_env={"SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(tmp)},
    )
    assert completed.returncode == 2, completed.stderr
    assert not calls()


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


@pytest.mark.parametrize(
    "options,environment,expected",
    [
        ((), {}, "example-gpu"),
        (("--gpu", "type_a"), {"GPU_PARTITION_TYPE_A": "mapped-gpu"}, "mapped-gpu"),
        (("--gpu", "type_a"), {}, "example-gpu"),
        (
            ("--gpu", "type_a", "--partition", "chosen-gpu"),
            {"GPU_PARTITION_TYPE_A": "mapped-gpu"},
            "chosen-gpu",
        ),
        (("--partition", "chosen-gpu"), {"PARTITIONS_GPU": ""}, "chosen-gpu"),
        (
            ("--gpu", "type_a"),
            {"PARTITIONS_GPU": "", "GPU_PARTITION_TYPE_A": "mapped-gpu"},
            "mapped-gpu",
        ),
        (
            ("--gpu", "type-a+v2"),
            {"GPU_PARTITION_TYPE_A_V2": "mapped-gpu"},
            "mapped-gpu",
        ),
    ],
)
def test_gpu_partition_precedence_is_explicit_then_type_mapping_then_base(
    shell, options, environment, expected
):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz2",
        "-o",
        str(tmp / "out"),
        *options,
        "--",
        "--input",
        "job.json",
        extra_env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    gpu = calls()[-1]["args"]
    assert f"--partition={expected}" in gpu
    assert sum(value.startswith("--partition=") for value in gpu) == 1
    if "--gpu" in options:
        assert f"--gres=gpu:{options[options.index('--gpu') + 1]}:1" in gpu


@pytest.mark.parametrize("default", ["single", "auto"])
def test_default_msa_site_setting_controls_prefetch_and_prediction(shell, default):
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
        extra_env={"DEFAULT_MSA": default},
    )
    assert completed.returncode == 0, completed.stderr
    rows = calls()
    assert len(rows) == (3 if default == "auto" else 2)
    for row in (rows[0], rows[-1]):
        assert row["args"][row["args"].index("--msa") + 1] == default
    if default == "auto":
        assert "--partition=example-cpu" in rows[1]["args"]
        assert "--dependency=afterok:701" in rows[-1]["args"]


def test_explicit_msa_overrides_a_valid_site_default(shell):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz2",
        "-o",
        str(tmp / "out"),
        "--msa",
        "single",
        "--",
        "--input",
        "job.json",
        extra_env={"DEFAULT_MSA": "auto"},
    )
    assert completed.returncode == 0, completed.stderr
    assert len(calls()) == 2
    assert "--dependency=afterok:701" not in calls()[-1]["args"]


def test_invalid_default_msa_is_rejected_without_calls(shell):
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
        extra_env={"DEFAULT_MSA": "invalid"},
    )
    assert completed.returncode == 2
    assert "DEFAULT_MSA" in completed.stderr
    assert not calls()


@pytest.mark.parametrize("models", [("boltz2",), ("boltz2", "protenix")])
@pytest.mark.parametrize("input_count", [1, 2])
def test_model_jobs_delegate_one_output_namespace_to_foldjax(
    shell, models, input_count
):
    from foldjax.api import resolve_requests
    from foldjax.cli import _parser, _request

    run, calls, tmp = shell
    files = []
    for index in range(input_count):
        path = tmp / f"job-{index}.json"
        path.write_text('{"entities": []}')
        files.append(str(path))
    weights = tmp / "weights"
    weights.write_bytes(b"fixture weights")
    output = tmp / "out"
    model_args = [value for model in models for value in ("-m", model)]
    completed = run(
        "foldjax-submit",
        *model_args,
        "-o",
        str(output),
        "--",
        "--input",
        *files,
        "--weights",
        str(weights),
    )
    assert completed.returncode == 0, completed.stderr
    plans = [row["args"] for row in calls() if row["tool"] == "foldjax"]
    gpu_jobs = [row["args"] for row in calls() if row["tool"] == "sbatch"]
    destinations = []
    for model, plan, gpu in zip(models, plans, gpu_jobs, strict=True):
        assert plan[plan.index("--output-dir") + 1] == str(output)
        assert gpu[gpu.index("--output-dir") + 1] == str(output)
        assert "--batch-output" in plan and "--batch-output" in gpu
        args = _parser().parse_args(plan)
        request = _request(args, jobs_root=tmp / "generated")
        resolved = resolve_requests(request, draw_seeds=False)
        expected = [output / model / Path(path).stem for path in files]
        assert [item.output_dir for item in resolved] == expected
        destinations.extend(expected)
    assert len(set(destinations)) == len(models) * input_count


def test_wrapper_output_policy_cannot_be_overridden_in_passthrough(shell):
    run, calls, tmp = shell
    completed = run(
        "foldjax-submit",
        "-m",
        "boltz2",
        "-o",
        str(tmp / "out"),
        "--",
        "--batch-output",
        "--input",
        "job.json",
    )
    assert completed.returncode == 2
    assert "managed by this wrapper" in completed.stderr
    assert not calls()
