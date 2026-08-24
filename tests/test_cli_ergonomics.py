"""The commands and flags a person uses before, during and after a run."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from foldjax.backends.base import Backend
from foldjax.cli import main
from foldjax.manifest import MANIFEST_NAME
from foldjax.registry import backend_override
from foldjax.schema import ModelCapabilities, PredictionResult, PredictionSample

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    monkeypatch.setenv("FOLDJAX_PROGRESS", "0")


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "weights.safetensors"
    # Building a second CLI argv must not rewrite the checkpoint and change its
    # stat identity; real CLI parsing only names an existing weight file too.
    if not path.exists():
        path.write_bytes(b"not a real checkpoint")
    return path


def _job(tmp_path: Path, name: str = "job") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_bare_sequence_becomes_a_job_file_the_plan_can_name(
    tmp_path: Path, capsys
) -> None:
    assert (
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--sequence",
                SEQUENCE,
                "--ligand",
                "ATP",
                "--name",
                "demo",
                "--weights",
                str(_weights(tmp_path)),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)

    generated = Path(plan["input"])
    assert generated.is_file()
    document = json.loads(generated.read_text())
    assert document["name"] == "demo"
    assert [entity["id"] for entity in document["entities"]] == ["A", "B"]
    assert document["entities"][1]["ccd"] == "ATP"
    # A readable stem, because it becomes the output directory.
    assert plan["output_dir"].endswith("demo")


def test_the_same_sequences_reuse_one_generated_file(tmp_path: Path, capsys) -> None:
    """Re-running one command must not litter, and two jobs must not collide."""

    def plan_for(sequence: str) -> str:
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--sequence",
                sequence,
                "--weights",
                str(_weights(tmp_path)),
            ]
        )
        return json.loads(capsys.readouterr().out)["input"]

    assert plan_for(SEQUENCE) == plan_for(SEQUENCE)
    assert plan_for("GGSGGSGGSGGS") != plan_for(SEQUENCE)


def test_a_ligand_without_a_sequence_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a --sequence"):
        main(["plan", "--model", "boltz2", "--ligand", "ATP"])


def test_input_and_sequence_are_alternatives(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alternatives"):
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--input",
                str(_job(tmp_path)),
                "--sequence",
                SEQUENCE,
            ]
        )

    with pytest.raises(ValueError, match="one of --input and --sequence"):
        main(["plan", "--model", "boltz2"])


def test_a_fasta_input_becomes_one_chain_per_record(tmp_path: Path, capsys) -> None:
    fasta = tmp_path / "target.fasta"
    fasta.write_text(f">A\n{SEQUENCE}\n>B\nGGSGGS\n", encoding="utf-8")

    assert (
        main(
            [
                "plan",
                "--model",
                "protenix",
                "--input",
                str(fasta),
                "--weights",
                str(_weights(tmp_path)),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)

    document = json.loads(Path(plan["input"]).read_text())
    assert [entity["id"] for entity in document["entities"]] == ["A", "B"]
    assert document["entities"][0]["sequence"] == SEQUENCE


def test_a_directory_of_jobs_is_a_batch_in_sorted_order(
    tmp_path: Path, capsys
) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    _job(jobs, "second")
    _job(jobs, "first")
    (jobs / "notes.txt").write_text("ignored", encoding="utf-8")

    assert (
        main(
            [
                "plan",
                "--model",
                "protenix",
                "--input",
                str(jobs),
                "--weights",
                str(_weights(tmp_path)),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)

    assert [Path(item["input"]).stem for item in plan] == ["first", "second"]


def test_an_empty_directory_says_so(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="no job files in directory"):
        main(["plan", "--model", "boltz2", "--input", str(empty)])


def test_models_for_reports_which_backends_can_run_a_job(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "ligand.json"
    path.write_text(
        json.dumps(
            {
                "entities": [
                    {"type": "protein", "id": "A", "sequence": SEQUENCE},
                    {"type": "ligand", "id": "L", "ccd": "ATP"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main(["models", "--for", str(path), "--json"]) == 0
    rows = {row["model"]: row for row in json.loads(capsys.readouterr().out)}

    assert rows["boltz2"]["runs"] is True
    assert rows["esmfold2"]["runs"] is True
    assert rows["esmfold2"]["reason"] is None
    # No weights in this store, and the answer arrives anyway.
    assert rows["boltz2"]["weights_ready"] is False


def test_doctor_reports_the_runtime_and_what_is_missing(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["foldjax"]
    assert {row["model"] for row in payload["models"]} == {
        "alphafold3",
        "boltz2",
        "esmfold2",
        "opendde",
        "openfold3",
        "protenix",
    }
    assert all(row["weights_ready"] is False for row in payload["models"])
    assert payload["templates"]


def test_cache_gc_reports_before_it_removes(tmp_path: Path, capsys) -> None:
    from foldjax import paths

    cache = paths.compile_cache_dir() / "boltz2"
    cache.mkdir(parents=True)
    entry = cache / "entry.bin"
    entry.write_bytes(b"x" * 4096)

    assert main(["cache", "gc", "--max-size", "1K"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed_files"] == 1
    assert payload["applied"] is False
    assert entry.is_file(), "a report must not delete anything"

    assert main(["cache", "gc", "--max-size", "1K", "--apply"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is True
    assert not entry.exists()


def test_cache_gc_needs_to_be_told_what_to_reclaim() -> None:
    with pytest.raises(ValueError, match="--older-than"):
        main(["cache", "gc"])


def test_cache_gc_rejects_an_unreadable_size() -> None:
    with pytest.raises(ValueError, match="20G or 500M"):
        main(["cache", "gc", "--max-size", "plenty"])


class _CountingBackend(Backend):
    """A stand-in model that records every run and can be told to fail.

    These tests drive the real execution path rather than replacing it: resume
    and keep-going now live inside `predict_batch`, and faking that away would
    leave the seed-level skipping -- the expensive part -- untested.
    """

    name = "boltz2"

    def __init__(self, fail: Callable[[Path, int], bool] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self._fail = fail or (lambda directory, seed: False)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model=self.name, input_formats=("native", "foldjax"))

    def predict(self, request):
        directory = Path(request.output_dir)
        self.calls.append((directory.name, request.seed))
        if self._fail(directory, request.seed):
            raise MemoryError("out of memory")
        structure = directory / "prediction.cif"
        structure.write_text("data_x\n_entry.id x\n", encoding="utf-8")
        return PredictionResult(
            model=self.name,
            samples=(
                PredictionSample(
                    seed=request.seed,
                    structure_path=structure,
                    scores={"mean_plddt": 88.5},
                ),
            ),
            output_dir=directory,
        )


def _predict_argv(tmp_path: Path, *inputs: Path, out: Path, extra: list[str] = []):
    return [
        "predict",
        "--model",
        "boltz2",
        "--input",
        *[str(path) for path in inputs],
        "--output-dir",
        str(out),
        "--weights",
        str(_weights(tmp_path)),
        "--no-cache",
        *extra,
    ]


def test_predict_prints_a_table_on_a_terminal_and_json_otherwise(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    job = _job(tmp_path)
    argv = _predict_argv(tmp_path, job, out=tmp_path / "out")

    with backend_override("boltz2", _CountingBackend):
        monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
        assert main(argv) == 0
        table = capsys.readouterr().out
        assert "model     boltz2" in table
        assert "mean_plddt" in table

        assert main([*argv, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["model"] == "boltz2"


def test_resume_skips_a_pair_that_already_finished(
    tmp_path: Path, capsys
) -> None:
    job = _job(tmp_path)
    out = tmp_path / "out"
    backend = _CountingBackend()

    with backend_override("boltz2", lambda: backend):
        assert main(_predict_argv(tmp_path, job, out=out, extra=["--json"])) == 0
        capsys.readouterr()
        assert len(backend.calls) == 1
        assert (out / MANIFEST_NAME).is_file()

        argv = _predict_argv(tmp_path, job, out=out, extra=["--json", "--resume"])
        assert main(argv) == 0
        payload = json.loads(capsys.readouterr().out)

    assert len(backend.calls) == 1, "the finished run must not be repeated"
    # A skipped run reports in the same shape as one that executed, so a script
    # using --resume does not have to branch on whether anything ran.
    assert payload["model"] == "boltz2"
    assert payload["samples"][0]["scores"]["mean_plddt"] == 88.5


def test_resume_repeats_only_the_seed_that_failed(tmp_path: Path, capsys) -> None:
    """The expensive case: one bad seed must not cost the three good ones.

    Every seed writes its own manifest, so a run that died on the third has
    already banked the first two. Before this, `--resume` looked only at the
    pair's directory -- which a partial run never gets -- and redid all of them.
    """
    job = _job(tmp_path)
    out = tmp_path / "out"
    seeds = ["--seeds", "0", "1", "2"]

    failing = _CountingBackend(fail=lambda directory, seed: seed == 1)
    with backend_override("boltz2", lambda: failing):
        code = main(
            _predict_argv(
                tmp_path, job, out=out, extra=[*seeds, "--json", "--keep-going"]
            )
        )
    capsys.readouterr()

    assert code == 3
    assert [seed for _name, seed in failing.calls] == [0, 1, 2]
    # No pair-level manifest: the run is not finished, and saying otherwise
    # would make the retry below skip the missing seed.
    assert not (out / MANIFEST_NAME).is_file()
    assert (out / "seed_0" / MANIFEST_NAME).is_file()
    assert not (out / "seed_1" / MANIFEST_NAME).is_file()

    failures = json.loads((out / "foldjax_failures.json").read_text())
    assert failures[0]["seed"] == 1
    assert failures[0]["error_type"] == "MemoryError"

    healthy = _CountingBackend()
    with backend_override("boltz2", lambda: healthy):
        assert (
            main(
                _predict_argv(
                    tmp_path, job, out=out, extra=[*seeds, "--json", "--resume"]
                )
            )
            == 0
        )
    capsys.readouterr()

    assert [seed for _name, seed in healthy.calls] == [1], "only the missing seed"
    assert (out / MANIFEST_NAME).is_file(), "now the pair is finished"


def test_keep_going_finishes_the_batch_and_reports_partial_failure(
    tmp_path: Path, capsys
) -> None:
    first = _job(tmp_path, "first")
    second = _job(tmp_path, "second")
    out = tmp_path / "out"

    def fails_on_first(directory: Path, seed: int) -> bool:
        return directory.name == "first"

    backend = _CountingBackend(fail=fails_on_first)
    with backend_override("boltz2", lambda: backend):
        assert (
            main(
                _predict_argv(
                    tmp_path, first, second, out=out, extra=["--json", "--keep-going"]
                )
            )
            == 3
        )
        payload = json.loads(capsys.readouterr().out)
        assert [name for name, _seed in backend.calls] == ["first", "second"]
        assert len(payload) == 1

        recorded = json.loads((out / "foldjax_failures.json").read_text())
        assert recorded[0]["error_type"] == "MemoryError"
        assert Path(recorded[0]["input"]).stem == "first"

        stopping = _CountingBackend(fail=fails_on_first)
        with backend_override("boltz2", lambda: stopping), pytest.raises(MemoryError):
            main(_predict_argv(tmp_path, first, second, out=tmp_path / "b", extra=[]))
        assert [name for name, _ in stopping.calls] == ["first"]


def test_show_renders_a_finished_directory(tmp_path: Path, capsys) -> None:
    job = _job(tmp_path)
    out = tmp_path / "out"

    with backend_override("boltz2", _CountingBackend):
        main(_predict_argv(tmp_path, job, out=out, extra=["--json"]))
    capsys.readouterr()

    assert main(["show", str(out)]) == 0
    rendered = capsys.readouterr().out
    assert "model     boltz2" in rendered
    assert "mean_plddt" in rendered

    assert main(["show", str(out), "--json"]) == 0
    documents = json.loads(capsys.readouterr().out)
    assert documents[0]["model"] == "boltz2"


def test_show_says_when_a_directory_holds_no_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="a run writes one"):
        main(["show", str(tmp_path)])


def test_interrupting_a_run_exits_cleanly(tmp_path: Path, monkeypatch, capsys) -> None:
    """Ctrl-C is an answer to a question, not a crash to be debugged."""
    import foldjax.cli as cli

    def interrupted(request):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "predict_batch", interrupted)
    argv = _predict_argv(tmp_path, _job(tmp_path), out=tmp_path / "out")
    monkeypatch.setattr("sys.argv", ["foldjax", *argv])

    with pytest.raises(SystemExit) as exit_info:
        cli.entrypoint()

    assert exit_info.value.code == 130
    assert "interrupted" in capsys.readouterr().err


def test_a_permission_error_is_not_a_traceback(tmp_path: Path, monkeypatch) -> None:
    """A read-only output directory is a fact about the machine, not a defect."""
    import foldjax.cli as cli

    def denied(request):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "predict_batch", denied)
    argv = _predict_argv(tmp_path, _job(tmp_path), out=tmp_path / "out")
    monkeypatch.setattr("sys.argv", ["foldjax", *argv])

    with pytest.raises(SystemExit) as exit_info:
        cli.entrypoint()

    assert exit_info.value.code == 2


def test_a_full_disk_names_the_store_and_the_way_to_reclaim_it(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import errno

    import foldjax.cli as cli

    def full(request):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cli, "predict_batch", full)
    argv = _predict_argv(tmp_path, _job(tmp_path), out=tmp_path / "out")
    monkeypatch.setattr("sys.argv", ["foldjax", *argv])

    with pytest.raises(SystemExit):
        cli.entrypoint()

    message = capsys.readouterr().err
    assert "cache gc" in message
    assert str(tmp_path / "store") in message


def test_a_missing_optional_package_names_its_extra(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import foldjax.cli as cli

    def missing(request):
        raise ModuleNotFoundError("No module named 'biotite'", name="biotite")

    monkeypatch.setattr(cli, "predict_batch", missing)
    argv = _predict_argv(tmp_path, _job(tmp_path), out=tmp_path / "out")
    monkeypatch.setattr("sys.argv", ["foldjax", *argv])

    with pytest.raises(SystemExit):
        cli.entrypoint()

    assert "uv sync --extra openfold3-preprocess" in capsys.readouterr().err


def test_the_cli_reports_its_version(capsys) -> None:
    from foldjax import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_structure_input_becomes_a_job_the_plan_can_name(
    tmp_path: Path, capsys
) -> None:
    deposition = tmp_path / "1abc.pdb"
    deposition.write_text(
        "ATOM      1  CA  MET A   1      10.000  10.000  10.000  1.00 20.00  \n"
        "ATOM      2  CA  LYS A   2      11.000  10.000  10.000  1.00 20.00  \n"
        "HETATM    3  PA  ATP B 101      20.000  20.000  20.000  1.00 20.00  \n"
        "END\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--input",
                str(deposition),
                "--weights",
                str(_weights(tmp_path)),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)

    document = json.loads(Path(plan["input"]).read_text())
    assert document["name"] == "1abc"
    assert document["entities"][0]["sequence"] == "MK"
    assert document["entities"][1]["ccd"] == "ATP"
    assert plan["output_dir"].endswith("1abc")


def test_an_affinity_binder_reaches_the_generated_job(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--sequence",
                SEQUENCE,
                "--ligand",
                "ATP",
                "--affinity-binder",
                "B",
                "--name",
                "aff",
                "--weights",
                str(_weights(tmp_path)),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)

    document = json.loads(Path(plan["input"]).read_text())
    assert document["properties"] == [{"affinity": {"binder": "B"}}]


def test_an_affinity_binder_needs_a_generated_job(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="its own properties field"):
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--input",
                str(_job(tmp_path)),
                "--affinity-binder",
                "B",
            ]
        )


def test_doctor_reports_which_msa_search_is_configured(monkeypatch, capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["msa"]["protein"]["kind"] == "remote"
    assert payload["msa"]["rna"]["kind"] == "unavailable"

    monkeypatch.setenv("FOLDJAX_MSA_COMMAND", "/opt/msa/run.sh --db uniref")
    monkeypatch.setenv("FOLDJAX_RNA_MSA_COMMAND", "/opt/msa/rna.sh")
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["msa"]["protein"] == {
        "kind": "local",
        "command": ["/opt/msa/run.sh", "--db", "uniref"],
    }
    assert payload["msa"]["rna"]["kind"] == "local"
