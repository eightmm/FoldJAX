"""A batch under `--keep-going` survives what one pair does, and only that.

Every failure here used to end the whole batch (or, for the manifest, to pass
as a warning with exit status 0); each test pins the run that must still
happen after it.
"""

import errno
import json
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import foldjax
import foldjax.api as api_module
from foldjax.backends.base import Backend
from foldjax.cli import main
from foldjax.registry import backend_override
from foldjax.schema import (
    ModelCapabilities,
    PredictionError,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)


class Scripted(Backend):
    """Fails on the run directories named in ``raises``; predicts the rest."""

    name = "boltz2"

    def __init__(self, raises: dict[str, BaseException] | None = None) -> None:
        self.raises = raises or {}
        self.ran: list[str] = []

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model=self.name, input_formats=("native", "foldjax"))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        error = self.raises.get(request.output_dir.name)
        if error is not None:
            raise error
        self.ran.append(request.output_dir.name)
        return PredictionResult(
            model=self.name,
            samples=(
                PredictionSample(
                    seed=request.seed,
                    coordinates=((0.0, 0.0, 0.0),),
                    scores={"iptm": 0.8},
                ),
            ),
            output_dir=request.output_dir,
        )


def _inputs(tmp_path: Path, *names: str) -> tuple[Path, ...]:
    paths = []
    for name in names:
        path = tmp_path / f"{name}.yaml"
        path.write_text("version: 1\n")
        paths.append(path)
    return tuple(paths)


def _batch(tmp_path: Path, *names: str, **overrides) -> PredictionRequest:
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    fields = {
        "model": "boltz2",
        "inputs": _inputs(tmp_path, *names),
        "weights": weights,
        "output_dir": tmp_path / "out",
        "seed": 7,
        "on_error": "continue",
    }
    fields.update(overrides)
    return PredictionRequest(**fields)


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError("Not supported on cpu."),
        pickle.UnpicklingError("invalid load key, '\\x00'."),
        KeyError("params/missing"),
        RuntimeError("INTERNAL: an XLA failure that is not an OOM"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_keep_going_records_any_exception_and_runs_the_rest(
    tmp_path: Path, error: Exception
) -> None:
    backend = Scripted({"a": error})
    with backend_override("boltz2", lambda: backend):
        report = foldjax.predict_batch(_batch(tmp_path, "a", "b"))

    assert backend.ran == ["b"]
    assert [failure.error_type for failure in report.failures] == [type(error).__name__]
    recorded = json.loads((tmp_path / "out" / "foldjax_failures.json").read_text())
    assert recorded[0]["input"].endswith("a.yaml")


def test_keep_going_still_stops_on_an_interrupt(tmp_path: Path) -> None:
    backend = Scripted({"a": KeyboardInterrupt()})
    with (
        backend_override("boltz2", lambda: backend),
        pytest.raises(KeyboardInterrupt),
    ):
        foldjax.predict_batch(_batch(tmp_path, "a", "b"))
    assert backend.ran == []


def test_a_run_directory_that_is_a_file_is_that_pairs_failure(
    tmp_path: Path,
) -> None:
    (tmp_path / "out" / "boltz2").mkdir(parents=True)
    (tmp_path / "out" / "boltz2" / "a").write_text("not a directory")
    backend = Scripted()
    with backend_override("boltz2", lambda: backend):
        report = foldjax.predict_batch(_batch(tmp_path, "a", "b"))

    assert backend.ran == ["b"]
    assert [failure.error_type for failure in report.failures] == ["NotADirectoryError"]


def test_an_input_that_cannot_be_resolved_is_recorded_not_fatal(
    tmp_path: Path, monkeypatch
) -> None:
    original = api_module.resolve_request

    def resolve(request, **kwargs):
        if request.input.stem == "a":
            raise ValueError("cannot read a")
        return original(request, **kwargs)

    monkeypatch.setattr(api_module, "resolve_request", resolve)
    backend = Scripted()
    with backend_override("boltz2", lambda: backend):
        report = foldjax.predict_batch(_batch(tmp_path, "a", "b"))

    assert backend.ran == ["b"]
    assert report.failures[0].error == "cannot read a"
    assert report.failures[0].seed is None
    assert report.failures[0].output_dir == tmp_path / "out" / "boltz2" / "a"

    # Without `continue`, resolution still refuses before anything runs.
    with backend_override("boltz2", lambda: backend), pytest.raises(ValueError):
        foldjax.predict_batch(_batch(tmp_path, "a", "b", on_error="stop"))


def test_a_clean_pass_removes_the_previous_failure_record(tmp_path: Path) -> None:
    with backend_override("boltz2", lambda: Scripted({"a": KeyError("x")})):
        foldjax.predict_batch(_batch(tmp_path, "a", "b"))
    record = tmp_path / "out" / "foldjax_failures.json"
    assert record.is_file()

    with backend_override("boltz2", lambda: Scripted()):
        report = foldjax.predict_batch(_batch(tmp_path, "a", "b"))

    assert report.ok
    assert not record.exists()


def test_a_manifest_the_disk_refuses_is_a_failure(tmp_path: Path, monkeypatch) -> None:
    from foldjax import manifest

    real = Path.write_text

    def full(self, *args, **kwargs):
        if self.name == manifest.MANIFEST_NAME:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full)
    backend = Scripted()
    with backend_override("boltz2", lambda: backend):
        report = foldjax.predict_batch(_batch(tmp_path, "a"))

    assert backend.ran == ["a"]
    assert not report.results
    assert report.failures[0].error_type == "OSError"
    assert "could not write the run manifest" in report.failures[0].error
    assert not (tmp_path / "out" / "boltz2" / "a" / manifest.MANIFEST_NAME).exists()


_HOLD = textwrap.dedent(
    """
    import fcntl, os, sys, time
    fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.write(fd, str(os.getpid()).encode())
    print("locked", flush=True)
    time.sleep(60)
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX")
def test_a_second_writer_of_one_run_directory_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "out" / "boltz2" / "a"
    directory.mkdir(parents=True)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(directory / api_module.LOCK_NAME)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        backend = Scripted()
        with backend_override("boltz2", lambda: backend):
            report = foldjax.predict_batch(_batch(tmp_path, "a", "b"))
            assert backend.ran == ["b"]
            (failure,) = report.failures
            assert failure.error_type == "PredictionError"
            assert f"process {holder.pid}" in failure.error
            assert api_module.LOCK_NAME in failure.error

            with pytest.raises(PredictionError, match="another FoldJAX run"):
                foldjax.predict_batch(_batch(tmp_path, "a", on_error="stop"))
    finally:
        holder.kill()
        holder.wait()

    # Released by the kernel with its holder; the directory is usable again.
    backend = Scripted()
    with backend_override("boltz2", lambda: backend):
        assert foldjax.predict_batch(_batch(tmp_path, "a")).ok
    assert backend.ran == ["a"]


def test_the_lock_is_released_after_each_pair(tmp_path: Path) -> None:
    import fcntl
    import os

    with backend_override("boltz2", lambda: Scripted({"b": KeyError("x")})):
        foldjax.predict_batch(_batch(tmp_path, "a", "b"))
    for name in ("a", "b"):
        lock = tmp_path / "out" / "boltz2" / name / api_module.LOCK_NAME
        fd = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX")
def test_the_lock_takes_the_umask_so_a_group_can_resume(tmp_path: Path) -> None:
    import os
    import stat

    previous = os.umask(0o002)
    try:
        with backend_override("boltz2", lambda: Scripted()):
            assert foldjax.predict_batch(_batch(tmp_path, "a")).ok
    finally:
        os.umask(previous)
    lock = tmp_path / "out" / "boltz2" / "a" / api_module.LOCK_NAME
    assert stat.S_IMODE(lock.stat().st_mode) == 0o664


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX")
def test_another_accounts_readable_lock_still_locks(tmp_path: Path) -> None:
    """A ``0644`` lock another member made: this run only needs to read it."""
    directory = tmp_path / "out" / "boltz2" / "a"
    directory.mkdir(parents=True)
    lock = directory / api_module.LOCK_NAME
    lock.write_text("")
    lock.chmod(0o444)
    backend = Scripted()
    with backend_override("boltz2", lambda: backend):
        assert foldjax.predict_batch(_batch(tmp_path, "a")).ok
    assert backend.ran == ["a"]


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX")
def test_an_unopenable_lock_names_itself_and_the_repair(tmp_path: Path) -> None:
    directory = tmp_path / "out" / "boltz2" / "a"
    directory.mkdir(parents=True)
    lock = directory / api_module.LOCK_NAME
    lock.write_text("")
    lock.chmod(0o000)
    try:
        with backend_override("boltz2", lambda: Scripted()):
            report = foldjax.predict_batch(_batch(tmp_path, "a"))
    finally:
        lock.chmod(0o644)
    (failure,) = report.failures
    assert failure.error_type == "PredictionError"
    assert str(lock) in failure.error
    assert "chmod g+rw" in failure.error


def test_cli_keep_going_records_an_unreadable_fasta_and_runs_the_rest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.fasta").write_text(">a\nMKTAYIAKQR\n")
    (batch / "b.fasta").write_text("")
    weights = tmp_path / "weights"
    weights.mkdir()
    backend = Scripted()
    argv = [
        "predict",
        "--model",
        "boltz2",
        "--input",
        str(batch),
        "--weights",
        str(weights),
        "--output-dir",
        str(tmp_path / "out"),
        "--msa",
        "single",
        "--seed",
        "1",
        "--json",
    ]
    with backend_override("boltz2", lambda: backend):
        code = main([*argv, "--keep-going"])

    assert code == 3
    assert backend.ran == ["a"]
    (record,) = json.loads((tmp_path / "out" / "foldjax_failures.json").read_text())
    assert record["input"] == str(batch / "b.fasta")
    assert record["output_dir"] == str(tmp_path / "out" / "boltz2" / "b")
    assert "no FASTA records" in record["error"]
    assert str(batch / "b.fasta") in capsys.readouterr().err

    # Without --keep-going the same directory still refuses up front.
    with backend_override("boltz2", lambda: Scripted()):
        with pytest.raises(ValueError, match="b.fasta"):
            main(argv)


def test_cli_failures_name_the_fasta_the_caller_wrote(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.fasta").write_text(">a\nMKTAYIAKQR\n")
    (batch / "c.fasta").write_text(">c\nMKTAYIAKQR\n")
    weights = tmp_path / "weights"
    weights.mkdir()
    backend = Scripted({"c": KeyError("broken checkpoint key")})
    with backend_override("boltz2", lambda: backend):
        code = main(
            [
                "predict",
                "--model",
                "boltz2",
                "--input",
                str(batch),
                "--weights",
                str(weights),
                "--output-dir",
                str(tmp_path / "out"),
                "--msa",
                "single",
                "--seed",
                "1",
                "--keep-going",
                "--json",
            ]
        )

    assert code == 3
    err = capsys.readouterr().err
    assert f"boltz2 · {batch / 'c.fasta'} seed 1 failed" in err
    (record,) = json.loads((tmp_path / "out" / "foldjax_failures.json").read_text())
    assert record["source"] == {
        "path": str(batch / "c.fasta"),
        "index": 0,
        "name": "c",
        "kind": "fasta",
    }
    manifest = json.loads(
        (tmp_path / "out" / "boltz2" / "a" / "foldjax_run.json").read_text()
    )
    assert manifest["input"]["source"]["path"] == str(batch / "a.fasta")
    assert manifest["input"]["source"]["kind"] == "fasta"
