"""`Job.store()` -- the managed-location sibling of `Job.write(path)`.

The command line has always written `--sequence` jobs into the FoldJAX store,
because generated input is still input: it is hashed into the run manifest and
it is what `plan` prints. The Python API had no way to ask for the same thing,
so callers invented a filename, which landed in the working directory and could
disagree with the job's own `name`. That code lived privately in `cli.py`; it
is now a method, and the CLI calls it.
"""

from __future__ import annotations

from pathlib import Path

from foldjax import Job


def _job(sequence: str = "MKTAYIAKQRQISFVKSHFSRQ", name: str = "store-test") -> Job:
    return Job.from_sequences((sequence,), name=name)


def test_a_stored_job_lands_in_the_store_not_the_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    stored = _job().store()
    assert stored.is_file()
    assert tmp_path / "home" in stored.parents
    assert not list(tmp_path.glob("*.json"))


def test_storing_the_same_job_twice_returns_one_file(
    tmp_path: Path, monkeypatch
) -> None:
    """Re-running the same sequences must reuse one file, not accumulate them."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    job = _job()
    assert job.store() == job.store()


def test_two_different_jobs_sharing_a_name_never_share_a_file(
    tmp_path: Path, monkeypatch
) -> None:
    """Both keep the readable stem; the content digest is the directory."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    first = _job("MKTAYIAKQRQISFVKSHFSRQ").store()
    second = _job("MKTAYIAKQRQISFVKSHFSRE").store()
    assert first != second
    assert first.name == second.name == "store-test.json"


def test_the_stem_does_not_depend_on_what_was_stored_before(
    tmp_path: Path, monkeypatch
) -> None:
    """The default output directory is the stem, so it must not drift."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    _job("MKTAYIAKQRQISFVKSHFSRE").store()
    later = _job("MKTAYIAKQRQISFVKSHFSRQ").store()
    fresh_home = tmp_path / "other"
    monkeypatch.setenv("FOLDJAX_HOME", str(fresh_home))
    alone = _job("MKTAYIAKQRQISFVKSHFSRQ").store()
    assert later.stem == alone.stem == "store-test"
    assert later.parent.name == alone.parent.name


def test_store_takes_an_explicit_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    stored = _job().store(tmp_path / "scratch")
    assert tmp_path / "scratch" in stored.parents
    assert not (tmp_path / "home").exists()


def test_the_stored_document_is_the_job(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    job = _job()
    assert Job.read(job.store()).to_document() == job.to_document()


def test_write_still_takes_an_explicit_path(tmp_path: Path) -> None:
    """`store` is the sibling of `write`, not its replacement."""
    target = tmp_path / "somewhere" / "explicit.json"
    assert _job().write(target) == target
    assert target.is_file()
