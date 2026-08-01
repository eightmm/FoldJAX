from __future__ import annotations

import re
from pathlib import Path

import pytest

from . import conftest
from .conftest import _unavailable


class _ParityConfig:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def getoption(self, name: str) -> bool:
        assert name == "--run-official-parity"
        return self.enabled


def test_tracked_tests_and_scripts_do_not_contain_developer_absolute_paths() -> None:
    # Now that the port is vendored, this covers every FoldJAX test and source
    # file rather than only the standalone chai_jax checkout.
    repository = Path(__file__).resolve().parents[3]
    developer_path = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")
    offenders = []
    for directory in (repository / "tests", repository / "src"):
        for path in directory.rglob("*.py"):
            if path == Path(__file__):
                continue
            if developer_path.search(path.read_text(encoding="utf-8")):
                offenders.append(path.relative_to(repository).as_posix())

    assert offenders == []


def test_missing_official_dependency_fails_when_opted_in() -> None:
    with pytest.raises(pytest.fail.Exception, match="missing fixture"):
        _unavailable(_ParityConfig(enabled=True), "missing fixture")


def test_missing_official_dependency_skips_without_opt_in() -> None:
    with pytest.raises(pytest.skip.Exception, match="missing fixture"):
        _unavailable(_ParityConfig(enabled=False), "missing fixture")


def test_missing_torch_fails_before_official_test_collection(monkeypatch) -> None:
    monkeypatch.setattr(conftest, "find_spec", lambda _name: None)

    with pytest.raises(pytest.UsageError, match="torch-bridge"):
        conftest._require_official_runtime(_ParityConfig(enabled=True))


def test_portable_collection_does_not_require_torch(monkeypatch) -> None:
    monkeypatch.setattr(conftest, "find_spec", lambda _name: None)

    conftest._require_official_runtime(_ParityConfig(enabled=False))


def test_official_tests_do_not_assume_a_sibling_chai_checkout() -> None:
    # ``parents[4]`` is the shared workspace root from this suite's vendored
    # location, so it is the spelling an official-parity test would reach for.
    suite = Path(__file__).resolve().parent
    offenders = []
    for path in suite.glob("*.py"):
        if path == Path(__file__):
            continue
        source = path.read_text(encoding="utf-8")
        if "official_parity" in source and "parents[4]" in source:
            offenders.append(path.relative_to(suite).as_posix())

    assert offenders == []
