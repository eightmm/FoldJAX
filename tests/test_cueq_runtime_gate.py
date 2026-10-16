"""A missing native dependency skips kernel coverage; code failures still fail."""

from __future__ import annotations

from subprocess import CompletedProcess

import pytest

from tests.cueq_runtime import require_cueq, skip_missing_cueq_subprocess


@pytest.mark.parametrize("cause", [ImportError("no package"), OSError("libcue_ops.so")])
@pytest.mark.parametrize("wrapped", [True, False])
def test_native_import_failures_skip(cause: Exception, wrapped: bool) -> None:
    def unavailable():
        if wrapped:
            raise RuntimeError("cuEquivariance JAX is required") from cause
        raise cause

    with pytest.raises(pytest.skip.Exception, match="runtime is unavailable"):
        require_cueq(unavailable)


def test_unrelated_runtime_failures_are_not_skipped() -> None:
    def broken():
        raise RuntimeError("kernel shape is invalid")

    with pytest.raises(RuntimeError, match="kernel shape is invalid"):
        require_cueq(broken)


def test_available_runtime_is_preserved() -> None:
    runtime = object()
    assert require_cueq(lambda: runtime) is runtime


@pytest.mark.parametrize("stderr", [
    "OSError: libcue_ops.so: cannot open shared object file: No such file",
    "ModuleNotFoundError: No module named 'cuequivariance_jax'",
])
def test_subprocess_native_import_failures_skip(stderr: str) -> None:
    completed = CompletedProcess([], 1, stdout="", stderr=stderr)
    with pytest.raises(pytest.skip.Exception, match="runtime is unavailable"):
        skip_missing_cueq_subprocess(completed)


@pytest.mark.parametrize("stderr", [
    "RuntimeError: a cuEquivariance kernel produced wrong output",
    "OSError: another.so: cannot open shared object file: No such file",
])
def test_subprocess_other_failures_remain_failures(stderr: str) -> None:
    completed = CompletedProcess([], 1, stdout="", stderr=stderr)
    skip_missing_cueq_subprocess(completed)
    assert completed.returncode == 1
