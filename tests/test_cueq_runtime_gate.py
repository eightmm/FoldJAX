"""A missing native dependency skips kernel coverage; code failures still fail."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
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


@pytest.mark.parametrize(
    "stderr",
    [
        "OSError: libcue_ops.so: cannot open shared object file: No such file",
        "ModuleNotFoundError: No module named 'cuequivariance_jax'",
    ],
)
def test_subprocess_native_import_failures_skip(stderr: str) -> None:
    completed = CompletedProcess([], 1, stdout="", stderr=stderr)
    with pytest.raises(pytest.skip.Exception, match="runtime is unavailable"):
        skip_missing_cueq_subprocess(completed)


@pytest.mark.parametrize(
    "stderr",
    [
        "RuntimeError: a cuEquivariance kernel produced wrong output",
        "OSError: another.so: cannot open shared object file: No such file",
    ],
)
def test_subprocess_other_failures_remain_failures(stderr: str) -> None:
    completed = CompletedProcess([], 1, stdout="", stderr=stderr)
    skip_missing_cueq_subprocess(completed)
    assert completed.returncode == 1


@pytest.mark.parametrize("cause", [ImportError("bad import"), OSError("kernel data")])
def test_an_unrelated_runtime_wrapper_does_not_turn_into_a_skip(cause) -> None:
    def broken():
        raise RuntimeError("kernel execution failed") from cause

    with pytest.raises(RuntimeError, match="kernel execution failed"):
        require_cueq(broken)


def _python_child(
    directory: Path,
    source: str,
    *,
    colored: bool = False,
) -> CompletedProcess[str]:
    prefix = ""
    if colored:
        prefix = textwrap.dedent("""\
            import sys, traceback
            def colored_error(kind, error, tb):
                rendered = ''.join(traceback.format_exception(kind, error, tb))
                sys.stderr.write('\\x1b[31m' + '\\n'.join(
                    '  ' + line for line in rendered.splitlines()
                ) + '\\x1b[0m\\n')
            sys.excepthook = colored_error
        """)
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = (
        str(directory) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    return subprocess.run(
        [sys.executable, "-c", prefix + textwrap.dedent(source)],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _python_with_broken_cueq(
    directory: Path,
    source: str,
    *,
    error: str = "OSError",
    message: str = "libcuda.so.1: cannot open shared object file: No such file",
    colored: bool = False,
) -> CompletedProcess[str]:
    # A real import in a fresh child fails without depending on the host's CUDA
    # installation or propagating the parent's monkeypatch into that child.
    (directory / "cuequivariance_jax.py").write_text(
        f"raise {error}({message!r})\n", encoding="utf-8"
    )
    return _python_child(directory, source, colored=colored)


def _python_with_broken_preload(
    directory: Path,
    source: str,
    *,
    error: str = "OSError",
    message: str = "libcuda.so.1: cannot open shared object file: No such file",
    colored: bool = False,
) -> CompletedProcess[str]:
    nvidia = directory / "nvidia"
    library = nvidia / "cu13" / "lib" / "libnvrtc.so.13"
    library.parent.mkdir(parents=True)
    library.touch()
    setup = textwrap.dedent(f"""\
        from importlib.machinery import ModuleSpec
        from foldjax.models import _cueq
        spec = ModuleSpec('nvidia', loader=None, is_package=True)
        spec.submodule_search_locations = [{str(nvidia)!r}]
        original_find_spec = _cueq.importlib.util.find_spec
        _cueq.importlib.util.find_spec = lambda name: (
            spec if name == 'nvidia' else original_find_spec(name)
        )
        def broken_cdll(*args, **kwargs):
            raise {error}({message!r})
        _cueq.ctypes.CDLL = broken_cdll
    """)
    return _python_child(directory, setup + textwrap.dedent(source), colored=colored)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("colored", [False, True])
@pytest.mark.parametrize(
    "error,message",
    [
        ("OSError", "libcuda.so.1: cannot open shared object file: No such file"),
        ("ImportError", "libcue_ops.so: undefined symbol: missing_cuda_symbol"),
    ],
)
def test_real_child_native_import_failure_skips(
    tmp_path, wrapped, colored, error, message
) -> None:
    source = (
        "from foldjax.models._cueq import load_cueq; load_cueq()"
        if wrapped
        else "import cuequivariance_jax"
    )
    completed = _python_with_broken_cueq(
        tmp_path,
        source,
        error=error,
        message=message,
        colored=colored,
    )
    assert completed.returncode == 1, completed.stderr
    assert "cuequivariance_jax.py" in completed.stderr
    assert error in completed.stderr
    if colored:
        assert "\x1b[" in completed.stderr
    with pytest.raises(pytest.skip.Exception, match="runtime is unavailable"):
        skip_missing_cueq_subprocess(completed)


@pytest.mark.parametrize("colored", [False, True])
@pytest.mark.parametrize(
    "error,message",
    [
        ("OSError", "libcuda.so.1: cannot open shared object file: No such file"),
        ("ImportError", "libnvrtc.so.13: undefined symbol: missing_cuda_symbol"),
    ],
)
def test_real_child_cueq_preload_failure_skips(
    tmp_path, colored, error, message
) -> None:
    completed = _python_with_broken_preload(
        tmp_path, "_cueq.load_cueq()", error=error, message=message, colored=colored
    )
    assert completed.returncode == 1, completed.stderr
    assert "preload_bundled_nvrtc" in completed.stderr
    assert "cuequivariance_jax" not in completed.stderr
    with pytest.raises(pytest.skip.Exception, match="runtime is unavailable"):
        skip_missing_cueq_subprocess(completed)


@pytest.mark.parametrize("error", ["AssertionError", "OSError"])
def test_real_child_final_failure_survives_an_earlier_cueq_preload_error(
    tmp_path, error
) -> None:
    completed = _python_with_broken_preload(
        tmp_path,
        f"""\
        import traceback
        try:
            _cueq.load_cueq()
        except OSError:
            traceback.print_exc()
        raise {error}('another.so: cannot open shared object file: No such file')
    """,
    )
    assert completed.returncode == 1
    assert "preload_bundled_nvrtc" in completed.stderr
    skip_missing_cueq_subprocess(completed)


@pytest.mark.parametrize("filename", ["other.py", "_cueq.py", "foldjax/other/_cueq.py"])
def test_real_child_non_cueq_preload_failure_remains_failure(tmp_path, filename):
    module = tmp_path / filename
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        "def preload_bundled_nvrtc():\n"
        "    raise OSError('libcuda.so.1: cannot open shared object file')\n",
        encoding="utf-8",
    )
    completed = _python_child(
        tmp_path,
        f"import runpy; runpy.run_path({str(module)!r})['preload_bundled_nvrtc']()",
    )
    assert completed.returncode == 1
    assert "preload_bundled_nvrtc" in completed.stderr
    skip_missing_cueq_subprocess(completed)


def test_real_child_cueq_preload_non_loader_error_remains_failure(tmp_path) -> None:
    completed = _python_with_broken_preload(
        tmp_path, "_cueq.load_cueq()", message="kernel data is invalid"
    )
    assert completed.returncode == 1
    assert "preload_bundled_nvrtc" in completed.stderr
    skip_missing_cueq_subprocess(completed)


@pytest.mark.parametrize(
    "error,message",
    [
        ("AssertionError", "kernel result differs"),
        ("RuntimeError", "cuEquivariance kernel execution failed"),
        ("ImportError", "another.so: cannot open shared object file: No such file"),
    ],
)
def test_real_child_final_failure_survives_an_earlier_logged_import_error(
    tmp_path, error, message
) -> None:
    completed = _python_with_broken_cueq(
        tmp_path,
        f"""\
        import traceback
        from foldjax.models._cueq import load_cueq
        try:
            load_cueq()
        except RuntimeError:
            traceback.print_exc()
        raise {error}({message!r})
    """,
    )
    assert completed.returncode == 1
    assert "cuEquivariance JAX is required by this backend" in completed.stderr
    assert f"{error}: {message}" in completed.stderr
    skip_missing_cueq_subprocess(completed)


def test_real_child_success_with_logged_import_noise_is_not_skipped(tmp_path) -> None:
    completed = _python_with_broken_cueq(
        tmp_path,
        """\
        import traceback
        from foldjax.models._cueq import load_cueq
        try:
            load_cueq()
        except RuntimeError:
            traceback.print_exc()
        print('probe succeeded')
    """,
    )
    assert completed.returncode == 0
    assert "cuEquivariance JAX is required by this backend" in completed.stderr
    skip_missing_cueq_subprocess(completed)


def test_real_child_kernel_cause_under_required_message_is_not_skipped(
    tmp_path,
) -> None:
    completed = _python_with_broken_cueq(
        tmp_path,
        """\
        try:
            raise AssertionError('kernel output differs')
        except AssertionError as error:
            raise RuntimeError(
                'cuEquivariance JAX is required by this backend'
            ) from error
    """,
    )
    assert completed.returncode == 1
    skip_missing_cueq_subprocess(completed)


@pytest.mark.parametrize(
    "stderr",
    [
        "ImportError: cannot import name 'renamed_api' from 'cuequivariance_jax'",
        "OSError: libcuda.so.1: cannot open shared object file: No such file",
        "OSError: libcuda.so.1: undefined symbol: runtime_failure",
        "ImportError: libcue_ops.so: cannot open shared object file\n"
        "AssertionError: wrong output",
    ],
)
def test_ambiguous_or_non_import_native_failures_remain_failures(stderr) -> None:
    skip_missing_cueq_subprocess(CompletedProcess([], 1, stdout="", stderr=stderr))


def test_a_signalled_child_is_not_skipped_from_its_old_import_log() -> None:
    skip_missing_cueq_subprocess(
        CompletedProcess(
            [],
            -6,
            stdout="",
            stderr="OSError: libcue_ops.so: cannot open shared object file",
        )
    )
