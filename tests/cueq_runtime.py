"""Optional native cuEquivariance imports for tests that execute its kernels."""

from __future__ import annotations

import builtins
import re
from collections.abc import Callable
from subprocess import CompletedProcess
from typing import Any

import pytest

_REQUIRED_CUEQ = "cuEquivariance JAX is required"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_EXCEPTION_LINE = re.compile(r"^[ \t]*([A-Za-z_][\w.]*):[ \t]*(.*)$", re.MULTILINE)
_CUEQ_PRELOAD_FRAME = re.compile(
    r'File "(?:[^"\n]*[/\\])?foldjax[/\\]models[/\\]_cueq\.py", '
    r"line \d+, in preload_bundled_nvrtc\r?$",
    re.MULTILINE,
)
_CHAIN_SEPARATORS = (
    "The above exception was the direct cause of the following exception:",
    "During handling of the above exception, another exception occurred:",
)


def require_cueq(loader: Callable[[], Any] | None = None) -> Any:
    from foldjax.models._cueq import load_cueq

    try:
        return (loader or load_cueq)()
    except (ImportError, OSError) as error:
        pytest.skip(f"cuEquivariance JAX runtime is unavailable: {error}")
    except RuntimeError as error:
        if not str(error).startswith(_REQUIRED_CUEQ) or not isinstance(
            error.__cause__, (ImportError, OSError)
        ):
            raise
        pytest.skip(f"cuEquivariance JAX runtime is unavailable: {error.__cause__}")


def _import_error_type(name: str) -> bool:
    error_type = getattr(builtins, name.rsplit(".", maxsplit=1)[-1], None)
    return isinstance(error_type, type) and issubclass(
        error_type, (ImportError, OSError)
    )


def skip_missing_cueq_subprocess(completed: CompletedProcess[str]) -> None:
    if completed.returncode <= 0:
        return
    stderr = _ANSI.sub("", completed.stderr)
    errors = list(_EXCEPTION_LINE.finditer(stderr))
    if not errors:
        return
    last = errors[-1]
    name, message = last.groups()
    wrapped = name == "RuntimeError" and message.startswith(_REQUIRED_CUEQ)
    if wrapped and len(errors) > 1:
        cause = errors[-2]
        chain = stderr[cause.end() : last.start()]
        if _import_error_type(cause[1]) and any(
            separator in chain for separator in _CHAIN_SEPARATORS
        ):
            pytest.skip(f"cuEquivariance JAX runtime is unavailable: {cause[2]}")
    if not _import_error_type(name):
        return
    lowered = message.lower()
    missing_package = "no module named" in lowered and "cuequivariance" in lowered
    traceback_start = stderr.rfind(
        "Traceback (most recent call last):", 0, last.start()
    )
    context = stderr[traceback_start : last.start()] if traceback_start >= 0 else ""
    cueq_import = "cuequivariance_jax" in context or "cuequivariance_ops_jax" in context
    cueq_preload = _CUEQ_PRELOAD_FRAME.search(context) is not None
    missing_library = ("libcue" in lowered or cueq_import or cueq_preload) and (
        "cannot open shared object file" in lowered or "undefined symbol" in lowered
    )
    # Earlier caught/logged import failures must not hide the final kernel or
    # assertion failure; raw CUDA dependencies need a cuEq import/preload frame.
    if missing_package or missing_library:
        pytest.skip(f"cuEquivariance JAX runtime is unavailable: {last[0].strip()}")
