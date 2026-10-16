"""Optional native cuEquivariance imports for tests that execute its kernels."""

from __future__ import annotations

from collections.abc import Callable
from subprocess import CompletedProcess
from typing import Any

import pytest


def require_cueq(loader: Callable[[], Any] | None = None) -> Any:
    from foldjax.models._cueq import load_cueq

    try:
        return (loader or load_cueq)()
    except (ImportError, OSError) as error:
        pytest.skip(f"cuEquivariance JAX runtime is unavailable: {error}")
    except RuntimeError as error:
        if not isinstance(error.__cause__, (ImportError, OSError)):
            raise
        pytest.skip(f"cuEquivariance JAX runtime is unavailable: {error.__cause__}")


def skip_missing_cueq_subprocess(completed: CompletedProcess[str]) -> None:
    if completed.returncode == 0:
        return
    for line in completed.stderr.splitlines():
        missing_package = (
            line.startswith(("ImportError:", "ModuleNotFoundError:"))
            and "cuequivariance" in line
        )
        missing_library = (
            line.startswith("OSError:")
            and "libcue" in line
            and "cannot open shared object file" in line
        )
        if missing_package or missing_library:
            pytest.skip(f"cuEquivariance JAX runtime is unavailable: {line}")
