from __future__ import annotations

import os
import subprocess
import sys


def _import_environment(overrides: dict[str, str] | None = None) -> list[str]:
    environment = os.environ.copy()
    environment.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    environment.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    environment.update(overrides or {})
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, foldjax.models.chai; print("  # noqa: S607
            "os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'], "
            "os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout.strip().split()


def test_chai_jax_sets_bounded_allocator_defaults_before_jax_import() -> None:
    assert _import_environment() == ["false", "0.90"]


def test_explicit_allocator_environment_takes_precedence() -> None:
    assert _import_environment(
        {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.50",
        }
    ) == ["true", "0.50"]
