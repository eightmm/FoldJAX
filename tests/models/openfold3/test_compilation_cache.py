"""The persistent compilation cache actually persists.

Compiling the released architecture takes minutes and scales with sequence length,
so without a cache that cost is paid in every process. The claim worth testing is
not that the config was set but that a *second process* reuses what the first
compiled.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from foldjax.models.openfold3.compilation import (
    default_cache_dir,
    enable_compilation_cache,
)

# Compiles slowly enough to clear the default threshold without needing a model.
_WORKER = """
import sys, jax, jax.numpy as jnp
from foldjax.models.openfold3.compilation import enable_compilation_cache
enable_compilation_cache(sys.argv[1], min_compile_time_secs=0.0)

@jax.jit
def f(x):
    for _ in range(24):
        x = jnp.tanh(x @ x.T) @ x
    return x

print(float(jnp.sum(f(jnp.ones((64, 64))))))
"""


def _run_worker(directory: Path) -> None:
    subprocess.run(
        [sys.executable, "-c", _WORKER, str(directory)],
        check=True,
        capture_output=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )


def _entries(directory: Path) -> set[Path]:
    return {path for path in directory.rglob("*") if path.is_file()}


def test_a_second_process_reuses_the_first_compile(tmp_path: Path) -> None:
    cache = tmp_path / "jit"
    _run_worker(cache)
    after_first = _entries(cache)
    assert after_first, "nothing was written to the cache directory"

    _run_worker(cache)
    after_second = _entries(cache)
    # A cache hit adds nothing; a miss would write a second executable.
    assert after_second == after_first, (
        "the second process compiled again instead of reusing the cache: "
        f"{sorted(path.name for path in after_second - after_first)}"
    )


def test_returns_and_creates_the_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "cache"
    assert not target.exists()
    assert enable_compilation_cache(target) == target
    assert target.is_dir()


def test_default_directory_honours_the_environment(monkeypatch, tmp_path) -> None:
    """The cache follows FOLDJAX_HOME, like everything else FoldJAX writes.

    ``Path.home`` is pinned at a directory that has no pre-FoldJAX cache in it,
    because the fallback below is checked by existence: on a machine that still
    has ``~/.cache/openfold3_jax/jit`` these assertions would read the legacy
    path and the test would pass or fail on whose machine it ran.
    """
    from foldjax import paths

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nobody"))
    # Same reason: this checkout's `.foldjax/` outranks the two fallbacks.
    monkeypatch.setattr(paths, "_repository_store", lambda: None)
    monkeypatch.delenv("FOLDJAX_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    monkeypatch.setenv("OPENFOLD3_JAX_CACHE", "/tmp/explicit-openfold3-cache")
    assert default_cache_dir() == Path("/tmp/explicit-openfold3-cache")

    monkeypatch.delenv("OPENFOLD3_JAX_CACHE")
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    assert default_cache_dir() == tmp_path / "home" / "openfold3" / "jit"

    monkeypatch.delenv("FOLDJAX_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "foldjax" / "openfold3" / "jit"

    monkeypatch.delenv("XDG_CACHE_HOME")
    assert (
        default_cache_dir()
        == tmp_path / "nobody" / ".cache" / "foldjax" / "openfold3" / "jit"
    )


def test_a_pre_foldjax_cache_keeps_being_used(monkeypatch, tmp_path) -> None:
    """A populated standalone-era cache is preferred over a fresh empty one.

    These executables run to hundreds of megabytes and take minutes to build, so
    a checkout that already filled ``~/.cache/openfold3_jax/jit`` should keep
    hitting it rather than silently recompiling everything into the new
    location.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("OPENFOLD3_JAX_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))

    legacy = tmp_path / ".cache" / "openfold3_jax" / "jit"
    legacy.mkdir(parents=True)
    assert default_cache_dir() == legacy

    # Once the FoldJAX location exists it wins, so a migrated cache is not
    # shadowed forever by the directory it was migrated from.
    current = tmp_path / "home" / "openfold3" / "jit"
    current.mkdir(parents=True)
    assert default_cache_dir() == current


def test_it_is_not_enabled_merely_by_importing() -> None:
    """A library must not start writing to disk on import."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jax, foldjax.models.openfold3; "
            "print(jax.config.jax_compilation_cache_dir)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.stdout.strip() in {"None", ""}
