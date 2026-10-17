"""A probe subprocess must inherit the path that decides which checkout it imports.

The context-parallel gates run in a subprocess because a forced device count
has to be set before JAX initialises, and each one builds the child's
environment from scratch so nothing ambient can change what is measured.
Dropping `PYTHONPATH` along with everything else made those gates test
whatever `foldjax` is installed against rather than the tree under test:
inside a git worktree the probe imported a different checkout and printed its
success line with the code under test reverted.

Every from-scratch environment in the suite must therefore splat
`inherited_environment()`. This is checked in the source because the failure
it guards against is invisible at runtime -- the gate passes either way.
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

import foldjax


def _test_files() -> list[pathlib.Path]:
    root = pathlib.Path(foldjax.__file__).parents[2] / "tests"
    return sorted(root.rglob("*.py"))


def _env_dicts(tree: ast.AST) -> list[ast.Dict]:
    """Every `env={...}` literal passed to a call in this module."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "env" and isinstance(keyword.value, ast.Dict):
                found.append(keyword.value)
    return found


def _inherits(env: ast.Dict) -> bool:
    """True when the literal splats something, which is how inheritance enters."""
    return any(key is None for key in env.keys)


def test_every_subprocess_environment_inherits_the_import_path() -> None:
    offenders = []
    checked = 0
    for path in _test_files():
        if path.name == "test_probe_environment_contract.py":
            continue
        for env in _env_dicts(ast.parse(path.read_text())):
            checked += 1
            if not _inherits(env):
                offenders.append(f"{path.name}:{env.lineno}")
    assert not offenders, (
        "these build a subprocess environment from scratch without inheriting "
        f"PYTHONPATH, so they test whatever foldjax is installed against rather "
        f"than this tree: {offenders}"
    )
    assert checked >= 10, (
        f"only {checked} subprocess environments found -- has the suite moved "
        "on and left this gate testing nothing?"
    )


def test_the_helper_carries_the_import_path(monkeypatch) -> None:
    from tests.models.cp_probe_env import inherited_environment

    monkeypatch.setenv("PYTHONPATH", "/somewhere/under/test/src")
    assert inherited_environment()["PYTHONPATH"] == "/somewhere/under/test/src"

    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert "PYTHONPATH" not in inherited_environment()
    assert inherited_environment()["PATH"]


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="requires POSIX permissions that root does not bypass",
)
def test_numba_cache_survives_read_only_source_and_home(tmp_path, monkeypatch) -> None:
    from tests.models.cp_probe_env import inherited_environment

    source = tmp_path / "source"
    home = tmp_path / "home"
    cache = tmp_path / "numba"
    source.mkdir()
    home.mkdir()
    (source / "cached_function.py").write_text(
        "from numba import njit\n"
        "@njit(cache=True)\n"
        "def twice(value):\n"
        "    return 2 * value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache))
    source.chmod(0o555)
    home.chmod(0o555)
    try:
        environment = {
            **inherited_environment(),
            "HOME": str(home),
            "PYTHONPATH": str(source),
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from cached_function import twice; assert twice(3) == 6",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert list(cache.rglob("*.nbi")), "Numba did not persist in the explicit cache"
    finally:
        source.chmod(0o755)
        home.chmod(0o755)
