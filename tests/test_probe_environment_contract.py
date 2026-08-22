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
import pathlib

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
