"""A backend that writes structures itself must skip the writer in trunk mode.

`stop_after="trunk"` compiles a graph that returns before the sampler, so the
prediction it hands back has no coordinates. Two backends write structures
themselves rather than delegating to their model's CLI, and both reached the
writer anyway: OpenFold3 had no branch at all (`IndexError` on an empty
coordinate shape) and ESMFold2 had one placed *below* the writer, which is the
same as not having it (`KeyError: 'sample_atom_coords'`). Neither is reachable
by a test that cannot load weights, so the order is checked in the source.

Backends that delegate to a model CLI do not call the writer here and are
skipped by construction -- the branch lives in that CLI instead.
"""

import ast
import inspect
import pathlib

import foldjax.backends


def _backend_sources() -> list[pathlib.Path]:
    directory = pathlib.Path(inspect.getfile(foldjax.backends)).parent
    return sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))


def _writes_structures(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_prediction_outputs"
    ]


def _trunk_early_returns(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(inner, ast.Attribute) and inner.attr == "stop_after"
            for inner in ast.walk(node.test)
        )
        and any(isinstance(inner, ast.Return) for inner in node.body)
    ]


def test_the_writer_is_below_the_trunk_branch_in_every_backend() -> None:
    checked = []
    for path in _backend_sources():
        tree = ast.parse(path.read_text())
        writers = _writes_structures(tree)
        if not writers:
            continue
        checked.append(path.name)
        returns = _trunk_early_returns(tree)
        assert returns, (
            f"{path.name} writes structures but never returns early on "
            "stop_after='trunk'; a trunk-only prediction has no coordinates"
        )
        assert min(returns) < min(writers), (
            f"{path.name} reaches write_prediction_outputs at line "
            f"{min(writers)} before its stop_after branch at line "
            f"{min(returns)}; a branch below the writer never runs"
        )
    assert checked, "no backend writes structures directly -- has the test gone stale?"
