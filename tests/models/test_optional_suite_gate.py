"""The optional-extra collection gate must stay honest.

``conftest.collect_ignore`` names vendored modules by path. A stale entry would
silently stop gating anything while still reading as covered, and a missing one
turns into a collection error the moment the extra is absent.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

from .conftest import _OPTIONAL_SUITES, collect_ignore

_HERE = Path(__file__).parent


def test_every_gated_module_exists() -> None:
    missing = [
        path
        for _extra, paths in _OPTIONAL_SUITES.values()
        for path in paths
        if not (_HERE / path).is_file()
    ]
    assert missing == []


def test_no_gate_entry_is_listed_twice() -> None:
    paths = [path for _extra, group in _OPTIONAL_SUITES.values() for path in group]
    assert len(paths) == len(set(paths))


def test_ignored_set_matches_the_dependencies_that_are_actually_missing() -> None:
    expected = [
        path
        for module, (_extra, paths) in _OPTIONAL_SUITES.items()
        if find_spec(module) is None
        for path in paths
    ]
    assert sorted(collect_ignore) == sorted(expected)


def test_this_module_is_collectable_without_any_optional_extra() -> None:
    """If the gate itself needed an extra, nothing above would ever run."""
    assert __name__.endswith("test_optional_suite_gate")
