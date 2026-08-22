"""What a context-parallel probe subprocess must inherit, and why.

These probes run in a subprocess because a forced device count has to be set
before JAX initialises, and each call site builds the child's environment from
scratch so that nothing lying around the parent's environment can change what
is measured. That is the right instinct for `JAX_PLATFORMS` and `XLA_FLAGS`
and the wrong one for `PYTHONPATH`: `foldjax` is installed editable, so a
child without it imports whatever the install points at rather than the tree
under test.

Inside a git worktree those are different checkouts. The gate then reports on
code nobody edited -- verified by breaking `_is_movable_array` in a worktree
and watching the probe print its success line anyway.
"""

from __future__ import annotations

import os

#: Absent, these either send the child at a different checkout or stop it
#: running at all. Everything else stays out, which is the point of the
#: from-scratch environment.
_INHERITED = ("PATH", "PYTHONPATH")


def inherited_environment() -> dict[str, str]:
    """The parent entries a probe child cannot do without."""

    environment = {name: os.environ[name] for name in _INHERITED if name in os.environ}
    environment.setdefault("PATH", "/usr/bin")
    return environment
