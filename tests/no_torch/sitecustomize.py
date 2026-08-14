"""Fail if production tries to import an external Torch/ESM tensor runtime.

Put this directory first on ``PYTHONPATH`` for real-weight CLI smoke tests.  A
meta-path blocker is stricter than checking ``sys.modules`` after the run: even
an attempted import that catches its own ImportError is visible as a failure.
"""

from __future__ import annotations

import importlib.abc
import sys

_BLOCKED_ROOTS = frozenset(
    {"esm", "lightning", "pytorch_lightning", "torch", "torchmetrics"}
)


class _NoExternalRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname.partition(".")[0] in _BLOCKED_ROOTS:
            # ImportError is deliberately not used: optional-dependency probes
            # commonly catch it, which would hide the attempted import this
            # release gate exists to detect.
            raise AssertionError(
                f"JAX-only verification blocked an import of {fullname!r}"
            )
        return None


sys.meta_path.insert(0, _NoExternalRuntimeImports())
