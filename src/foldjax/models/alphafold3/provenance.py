"""Immutable identity of the AlphaFold 3 source carried by FoldJAX."""

from __future__ import annotations

# AlphaFold 3 derives its package version from Git.  The upstream fallback in
# ``version.py`` and ``pyproject.toml`` remains 3.0.2 even at the v3.0.4 tag, so
# a vendored source tree without ``.git`` cannot use either fallback as its
# release identity.
UPSTREAM_VERSION = "3.0.4"
UPSTREAM_COMMIT = "85c4d20505fd5cef05eac22b534d4e793971ae69"

# Every other carried Python/C++ source file is byte-identical to the commit
# above.  These bounded runtime patches keep generated data out of the wheel,
# avoid process-global cache leakage, and bound memory without changing the
# public model/checkpoint contract.  The checkout drift test makes additions to
# this set an explicit review decision.
UPSTREAM_PATCHES = frozenset(
    {
        "constants/chemical_components.py",
        "data/pipeline.py",
        "model/confidences.py",
        "model/features.py",
        "model/model.py",
        "model/network/diffusion_head.py",
        "structure/chemical_components.py",
        "structure/parsing.py",
    }
)

# Copies of the repository-level parameter/output terms live inside the Python
# package as well so a built wheel retains them next to the carried source.
PACKAGE_ONLY_FILES = frozenset(
    {
        "OUTPUT_TERMS_OF_USE.md",
        "WEIGHTS_PROHIBITED_USE_POLICY.md",
        "WEIGHTS_TERMS_OF_USE.md",
    }
)

__all__ = [
    "PACKAGE_ONLY_FILES",
    "UPSTREAM_COMMIT",
    "UPSTREAM_PATCHES",
    "UPSTREAM_VERSION",
]
