"""Where FoldJAX keeps downloads, converted weights, and compile caches.

One root so a machine has a single place to look, size, and clear. Override it
with ``FOLDJAX_HOME``; otherwise it follows ``XDG_CACHE_HOME`` and falls back to
``~/.cache/foldjax``.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV = "FOLDJAX_HOME"


def foldjax_home() -> Path:
    """Return the FoldJAX root directory. Not created as a side effect."""
    override = os.environ.get(_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "foldjax"


def downloads_dir(model: str | None = None) -> Path:
    """Released files exactly as published, kept so a re-convert needs no network."""
    root = foldjax_home() / "downloads"
    return root if model is None else root / model


def weights_dir(model: str | None = None) -> Path:
    """Converted, torch-free JAX weights that prediction actually loads."""
    root = foldjax_home() / "weights"
    return root if model is None else root / model


def assets_dir() -> Path:
    """Shared chemistry assets (CCD dictionaries, RDKit caches, template maps)."""
    return foldjax_home() / "assets"


def compile_cache_dir() -> Path:
    """Default XLA persistent-compilation cache root.

    ``api.predict`` narrows this per model, weight identity, and compile-
    relevant options, so one root is safe to share across every backend.
    """
    return foldjax_home() / "compile"


def describe() -> dict[str, str]:
    """Human-readable map of the locations, for `foldjax home`."""
    return {
        "home": str(foldjax_home()),
        "downloads": str(downloads_dir()),
        "weights": str(weights_dir()),
        "assets": str(assets_dir()),
        "compile_cache": str(compile_cache_dir()),
    }
