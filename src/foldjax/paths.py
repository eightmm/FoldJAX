"""Where FoldJAX keeps downloads, prediction assets, and compile caches.

One root so a machine has a single place to look, size, and clear. Override it
with ``FOLDJAX_HOME``; a source checkout that has a ``.foldjax/`` directory keeps
its store there; otherwise it follows ``XDG_CACHE_HOME`` and falls back to
``~/.cache/foldjax``.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV = "FOLDJAX_HOME"

#: A store inside the working tree. `.gitignore` already covers this name.
_LOCAL_DIR = ".foldjax"


def _repository_store() -> Path | None:
    """A ``.foldjax/`` store in the checkout this module was imported from.

    Only when it already exists: creating it is the user's decision, made by
    making the directory. This file is ``<repository>/src/foldjax/paths.py``, so
    the root is three levels up; an installed wheel has no such directory next to
    it and falls through to the cache.
    """
    candidate = Path(__file__).resolve().parents[2] / _LOCAL_DIR
    return candidate if candidate.is_dir() else None


def foldjax_home() -> Path:
    """Return the FoldJAX root directory. Not created as a side effect."""
    override = os.environ.get(_ENV)
    if override:
        return Path(override).expanduser()
    local = _repository_store()
    if local is not None:
        return local
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "foldjax"


def downloads_dir(model: str | None = None) -> Path:
    """Released files exactly as published, kept so a re-convert needs no network."""
    root = foldjax_home() / "downloads"
    return root if model is None else root / model


def weights_dir(model: str | None = None) -> Path:
    """Prediction-ready model checkpoints and asset directories."""
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


def msa_cache_dir() -> Path:
    """Searched alignments, addressed by sequence and search provenance.

    Deliberately not per model: the alignment for a sequence is the same
    alignment whichever model consumes it, so running one target through three
    backends searches once.
    """
    return foldjax_home() / "msa"


def runtime_dir(model: str | None = None) -> Path:
    """Generated native artifacts that cannot live in a read-only wheel."""
    root = foldjax_home() / "runtime"
    return root if model is None else root / model


def describe() -> dict[str, str]:
    """Human-readable map of the locations, for `foldjax home`."""
    return {
        "home": str(foldjax_home()),
        "downloads": str(downloads_dir()),
        "weights": str(weights_dir()),
        "assets": str(assets_dir()),
        "compile_cache": str(compile_cache_dir()),
        "msa": str(msa_cache_dir()),
        "runtime": str(runtime_dir()),
    }
