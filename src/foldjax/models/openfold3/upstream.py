"""Locate an upstream OpenFold3 checkout, for the torch-parity gates only.

**Nothing FoldJAX ships needs this.** Featurization used to: it delegated to
upstream's data pipeline through `ensure_importable()`, which made a sibling
checkout a requirement for anyone building features. That pipeline is vendored at
`.._upstream` now, so both prediction and featurization run with nothing outside
this repository.

What is left is the parity suite, which compares this port against upstream's own
*torch modules* -- `OpenFold3.forward`, `run_trunk`, `DiffusionModule` -- and
those are the thing being ported, so carrying a copy of them here would mean
diffing the port against a copy of its own source. The other three ports make the
same trade: their `*_checkpoint_parity` tests need `../boltz`, `../protenix` and
`../OpenDDE`, and skip without them.

So: a missing checkout skips a development gate. It does not affect using the
package.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

#: Set this to a checkout to override the search.
SOURCE_ENV_VAR = "OPENFOLD3_SOURCE"


def _is_checkout(path: Path) -> bool:
    """Whether ``path`` is an OpenFold3 checkout rather than a same-named directory.

    The model package is the marker: ``openfold3_weights`` and a stray output
    directory would both pass a bare ``is_dir()`` check.
    """
    return (path / "openfold3" / "core" / "model").is_dir()


def candidate_roots() -> tuple[Path, ...]:
    """Places a sibling checkout could be, most specific first."""
    candidates: list[Path] = []
    explicit = os.environ.get(SOURCE_ENV_VAR)
    if explicit:
        candidates.append(Path(explicit).expanduser())
    # ``.../<workspace>/foldjax/src/foldjax/models/openfold3/upstream.py``: the
    # repository root is five levels above this file, and the workspace holding
    # both checkouts is its parent. The depth changed when this port was
    # vendored -- it used to be a top-level package two levels down.
    repository_root = Path(__file__).resolve().parents[4]
    candidates.append(repository_root.parent / "openfold3")
    cwd = Path.cwd()
    candidates.extend([cwd / "openfold3", cwd.parent / "openfold3"])
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def find_source() -> Path | None:
    """The first candidate that is really a checkout, or ``None``."""
    for candidate in candidate_roots():
        if _is_checkout(candidate):
            return candidate
    return None


def ensure_importable() -> Path | None:
    """Make ``import openfold3`` work, and report where it was found.

    Returns the checkout that was added to ``sys.path``, or ``None`` when
    ``openfold3`` is already importable -- an installed copy is left alone rather
    than shadowed by a checkout that may be a different revision.

    Raises:
        ImportError: no checkout was found, with the paths that were tried, since
            "No module named 'openfold3'" does not say that a *sibling directory*
            is what is missing.
    """
    if importlib.util.find_spec("openfold3") is not None:
        return None
    source = find_source()
    if source is None:
        tried = "\n  ".join(str(path) for path in candidate_roots())
        raise ImportError(
            "no OpenFold3 checkout found. Featurization and the parity gate need "
            "upstream's own code, which is not on PyPI, so it has to be a checkout. "
            f"Set ${SOURCE_ENV_VAR} to one. Tried:\n  {tried}"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source
