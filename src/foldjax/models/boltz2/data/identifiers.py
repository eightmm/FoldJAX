"""Validation and path resolution for Boltz CCD molecule files.

The molecule bundle contains trusted pickles, but the CCD code selecting one of
those files can come from a user-authored job.  Keep that selector to the
canonical CCD filename alphabet and resolve the selected file inside the
configured molecule directory before any unpickling happens.
"""

from __future__ import annotations

import re
from pathlib import Path

_CCD_IDENTIFIER = re.compile(r"[A-Z0-9]{1,16}\Z", flags=re.ASCII)


def validate_ccd_identifier(value: object, *, field: str = "CCD identifier") -> str:
    """Return a canonical, path-safe CCD identifier.

    Released Boltz molecule filenames use uppercase ASCII letters and digits.
    Sixteen characters leaves room beyond the current one-to-five-character
    bundle while bounding filesystem work and rejecting every path spelling.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    identifier = value.strip()
    if _CCD_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(
            f"{field} must contain 1-16 uppercase ASCII letters or digits"
        )
    return identifier


def resolve_molecule_pickle(molecule_dir: str | Path, identifier: object) -> Path:
    """Resolve one trusted molecule pickle without crossing its root boundary."""
    code = validate_ccd_identifier(identifier)
    try:
        root = Path(molecule_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("configured molecule directory does not exist") from error
    if not root.is_dir():
        raise ValueError("configured molecule path is not a directory")

    try:
        candidate = (root / f"{code}.pkl").resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"CCD component {code} not found!") from error
    except (OSError, RuntimeError) as error:
        raise ValueError(f"CCD component {code} cannot be resolved safely") from error

    if not candidate.is_relative_to(root):
        raise ValueError(
            "CCD component file resolves outside the configured molecule directory"
        )
    if not candidate.is_file():
        raise ValueError(f"CCD component {code} is not a regular file")
    return candidate
