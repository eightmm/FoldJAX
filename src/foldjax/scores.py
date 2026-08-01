"""Shared confidence-score normalization.

AlphaFold 3 and Protenix both write a per-sample summary JSON beside each
structure. Their key names differ and are left untranslated, because a ``ptm``
from one model is not the same quantity as a ``ptm`` from another; only the
scalar/array split is common.
"""

from __future__ import annotations

import json
from pathlib import Path


def scalar_scores(path: Path) -> dict[str, float]:
    """Return the scalar fields of a summary confidence JSON, or ``{}``.

    A missing or non-object file yields no scores rather than an error: the
    structure is the primary result and confidence output is optional in several
    of the native runners.
    """
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
