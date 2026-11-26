"""Private compact storage for OpenDDE atom categorical features.

OpenDDE's public feature ABI uses dense, int64 one-hot arrays.  A managed
prediction is the one place where those arrays are known to have come from the
trusted featurizer, so it may retain the same information as small uint8 IDs
until the compiled model needs the historical dense inputs again.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from foldjax.models._feature_storage import (
    COMPACT_REF_ATOM_CATEGORIES_MARKER,
    COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
    COMPACT_REF_ELEMENT_IDS,
)
from foldjax.models._feature_storage import (
    compact_ref_atom_category_storage as _compact_ref_atom_category_storage,
)
from foldjax.models._feature_storage import (
    validate_compact_ref_atom_categories as _validate_compact_ref_atom_categories,
)

_PORT_LABEL = "OpenDDE"


def compact_ref_atom_category_storage(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compact exact generated atom categories, otherwise keep dense storage.

    The public dense pair wins over every stale private leaf and is the only
    input considered for new compaction.  Unknown/custom values deliberately
    retain their identity: a compact representation is valid only for int64
    0/1 one-hot-or-zero arrays with OpenDDE's released category dimensions.
    """

    return _compact_ref_atom_category_storage(
        features,
        dense_dtype=np.int64,
        allow_batch=True,
        port_label=_PORT_LABEL,
        require_non_empty=False,
    )


def validate_compact_ref_atom_categories(features: Mapping[str, Any]) -> None:
    """Validate private compact provenance on concrete host arrays.

    Dense public features always take precedence and intentionally make stale
    private leaves irrelevant.  Without the dense pair the marker and both ID
    arrays are an all-or-nothing ABI, checked before JIT turns values into
    tracers.
    """

    _validate_compact_ref_atom_categories(
        features,
        allow_batch=True,
        port_label=_PORT_LABEL,
        require_non_empty=False,
    )


__all__ = [
    "COMPACT_REF_ATOM_CATEGORIES_MARKER",
    "COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES",
    "COMPACT_REF_ATOM_NAME_CHAR_IDS",
    "COMPACT_REF_ELEMENT_IDS",
    "compact_ref_atom_category_storage",
    "validate_compact_ref_atom_categories",
]
