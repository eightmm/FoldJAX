"""Private compact storage for managed Protenix atom categories.

The public Protenix feature ABI carries element and atom-name categories as
dense float32 one-hot arrays.  Generated JSON features have a narrower private
contract: after validation and optional serving padding, exact one-hot-or-zero
rows may be retained as uint8 IDs until the consolidated graph reconstructs the
historical arrays immediately before their existing consumers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from foldjax.models._feature_storage import (
    COMPACT_REF_ATOM_CATEGORIES_MARKER,
    COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
    COMPACT_REF_ATOM_NAME_CHAR_SENTINEL,
    COMPACT_REF_ELEMENT_IDS,
    COMPACT_REF_ELEMENT_SENTINEL,
)
from foldjax.models._feature_storage import (
    compact_ref_atom_category_storage as _compact_ref_atom_category_storage,
)
from foldjax.models._feature_storage import (
    validate_compact_ref_atom_categories as _validate_compact_ref_atom_categories,
)
from foldjax.models._output_features import _EXPLICIT_OUTPUT_ATOM_FIELDS

_PORT_LABEL = "Protenix"


def compact_ref_atom_category_storage(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compact exact generated float32 categories, otherwise retain dense data.

    Dense public features are authoritative and discard stale private leaves.
    Compaction is deliberately stricter than the model ABI: only unbatched
    NumPy float32 arrays with exact unsigned 0/1, at most one set bit per row,
    and the released category widths are admitted.  This makes custom dtypes,
    non-finite values, negative zero, multi-hot values, and shape drift fall
    back without changing their representation.
    """

    return _compact_ref_atom_category_storage(
        features,
        dense_dtype=np.float32,
        allow_batch=False,
        port_label=_PORT_LABEL,
        require_non_empty=True,
    )


def validate_compact_ref_atom_categories(features: Mapping[str, Any]) -> None:
    """Validate the private v1 form on concrete host arrays before tracing."""

    _validate_compact_ref_atom_categories(
        features,
        allow_batch=False,
        port_label=_PORT_LABEL,
        require_non_empty=True,
    )


def drop_dense_categories_from_writer_snapshot(
    features: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Drop only dense categories when explicit writer metadata is complete.

    The Protenix writer prefers the five ``output_atom_*`` arrays and never
    decodes the dense category pair when all five are present.  Length checks
    keep the fallback conservative: static/custom or incomplete metadata keeps
    both dense arrays, preserving the writer's historical decoding path and
    error behaviour.
    """

    element = features.get("ref_element")
    chars = features.get("ref_atom_name_chars")
    if element is None or chars is None:
        return features
    element_shape = getattr(element, "shape", None)
    chars_shape = getattr(chars, "shape", None)
    if (
        element_shape is None
        or len(element_shape) != 2
        or element_shape[1] != COMPACT_REF_ELEMENT_SENTINEL
        or chars_shape != (element_shape[0], 4, COMPACT_REF_ATOM_NAME_CHAR_SENTINEL)
    ):
        return features
    n_atom = int(element_shape[0])
    if not all(
        name in features and getattr(features[name], "shape", None) == (n_atom,)
        for name in _EXPLICIT_OUTPUT_ATOM_FIELDS
    ):
        return features
    out = dict(features)
    del out["ref_element"]
    del out["ref_atom_name_chars"]
    return out


__all__ = [
    "COMPACT_REF_ATOM_CATEGORIES_MARKER",
    "COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES",
    "COMPACT_REF_ATOM_NAME_CHAR_IDS",
    "COMPACT_REF_ELEMENT_IDS",
    "compact_ref_atom_category_storage",
    "drop_dense_categories_from_writer_snapshot",
    "validate_compact_ref_atom_categories",
]
