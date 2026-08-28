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

COMPACT_REF_ATOM_CATEGORIES_MARKER = "_foldjax_compact_ref_atom_categories"
COMPACT_REF_ELEMENT_IDS = "_foldjax_compact_ref_element_ids"
COMPACT_REF_ATOM_NAME_CHAR_IDS = "_foldjax_compact_ref_atom_name_char_ids"
COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES = (
    COMPACT_REF_ATOM_CATEGORIES_MARKER,
    COMPACT_REF_ELEMENT_IDS,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
)

_MARKER_VERSION = 1
_ELEMENT_SENTINEL = 128
_ATOM_NAME_CHAR_SENTINEL = 64


def compact_ref_atom_category_storage(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compact exact generated atom categories, otherwise keep dense storage.

    The public dense pair wins over every stale private leaf and is the only
    input considered for new compaction.  Unknown/custom values deliberately
    retain their identity: a compact representation is valid only for int64
    0/1 one-hot-or-zero arrays with OpenDDE's released category dimensions.
    """

    dense_element = features.get("ref_element")
    dense_chars = features.get("ref_atom_name_chars")
    if dense_element is not None or dense_chars is not None:
        out = dict(features)
        for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES:
            out.pop(name, None)
        if _can_compact_dense_categories(dense_element, dense_chars):
            element = np.asarray(dense_element)
            chars = np.asarray(dense_chars)
            element_ids = _one_hot_ids(element, _ELEMENT_SENTINEL)
            char_ids = _one_hot_ids(chars, _ATOM_NAME_CHAR_SENTINEL)
            del out["ref_element"]
            del out["ref_atom_name_chars"]
            out[COMPACT_REF_ATOM_CATEGORIES_MARKER] = np.asarray(
                _MARKER_VERSION, dtype=np.uint8
            )
            out[COMPACT_REF_ELEMENT_IDS] = element_ids
            out[COMPACT_REF_ATOM_NAME_CHAR_IDS] = char_ids
        return out

    # A managed caller should never supply private storage itself.  Refuse a
    # marker-bearing partial/malformed payload rather than silently treating it
    # as a custom dense fallback, which could make a cache hit depend on stale
    # provenance.
    if any(name in features for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES):
        validate_compact_ref_atom_categories(features)
    return features


def validate_compact_ref_atom_categories(features: Mapping[str, Any]) -> None:
    """Validate private compact provenance on concrete host arrays.

    Dense public features always take precedence and intentionally make stale
    private leaves irrelevant.  Without the dense pair the marker and both ID
    arrays are an all-or-nothing ABI, checked before JIT turns values into
    tracers.
    """

    if "ref_element" in features or "ref_atom_name_chars" in features:
        return
    present = {
        name: name in features for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES
    }
    if not any(present.values()):
        return
    missing = [name for name, exists in present.items() if not exists]
    if missing:
        raise KeyError(
            "OpenDDE compact ref atom categories are incomplete; missing "
            + ", ".join(missing)
        )
    marker = np.asarray(features[COMPACT_REF_ATOM_CATEGORIES_MARKER])
    element_ids = np.asarray(features[COMPACT_REF_ELEMENT_IDS])
    char_ids = np.asarray(features[COMPACT_REF_ATOM_NAME_CHAR_IDS])
    if marker.shape != () or marker.dtype != np.dtype(np.uint8):
        raise ValueError(
            "OpenDDE compact ref atom category marker must be scalar uint8"
        )
    if int(marker) != _MARKER_VERSION:
        raise ValueError("OpenDDE compact ref atom category marker must be v1")
    if element_ids.dtype != np.dtype(np.uint8):
        raise ValueError("OpenDDE compact ref element IDs must have dtype uint8")
    if char_ids.dtype != np.dtype(np.uint8):
        raise ValueError("OpenDDE compact ref atom-name IDs must have dtype uint8")
    if element_ids.ndim < 1 or char_ids.shape != (*element_ids.shape, 4):
        raise ValueError(
            "OpenDDE compact ref atom category IDs must have shapes [..., A] "
            "and [..., A, 4]"
        )
    if np.any(element_ids > _ELEMENT_SENTINEL):
        raise ValueError("OpenDDE compact ref element IDs exceed sentinel 128")
    if np.any(char_ids > _ATOM_NAME_CHAR_SENTINEL):
        raise ValueError("OpenDDE compact ref atom-name IDs exceed sentinel 64")


def _can_compact_dense_categories(element: Any, chars: Any) -> bool:
    if not isinstance(element, np.ndarray) or not isinstance(chars, np.ndarray):
        return False
    if (
        element.dtype != np.dtype(np.int64)
        or chars.dtype != np.dtype(np.int64)
        or element.ndim < 2
        or chars.ndim != element.ndim + 1
        or element.shape[-1] != _ELEMENT_SENTINEL
        or chars.shape != (*element.shape[:-1], 4, _ATOM_NAME_CHAR_SENTINEL)
        or element.size == 0
        or chars.size == 0
    ):
        return False
    return _exact_one_hot_or_zero(element) and _exact_one_hot_or_zero(chars)


def _exact_one_hot_or_zero(array: np.ndarray) -> bool:
    # int64 was established above, so min/max and the uint8 count cannot hide
    # fractional, non-finite, signed-zero, or multi-hot custom values.
    if np.any(np.min(array, axis=-1) < 0) or np.any(np.max(array, axis=-1) > 1):
        return False
    return not np.any(np.sum(array, axis=-1, dtype=np.uint8) > 1)


def _one_hot_ids(array: np.ndarray, sentinel: int) -> np.ndarray:
    counts = np.sum(array, axis=-1, dtype=np.uint8)
    ids = np.argmax(array, axis=-1).astype(np.uint8, copy=False)
    ids = ids.copy()
    ids[counts == 0] = sentinel
    return ids


__all__ = [
    "COMPACT_REF_ATOM_CATEGORIES_MARKER",
    "COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES",
    "COMPACT_REF_ATOM_NAME_CHAR_IDS",
    "COMPACT_REF_ELEMENT_IDS",
    "compact_ref_atom_category_storage",
    "validate_compact_ref_atom_categories",
]
