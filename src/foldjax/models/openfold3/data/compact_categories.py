"""Private compact storage for OpenFold3 atom categorical features.

OpenFold3's portable feature ABI carries element and atom-name categories as
dense ``int32`` one-hot arrays.  Managed prediction has already validated and
optionally padded those arrays, so it can retain the same information as small
``uint8`` IDs until the compiled graph reconstructs the historical inputs.
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
_ELEMENT_CLASSES = 119
_ATOM_NAME_CHAR_CLASSES = 64


def compact_ref_atom_category_storage(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compact exact generated atom categories, otherwise keep dense storage.

    The public dense pair is authoritative and removes stale private leaves.
    Compaction is deliberately narrower than accepting arbitrary one-hot data:
    it requires OpenFold3's validated batch/dtype contract and requires every
    live atom to have exactly one element and four character categories while
    every padded atom retains the historical all-zero vectors.
    """

    dense_element = features.get("ref_element")
    dense_chars = features.get("ref_atom_name_chars")
    if dense_element is not None or dense_chars is not None:
        out = dict(features)
        for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES:
            out.pop(name, None)
        atom_mask = features.get("atom_mask")
        if _can_compact_dense_categories(dense_element, dense_chars, atom_mask):
            element = np.asarray(dense_element)
            chars = np.asarray(dense_chars)
            mask = np.asarray(atom_mask).astype(bool, copy=False)
            element_ids = np.argmax(element, axis=-1).astype(np.uint8, copy=False)
            char_ids = np.argmax(chars, axis=-1).astype(np.uint8, copy=False)
            element_ids = element_ids.copy()
            char_ids = char_ids.copy()
            element_ids[~mask] = _ELEMENT_CLASSES
            char_ids[~mask, :] = _ATOM_NAME_CHAR_CLASSES
            del out["ref_element"]
            del out["ref_atom_name_chars"]
            out[COMPACT_REF_ATOM_CATEGORIES_MARKER] = np.asarray(
                _MARKER_VERSION, dtype=np.uint8
            )
            out[COMPACT_REF_ELEMENT_IDS] = element_ids
            out[COMPACT_REF_ATOM_NAME_CHAR_IDS] = char_ids
        return out

    if any(name in features for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES):
        validate_compact_ref_atom_categories(features)
    return features


def validate_compact_ref_atom_categories(features: Mapping[str, Any]) -> None:
    """Validate private compact provenance on concrete host arrays."""

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
            "OpenFold3 compact ref atom categories are incomplete; missing "
            + ", ".join(missing)
        )

    marker = np.asarray(features[COMPACT_REF_ATOM_CATEGORIES_MARKER])
    element_ids = np.asarray(features[COMPACT_REF_ELEMENT_IDS])
    char_ids = np.asarray(features[COMPACT_REF_ATOM_NAME_CHAR_IDS])
    if marker.shape != () or marker.dtype != np.dtype(np.uint8):
        raise ValueError(
            "OpenFold3 compact ref atom category marker must be scalar uint8"
        )
    if int(marker) != _MARKER_VERSION:
        raise ValueError("OpenFold3 compact ref atom category marker must be v1")
    if element_ids.dtype != np.dtype(np.uint8):
        raise ValueError("OpenFold3 compact ref element IDs must have dtype uint8")
    if char_ids.dtype != np.dtype(np.uint8):
        raise ValueError("OpenFold3 compact ref atom-name IDs must have dtype uint8")
    if (
        element_ids.ndim != 2
        or element_ids.shape[0] != 1
        or element_ids.shape[1] < 1
        or char_ids.shape != (*element_ids.shape, 4)
    ):
        raise ValueError(
            "OpenFold3 compact ref atom category IDs must have shapes (1, A) "
            "and (1, A, 4)"
        )
    if np.any(element_ids > _ELEMENT_CLASSES):
        raise ValueError("OpenFold3 compact ref element IDs exceed sentinel 119")
    if np.any(char_ids > _ATOM_NAME_CHAR_CLASSES):
        raise ValueError("OpenFold3 compact ref atom-name IDs exceed sentinel 64")

    if "atom_mask" not in features:
        raise KeyError("OpenFold3 compact ref atom categories need atom_mask")
    atom_mask = np.asarray(features["atom_mask"])
    if atom_mask.dtype != np.dtype(np.float32) or atom_mask.shape != element_ids.shape:
        raise ValueError(
            "OpenFold3 compact ref atom category atom_mask must have dtype "
            "float32 and shape (1, A)"
        )
    if not np.isin(atom_mask, (0.0, 1.0)).all():
        raise ValueError(
            "OpenFold3 compact ref atom category atom_mask must be binary"
        )
    live = atom_mask.astype(bool, copy=False)
    if not _has_nonempty_real_prefix(live):
        raise ValueError(
            "OpenFold3 compact ref atom category atom_mask must contain a "
            "non-empty real prefix"
        )
    if not np.array_equal(element_ids == _ELEMENT_CLASSES, ~live):
        raise ValueError(
            "OpenFold3 compact ref element sentinel 119 must identify exactly "
            "the padded atoms"
        )
    if not np.array_equal(
        char_ids == _ATOM_NAME_CHAR_CLASSES,
        np.broadcast_to((~live)[..., None], char_ids.shape),
    ):
        raise ValueError(
            "OpenFold3 compact ref atom-name sentinel 64 must identify exactly "
            "the padded atoms"
        )


def _can_compact_dense_categories(element: Any, chars: Any, atom_mask: Any) -> bool:
    if not all(isinstance(value, np.ndarray) for value in (element, chars, atom_mask)):
        return False
    if (
        element.dtype != np.dtype(np.int32)
        or chars.dtype != np.dtype(np.int32)
        or atom_mask.dtype != np.dtype(np.float32)
        or element.ndim != 3
        or element.shape[0] != 1
        or element.shape[-1] != _ELEMENT_CLASSES
        or element.shape[1] < 1
        or chars.shape != (*element.shape[:-1], 4, _ATOM_NAME_CHAR_CLASSES)
        or atom_mask.shape != element.shape[:-1]
    ):
        return False
    if not np.isin(atom_mask, (0.0, 1.0)).all():
        return False
    if not _binary_categories(element) or not _binary_categories(chars):
        return False
    live = atom_mask.astype(bool, copy=False)
    if not _has_nonempty_real_prefix(live):
        return False
    element_counts = np.sum(element, axis=-1, dtype=np.uint8)
    char_counts = np.sum(chars, axis=-1, dtype=np.uint8)
    return np.array_equal(element_counts, live) and np.array_equal(
        char_counts, np.broadcast_to(live[..., None], char_counts.shape)
    )


def _binary_categories(array: np.ndarray) -> bool:
    return not (
        np.any(np.min(array, axis=-1) < 0)
        or np.any(np.max(array, axis=-1) > 1)
    )


def _has_nonempty_real_prefix(live: np.ndarray) -> bool:
    count = int(np.count_nonzero(live))
    return count > 0 and np.array_equal(
        live, np.arange(live.shape[-1])[None, :] < count
    )


__all__ = [
    "COMPACT_REF_ATOM_CATEGORIES_MARKER",
    "COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES",
    "COMPACT_REF_ATOM_NAME_CHAR_IDS",
    "COMPACT_REF_ELEMENT_IDS",
    "compact_ref_atom_category_storage",
    "validate_compact_ref_atom_categories",
]
