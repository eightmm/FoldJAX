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
_EXPLICIT_OUTPUT_ATOM_FIELDS = (
    "output_atom_name",
    "output_atom_element",
    "output_atom_res_name",
    "output_atom_chain_id",
    "output_atom_res_id",
)


def compact_ref_atom_category_storage(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compact exact generated float32 categories, otherwise retain dense data.

    Dense public features are authoritative and discard stale private leaves.
    Compaction is deliberately stricter than the model ABI: only unbatched
    NumPy float32 arrays with exact unsigned 0/1, at most one set bit per row,
    and the released category widths are admitted.  This makes custom dtypes,
    non-finite values, negative zero, multi-hot values, and shape drift fall
    back without changing their representation.
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
            del out["ref_element"]
            del out["ref_atom_name_chars"]
            out[COMPACT_REF_ATOM_CATEGORIES_MARKER] = np.asarray(
                _MARKER_VERSION, dtype=np.uint8
            )
            out[COMPACT_REF_ELEMENT_IDS] = _one_hot_ids(element, _ELEMENT_SENTINEL)
            out[COMPACT_REF_ATOM_NAME_CHAR_IDS] = _one_hot_ids(
                chars, _ATOM_NAME_CHAR_SENTINEL
            )
        return out

    if any(name in features for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES):
        validate_compact_ref_atom_categories(features)
    return features


def validate_compact_ref_atom_categories(features: Mapping[str, Any]) -> None:
    """Validate the private v1 form on concrete host arrays before tracing."""

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
            "Protenix compact ref atom categories are incomplete; missing "
            + ", ".join(missing)
        )

    marker = np.asarray(features[COMPACT_REF_ATOM_CATEGORIES_MARKER])
    element_ids = np.asarray(features[COMPACT_REF_ELEMENT_IDS])
    char_ids = np.asarray(features[COMPACT_REF_ATOM_NAME_CHAR_IDS])
    if marker.shape != () or marker.dtype != np.dtype(np.uint8):
        raise ValueError(
            "Protenix compact ref atom category marker must be scalar uint8"
        )
    if int(marker) != _MARKER_VERSION:
        raise ValueError("Protenix compact ref atom category marker must be v1")
    if element_ids.dtype != np.dtype(np.uint8):
        raise ValueError("Protenix compact ref element IDs must have dtype uint8")
    if char_ids.dtype != np.dtype(np.uint8):
        raise ValueError("Protenix compact ref atom-name IDs must have dtype uint8")
    if element_ids.ndim != 1 or char_ids.shape != (element_ids.shape[0], 4):
        raise ValueError(
            "Protenix compact ref atom category IDs must have shapes [A] and [A, 4]"
        )
    if element_ids.size == 0:
        raise ValueError("Protenix compact ref atom category IDs must be non-empty")
    if np.any(element_ids > _ELEMENT_SENTINEL):
        raise ValueError("Protenix compact ref element IDs exceed sentinel 128")
    if np.any(char_ids > _ATOM_NAME_CHAR_SENTINEL):
        raise ValueError("Protenix compact ref atom-name IDs exceed sentinel 64")


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
        or element_shape[1] != _ELEMENT_SENTINEL
        or chars_shape != (element_shape[0], 4, _ATOM_NAME_CHAR_SENTINEL)
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


def _can_compact_dense_categories(element: Any, chars: Any) -> bool:
    if not isinstance(element, np.ndarray) or not isinstance(chars, np.ndarray):
        return False
    if (
        element.dtype != np.dtype(np.float32)
        or chars.dtype != np.dtype(np.float32)
        or element.ndim != 2
        or chars.shape != (element.shape[0], 4, _ATOM_NAME_CHAR_SENTINEL)
        or element.shape[1] != _ELEMENT_SENTINEL
        or element.shape[0] == 0
    ):
        return False
    return _exact_one_hot_or_zero(element) and _exact_one_hot_or_zero(chars)


def _exact_one_hot_or_zero(array: np.ndarray) -> bool:
    if not np.all(np.isfinite(array)):
        return False
    if np.any(np.signbit(array[array == 0])):
        return False
    if not np.all((array == 0.0) | (array == 1.0)):
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
    "drop_dense_categories_from_writer_snapshot",
    "validate_compact_ref_atom_categories",
]
