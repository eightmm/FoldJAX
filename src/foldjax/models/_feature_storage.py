"""Conservative host-storage compaction for model-bound feature trees."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

_MSA_ID_FIELDS = ("msa", "msa_loop_tape")
_BINARY_MSA_FIELDS = (
    "has_deletion",
    "msa_mask",
    "msa_paired",
    "msa_attention_mask",
    "has_deletion_loop_tape",
    "msa_attention_mask_loop_tape",
)

COMPACT_REF_ATOM_CATEGORIES_MARKER = "_foldjax_compact_ref_atom_categories"
COMPACT_REF_ELEMENT_IDS = "_foldjax_compact_ref_element_ids"
COMPACT_REF_ATOM_NAME_CHAR_IDS = "_foldjax_compact_ref_atom_name_char_ids"
COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES = (
    COMPACT_REF_ATOM_CATEGORIES_MARKER,
    COMPACT_REF_ELEMENT_IDS,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
)

COMPACT_REF_ATOM_MARKER_VERSION = 1
COMPACT_REF_ELEMENT_SENTINEL = 128
COMPACT_REF_ATOM_NAME_CHAR_SENTINEL = 64


def _exact_unsigned_binary(array: np.ndarray) -> bool:
    """Whether ``array`` is exactly representable as an unsigned boolean mask."""

    if array.dtype.hasobject or not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        return False
    try:
        binary = array.astype(bool)
        if not np.array_equal(array, binary):
            return False
        if np.issubdtype(array.dtype, np.floating):
            zero = array == 0
            if np.any(np.signbit(array[zero])):
                return False
    except (TypeError, ValueError):
        return False
    return True


def compact_msa_storage(
    features: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Narrow model-bound categorical MSA arrays without changing their values.

    Public featurizers retain their native dtypes.  Prediction entry points call
    this helper only after feature validation and keep the returned shallow copy
    as their private model input.  Unknown/custom layouts fall back by identity:

    * MSA id arrays must be non-empty NumPy integers representable as uint8;
    * present deletion/mask/paired fields must contain exact unsigned 0/1 values;
    * negative zero is rejected because converting it to ``bool`` loses a bit
      that can be observable when a custom caller multiplies by non-finite data.

    The model casts ``msa`` back to ``int32`` for indexing and binary fields to
    their historical compute dtype at the first consuming operation.  Thus this
    changes host/device argument storage, not model arithmetic.
    """

    categorical: dict[str, np.ndarray] = {}
    for name in _MSA_ID_FIELDS:
        if name not in features:
            continue
        value = features[name]
        if not isinstance(value, np.ndarray):
            return features
        if (
            value.size == 0
            or value.dtype.hasobject
            or not np.issubdtype(value.dtype, np.integer)
        ):
            return features
        try:
            if int(np.min(value)) < 0 or int(np.max(value)) > 255:
                return features
        except (TypeError, ValueError):
            return features
        categorical[name] = value.astype(np.uint8, copy=False)
    if not categorical:
        return features

    binary: dict[str, np.ndarray] = {}
    for name in _BINARY_MSA_FIELDS:
        if name not in features:
            continue
        value = features[name]
        if not isinstance(value, np.ndarray) or not _exact_unsigned_binary(value):
            return features
        binary[name] = value.astype(bool, copy=False)

    compact = dict(features)
    compact.update(categorical)
    compact.update(binary)
    return compact


def compact_ref_atom_category_storage(
    features: Mapping[str, Any],
    *,
    dense_dtype: np.dtype[Any] | type[np.generic],
    allow_batch: bool,
    port_label: str,
    require_non_empty: bool = False,
) -> Mapping[str, Any]:
    """Compact exact generated atom categories, otherwise keep dense storage.

    Dense public features are authoritative and discard stale private leaves.
    Only exact NumPy arrays matching ``dense_dtype``, the port's rank rules,
    and exact unsigned 0/1 one-hot-or-zero rows are compacted to uint8 IDs.
    """

    dense_element = features.get("ref_element")
    dense_chars = features.get("ref_atom_name_chars")
    if dense_element is not None or dense_chars is not None:
        out = dict(features)
        for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES:
            out.pop(name, None)
        if _can_compact_dense_categories(
            dense_element,
            dense_chars,
            dense_dtype=dense_dtype,
            allow_batch=allow_batch,
        ):
            element = np.asarray(dense_element)
            chars = np.asarray(dense_chars)
            element_ids = _one_hot_ids(element, COMPACT_REF_ELEMENT_SENTINEL)
            char_ids = _one_hot_ids(chars, COMPACT_REF_ATOM_NAME_CHAR_SENTINEL)
            del out["ref_element"]
            del out["ref_atom_name_chars"]
            out[COMPACT_REF_ATOM_CATEGORIES_MARKER] = np.asarray(
                COMPACT_REF_ATOM_MARKER_VERSION, dtype=np.uint8
            )
            out[COMPACT_REF_ELEMENT_IDS] = element_ids
            out[COMPACT_REF_ATOM_NAME_CHAR_IDS] = char_ids
        return out

    if any(name in features for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES):
        validate_compact_ref_atom_categories(
            features,
            allow_batch=allow_batch,
            port_label=port_label,
            require_non_empty=require_non_empty,
        )
    return features


def validate_compact_ref_atom_categories(
    features: Mapping[str, Any],
    *,
    allow_batch: bool,
    port_label: str,
    require_non_empty: bool = False,
) -> None:
    """Validate the private v1 compact category form on concrete host arrays."""

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
            f"{port_label} compact ref atom categories are incomplete; missing "
            + ", ".join(missing)
        )

    marker = np.asarray(features[COMPACT_REF_ATOM_CATEGORIES_MARKER])
    element_ids = np.asarray(features[COMPACT_REF_ELEMENT_IDS])
    char_ids = np.asarray(features[COMPACT_REF_ATOM_NAME_CHAR_IDS])
    if marker.shape != () or marker.dtype != np.dtype(np.uint8):
        raise ValueError(
            f"{port_label} compact ref atom category marker must be scalar uint8"
        )
    if int(marker) != COMPACT_REF_ATOM_MARKER_VERSION:
        raise ValueError(f"{port_label} compact ref atom category marker must be v1")
    if element_ids.dtype != np.dtype(np.uint8):
        raise ValueError(f"{port_label} compact ref element IDs must have dtype uint8")
    if char_ids.dtype != np.dtype(np.uint8):
        raise ValueError(
            f"{port_label} compact ref atom-name IDs must have dtype uint8"
        )
    if allow_batch:
        if element_ids.ndim < 1 or char_ids.shape != (*element_ids.shape, 4):
            raise ValueError(
                f"{port_label} compact ref atom category IDs must have shapes [..., A] "
                "and [..., A, 4]"
            )
    else:
        if element_ids.ndim != 1 or char_ids.shape != (element_ids.shape[0], 4):
            raise ValueError(
                f"{port_label} compact ref atom category IDs must have shapes [A] "
                "and [A, 4]"
            )
    if require_non_empty and element_ids.size == 0:
        raise ValueError(
            f"{port_label} compact ref atom category IDs must be non-empty"
        )
    if np.any(element_ids > COMPACT_REF_ELEMENT_SENTINEL):
        raise ValueError(
            f"{port_label} compact ref element IDs exceed sentinel "
            f"{COMPACT_REF_ELEMENT_SENTINEL}"
        )
    if np.any(char_ids > COMPACT_REF_ATOM_NAME_CHAR_SENTINEL):
        raise ValueError(
            f"{port_label} compact ref atom-name IDs exceed sentinel "
            f"{COMPACT_REF_ATOM_NAME_CHAR_SENTINEL}"
        )


def _can_compact_dense_categories(
    element: Any,
    chars: Any,
    *,
    dense_dtype: np.dtype[Any] | type[np.generic],
    allow_batch: bool,
) -> bool:
    expected_dtype = np.dtype(dense_dtype)
    if not isinstance(element, np.ndarray) or not isinstance(chars, np.ndarray):
        return False
    if element.dtype != expected_dtype or chars.dtype != expected_dtype:
        return False
    if allow_batch:
        if (
            element.ndim < 2
            or chars.ndim != element.ndim + 1
            or element.shape[-1] != COMPACT_REF_ELEMENT_SENTINEL
            or chars.shape
            != (*element.shape[:-1], 4, COMPACT_REF_ATOM_NAME_CHAR_SENTINEL)
            or element.size == 0
            or chars.size == 0
        ):
            return False
    else:
        if (
            element.ndim != 2
            or chars.shape != (element.shape[0], 4, COMPACT_REF_ATOM_NAME_CHAR_SENTINEL)
            or element.shape[1] != COMPACT_REF_ELEMENT_SENTINEL
            or element.shape[0] == 0
        ):
            return False
    return _exact_one_hot_or_zero(element) and _exact_one_hot_or_zero(chars)


def _exact_one_hot_or_zero(array: np.ndarray) -> bool:
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            return False
        if np.any(np.signbit(array[array == 0])):
            return False
        if not np.all((array == 0.0) | (array == 1.0)):
            return False
        return not np.any(np.sum(array, axis=-1, dtype=np.uint8) > 1)
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
    "COMPACT_REF_ATOM_MARKER_VERSION",
    "COMPACT_REF_ATOM_NAME_CHAR_IDS",
    "COMPACT_REF_ATOM_NAME_CHAR_SENTINEL",
    "COMPACT_REF_ELEMENT_IDS",
    "COMPACT_REF_ELEMENT_SENTINEL",
    "compact_msa_storage",
    "compact_ref_atom_category_storage",
    "validate_compact_ref_atom_categories",
]
