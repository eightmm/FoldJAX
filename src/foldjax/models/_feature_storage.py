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


__all__ = ["compact_msa_storage"]
