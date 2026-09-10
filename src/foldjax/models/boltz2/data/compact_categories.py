"""Private compact storage for Boltz-2 categorical pair and atom features.

Boltz-2's publisher featurizer hands the model five categorical arrays in
their widest form: ``contact_conditioning``, ``ref_element`` and
``ref_atom_name_chars`` are ``int64`` one-hots, ``type_bonds`` is an ``int64``
table index and ``token_bonds`` is a ``float32`` 0/1 flag.  Managed prediction
has already validated them, so it carries the same information as ``uint8``
IDs and a ``bool`` flag until the compiled graph rebuilds the historical
arrays (:mod:`foldjax.models.boltz2.models._compact_categories`).

Public featurizer output, direct/custom calls and eager steering keep the
dense arrays; only the managed prediction path reaches this module.  The
contract is the one :mod:`foldjax.models.boltz2.data.ownership` set: dense
input is authoritative and removes stale private leaves, an unfamiliar dense
layout falls back by identity, and an incomplete or wrong-version private-only
representation fails explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

COMPACT_CONTACT_CONDITIONING: Final = "_foldjax_compact_contact_conditioning"
CONTACT_CONDITIONING_IDS: Final = "_foldjax_contact_conditioning_ids"
COMPACT_TOKEN_BONDS: Final = "_foldjax_compact_token_bonds"
TOKEN_BONDS_FLAGS: Final = "_foldjax_token_bonds_flags"
TYPE_BONDS_IDS: Final = "_foldjax_type_bonds_ids"
COMPACT_REF_ATOM_CATEGORIES: Final = "_foldjax_compact_ref_atom_categories"
REF_ELEMENT_IDS: Final = "_foldjax_ref_element_ids"
REF_ATOM_NAME_CHAR_IDS: Final = "_foldjax_ref_atom_name_char_ids"

#: Class counts of the publisher one-hots.  Each doubles as the sentinel ID
#: for the all-zero rows that token/atom padding produces, because
#: ``jax.nn.one_hot`` returns a zero row for an out-of-range index.
CONTACT_CONDITIONING_CLASSES: Final = 5
REF_ELEMENT_CLASSES: Final = 128
REF_ATOM_NAME_CHAR_CLASSES: Final = 64

_COMPACT_VERSION: Final = 1

CONTACT_PRIVATE_FEATURES: Final = (
    COMPACT_CONTACT_CONDITIONING,
    CONTACT_CONDITIONING_IDS,
)
TOKEN_BOND_PRIVATE_FEATURES: Final = (
    COMPACT_TOKEN_BONDS,
    TOKEN_BONDS_FLAGS,
    TYPE_BONDS_IDS,
)
REF_ATOM_PRIVATE_FEATURES: Final = (
    COMPACT_REF_ATOM_CATEGORIES,
    REF_ELEMENT_IDS,
    REF_ATOM_NAME_CHAR_IDS,
)
PRIVATE_CATEGORY_FEATURES: Final = (
    *CONTACT_PRIVATE_FEATURES,
    *TOKEN_BOND_PRIVATE_FEATURES,
    *REF_ATOM_PRIVATE_FEATURES,
)

# Bound the largest temporary made while proving a publisher array exact.  The
# pair one-hot is hundreds of MiB at production bucket sizes, so a whole-tensor
# comparison would briefly duplicate a material fraction of it.
_VALIDATION_CHUNK_ELEMENTS: Final = 8 * 1024 * 1024


def _chunk_rows(trailing: int) -> int:
    return max(1, _VALIDATION_CHUNK_ELEMENTS // max(1, trailing))


def _category_ids(dense: Any, *, classes: int, ndim: int) -> np.ndarray | None:
    """Return ``uint8`` IDs for an exact ``int64`` one-hot-or-zero array.

    ``None`` rejects every layout the graph cannot rebuild bit-for-bit: a
    non-``int64`` publisher array, a wrong rank or class count, a value other
    than 0/1, or a row with more than one hit.  An all-zero row -- what token
    and atom padding leaves behind -- becomes the ``classes`` sentinel.
    """

    if not isinstance(dense, np.ndarray) or dense.ndim != ndim:
        return None
    if dense.dtype != np.dtype(np.int64):
        return None
    if dense.shape[-1] != classes or dense.size == 0:
        return None
    if classes > np.iinfo(np.uint8).max:  # pragma: no cover - constant guard
        return None

    ids = np.empty(dense.shape[:-1], dtype=np.uint8)
    trailing = int(np.prod(dense.shape[2:], dtype=np.int64))
    rows = dense.shape[1]
    for start in range(0, rows, _chunk_rows(trailing)):
        stop = min(rows, start + _chunk_rows(trailing))
        block = dense[:, start:stop]
        if not np.all((block == 0) | (block == 1)):
            return None
        counts = np.count_nonzero(block, axis=-1)
        if np.any(counts > 1):
            return None
        selected = np.argmax(block, axis=-1).astype(np.uint8, copy=False)
        ids[:, start:stop] = np.where(counts == 1, selected, np.uint8(classes))
    return ids


def _binary_flags(dense: Any, *, ndim: int) -> np.ndarray | None:
    """Return a ``bool`` copy of an exact ``float32`` 0/1 array, else ``None``.

    Negative zero is rejected with the other inexact values: converting it to
    ``bool`` loses a bit a custom caller can observe by multiplying it with
    non-finite data.
    """

    if not isinstance(dense, np.ndarray) or dense.ndim != ndim:
        return None
    if dense.dtype != np.dtype(np.float32) or dense.size == 0:
        return None

    flags = np.empty(dense.shape, dtype=bool)
    trailing = int(np.prod(dense.shape[2:], dtype=np.int64))
    rows = dense.shape[1]
    for start in range(0, rows, _chunk_rows(trailing)):
        stop = min(rows, start + _chunk_rows(trailing))
        block = dense[:, start:stop]
        zeros = block == 0
        if not np.all(zeros | (block == 1)):
            return None
        if np.any(np.signbit(block) & zeros):
            return None
        flags[:, start:stop] = block != 0
    return flags


def _table_ids(dense: Any, *, ndim: int) -> np.ndarray | None:
    """Return ``uint8`` IDs for an ``int64`` table index that fits, else ``None``."""

    if not isinstance(dense, np.ndarray) or dense.ndim != ndim:
        return None
    if dense.dtype != np.dtype(np.int64) or dense.size == 0:
        return None
    if int(np.min(dense)) < 0 or int(np.max(dense)) > np.iinfo(np.uint8).max:
        return None
    return dense.astype(np.uint8, copy=False)


def _without(features: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    clean = dict(features)
    for name in names:
        clean.pop(name, None)
    return clean


def _marker(features: Mapping[str, Any], name: str) -> None:
    value = features[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.shape != () or value.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} must be a scalar uint8")
    if int(value.item()) != _COMPACT_VERSION:
        raise ValueError(f"{name} must contain version {_COMPACT_VERSION}")


def _require_group(features: Mapping[str, Any], names: tuple[str, ...]) -> bool:
    """Whether a complete private group is present; incomplete ones raise."""

    present = [name for name in names if name in features]
    if not present:
        return False
    if len(present) != len(names):
        missing = [name for name in names if name not in features]
        raise ValueError(
            "compact Boltz-2 categories require every member of their private "
            "group; missing " + ", ".join(missing)
        )
    return True


def _ids_payload(
    features: Mapping[str, Any],
    name: str,
    *,
    dtype: np.dtype,
    ndim: int,
    sentinel: int | None,
) -> np.ndarray:
    value = features[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if sentinel is not None and value.size and int(np.max(value)) > sentinel:
        raise ValueError(f"{name} entries must be at most the sentinel {sentinel}")
    return value


def _pair_shape(features: Mapping[str, Any]) -> tuple[int, ...] | None:
    mask = features.get("token_pad_mask")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    return (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[1]))


def _atom_shape(features: Mapping[str, Any]) -> tuple[int, ...] | None:
    mask = features.get("atom_pad_mask")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    return (int(mask.shape[0]), int(mask.shape[1]))


def _check_shape(
    value: np.ndarray, expected: tuple[int, ...] | None, name: str
) -> None:
    if expected is not None and tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {value.shape} does not match {expected}")


def validate_compact_contact_conditioning(features: Mapping[str, Any]) -> None:
    """Reject an incomplete or malformed private contact representation."""

    if not _require_group(features, CONTACT_PRIVATE_FEATURES):
        return
    _marker(features, COMPACT_CONTACT_CONDITIONING)
    ids = _ids_payload(
        features,
        CONTACT_CONDITIONING_IDS,
        dtype=np.dtype(np.uint8),
        ndim=3,
        sentinel=CONTACT_CONDITIONING_CLASSES,
    )
    _check_shape(ids, _pair_shape(features), CONTACT_CONDITIONING_IDS)


def validate_compact_token_bonds(features: Mapping[str, Any]) -> None:
    """Reject an incomplete or malformed private bond representation."""

    if not _require_group(features, TOKEN_BOND_PRIVATE_FEATURES):
        return
    _marker(features, COMPACT_TOKEN_BONDS)
    pair = _pair_shape(features)
    flags = _ids_payload(
        features,
        TOKEN_BONDS_FLAGS,
        dtype=np.dtype(np.bool_),
        ndim=4,
        sentinel=None,
    )
    _check_shape(flags, None if pair is None else (*pair, 1), TOKEN_BONDS_FLAGS)
    ids = _ids_payload(
        features, TYPE_BONDS_IDS, dtype=np.dtype(np.uint8), ndim=3, sentinel=None
    )
    _check_shape(ids, pair, TYPE_BONDS_IDS)


def validate_compact_ref_atom_categories(features: Mapping[str, Any]) -> None:
    """Reject an incomplete or malformed private atom-category representation."""

    if not _require_group(features, REF_ATOM_PRIVATE_FEATURES):
        return
    _marker(features, COMPACT_REF_ATOM_CATEGORIES)
    atom = _atom_shape(features)
    element = _ids_payload(
        features,
        REF_ELEMENT_IDS,
        dtype=np.dtype(np.uint8),
        ndim=2,
        sentinel=REF_ELEMENT_CLASSES,
    )
    _check_shape(element, atom, REF_ELEMENT_IDS)
    chars = _ids_payload(
        features,
        REF_ATOM_NAME_CHAR_IDS,
        dtype=np.dtype(np.uint8),
        ndim=3,
        sentinel=REF_ATOM_NAME_CHAR_CLASSES,
    )
    _check_shape(chars, None if atom is None else (*atom, 4), REF_ATOM_NAME_CHAR_IDS)


def validate_compact_categories(features: Mapping[str, Any]) -> None:
    """Validate every private category group present on concrete host arrays."""

    validate_compact_contact_conditioning(features)
    validate_compact_token_bonds(features)
    validate_compact_ref_atom_categories(features)


def compact_contact_conditioning_storage(
    features: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replace the dense contact one-hot with a private ``uint8`` category."""

    if "contact_conditioning" not in features:
        validate_compact_contact_conditioning(features)
        return features
    out = _without(features, CONTACT_PRIVATE_FEATURES)
    ids = _category_ids(
        features["contact_conditioning"],
        classes=CONTACT_CONDITIONING_CLASSES,
        ndim=4,
    )
    if ids is None:
        return out
    del out["contact_conditioning"]
    out[COMPACT_CONTACT_CONDITIONING] = np.asarray(_COMPACT_VERSION, dtype=np.uint8)
    out[CONTACT_CONDITIONING_IDS] = ids
    return out


def compact_token_bond_storage(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Replace the dense bond flag/type pair with a ``bool`` mask and ``uint8`` IDs."""

    if "token_bonds" not in features and "type_bonds" not in features:
        validate_compact_token_bonds(features)
        return features
    out = _without(features, TOKEN_BOND_PRIVATE_FEATURES)
    if "token_bonds" not in features or "type_bonds" not in features:
        return out
    flags = _binary_flags(features["token_bonds"], ndim=4)
    ids = _table_ids(features["type_bonds"], ndim=3)
    if flags is None or ids is None:
        return out
    del out["token_bonds"]
    del out["type_bonds"]
    out[COMPACT_TOKEN_BONDS] = np.asarray(_COMPACT_VERSION, dtype=np.uint8)
    out[TOKEN_BONDS_FLAGS] = flags
    out[TYPE_BONDS_IDS] = ids
    return out


def compact_ref_atom_category_storage(
    features: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replace the dense atom element/name one-hots with private ``uint8`` IDs."""

    if "ref_element" not in features and "ref_atom_name_chars" not in features:
        validate_compact_ref_atom_categories(features)
        return features
    out = _without(features, REF_ATOM_PRIVATE_FEATURES)
    if "ref_element" not in features or "ref_atom_name_chars" not in features:
        return out
    element = _category_ids(
        features["ref_element"], classes=REF_ELEMENT_CLASSES, ndim=3
    )
    chars = _category_ids(
        features["ref_atom_name_chars"], classes=REF_ATOM_NAME_CHAR_CLASSES, ndim=4
    )
    if element is None or chars is None:
        return out
    del out["ref_element"]
    del out["ref_atom_name_chars"]
    out[COMPACT_REF_ATOM_CATEGORIES] = np.asarray(_COMPACT_VERSION, dtype=np.uint8)
    out[REF_ELEMENT_IDS] = element
    out[REF_ATOM_NAME_CHAR_IDS] = chars
    return out


def compact_category_storage(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Apply every Boltz-2 category compaction to one managed feature mapping."""

    compact = compact_contact_conditioning_storage(features)
    compact = compact_token_bond_storage(compact)
    return compact_ref_atom_category_storage(compact)


__all__ = [
    "COMPACT_CONTACT_CONDITIONING",
    "COMPACT_REF_ATOM_CATEGORIES",
    "COMPACT_TOKEN_BONDS",
    "CONTACT_CONDITIONING_CLASSES",
    "CONTACT_CONDITIONING_IDS",
    "CONTACT_PRIVATE_FEATURES",
    "PRIVATE_CATEGORY_FEATURES",
    "REF_ATOM_NAME_CHAR_CLASSES",
    "REF_ATOM_NAME_CHAR_IDS",
    "REF_ATOM_PRIVATE_FEATURES",
    "REF_ELEMENT_CLASSES",
    "REF_ELEMENT_IDS",
    "TOKEN_BONDS_FLAGS",
    "TOKEN_BOND_PRIVATE_FEATURES",
    "TYPE_BONDS_IDS",
    "compact_category_storage",
    "compact_contact_conditioning_storage",
    "compact_ref_atom_category_storage",
    "compact_token_bond_storage",
    "validate_compact_categories",
    "validate_compact_contact_conditioning",
    "validate_compact_ref_atom_categories",
    "validate_compact_token_bonds",
]
