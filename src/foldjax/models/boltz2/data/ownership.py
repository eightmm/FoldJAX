"""Private compact storage for Boltz-2 token/atom ownership features."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

COMPACT_TOKEN_TO_REP_ATOM: Final = "_foldjax_compact_token_to_rep_atom"
TOKEN_TO_REP_ATOM_INDEX: Final = "_foldjax_token_to_rep_atom_index"
COMPACT_ATOM_TO_TOKEN: Final = "_foldjax_compact_atom_to_token"
ATOM_TO_TOKEN_INDEX: Final = "_foldjax_atom_to_token_index"
_COMPACT_VERSION: Final = 1
_PRIVATE_FIELDS: Final = frozenset(
    {COMPACT_TOKEN_TO_REP_ATOM, TOKEN_TO_REP_ATOM_INDEX}
)
_ATOM_TO_TOKEN_PRIVATE_FIELDS: Final = frozenset(
    {COMPACT_ATOM_TO_TOKEN, ATOM_TO_TOKEN_INDEX}
)
_LEGACY_ATOM_TO_TOKEN_PRIVATE_FIELDS: Final = frozenset(
    {"atom_to_token_ids_global", "atom_to_token_valid"}
)

# Bound the largest temporary binary mask made while validating publisher
# features.  At the largest supported production buckets the dense source is
# hundreds of MiB, so validating the complete tensor in one expression would
# temporarily duplicate a material fraction of it.
_VALIDATION_CHUNK_ELEMENTS: Final = 8 * 1024 * 1024


def _feature_shape(
    features: Mapping[str, Any], name: str, ndim: int
) -> tuple[int, ...] | None:
    value = features.get(name)
    if not isinstance(value, np.ndarray) or value.ndim != ndim:
        return None
    return tuple(int(size) for size in value.shape)


def _exact_unsigned_binary(value: np.ndarray) -> bool:
    if value.dtype.hasobject or not (
        np.issubdtype(value.dtype, np.bool_)
        or np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        return False
    zeros = value == 0
    if not np.all(zeros | (value == 1)):
        return False
    return not (
        np.issubdtype(value.dtype, np.floating)
        and np.any(np.signbit(value) & zeros)
    )


def _validate_private_representation(features: Mapping[str, Any]) -> None:
    """Reject an incomplete or malformed model-private representation."""

    has_marker = COMPACT_TOKEN_TO_REP_ATOM in features
    has_index = TOKEN_TO_REP_ATOM_INDEX in features
    if not has_marker or not has_index:
        raise ValueError(
            "compact representative storage requires both its provenance "
            "marker and index payload"
        )

    marker = features[COMPACT_TOKEN_TO_REP_ATOM]
    if not isinstance(marker, np.ndarray):
        raise TypeError(f"{COMPACT_TOKEN_TO_REP_ATOM} must be a NumPy array")
    if marker.shape != () or marker.dtype != np.dtype(np.uint8):
        raise TypeError(
            f"{COMPACT_TOKEN_TO_REP_ATOM} must be a scalar uint8"
        )
    if int(marker.item()) != _COMPACT_VERSION:
        raise ValueError(
            f"{COMPACT_TOKEN_TO_REP_ATOM} must contain version "
            f"{_COMPACT_VERSION}"
        )

    value = features[TOKEN_TO_REP_ATOM_INDEX]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{TOKEN_TO_REP_ATOM_INDEX} must be a NumPy array")
    if value.dtype != np.dtype(np.int32):
        raise TypeError(f"{TOKEN_TO_REP_ATOM_INDEX} must have dtype int32")
    token_shape = _feature_shape(features, "token_pad_mask", 2)
    atom_shape = _feature_shape(features, "atom_pad_mask", 2)
    if token_shape is None or atom_shape is None or token_shape[0] != atom_shape[0]:
        raise ValueError(
            "compact representative indices require batched token/atom masks"
        )
    if value.shape != token_shape:
        raise ValueError(
            f"{TOKEN_TO_REP_ATOM_INDEX} shape {value.shape} does not match "
            f"token_pad_mask {token_shape}"
        )
    atom_count = atom_shape[1]
    if atom_count <= 0:
        raise ValueError("compact representative indices require a non-empty atom axis")
    if value.size and (np.any(value < -1) or np.any(value >= atom_count)):
        raise ValueError(
            f"{TOKEN_TO_REP_ATOM_INDEX} entries must be -1 or in "
            f"[0, {atom_count})"
        )
    token_pad_mask = features["token_pad_mask"]
    atom_pad_mask = features["atom_pad_mask"]
    if not _exact_unsigned_binary(token_pad_mask) or not _exact_unsigned_binary(
        atom_pad_mask
    ):
        raise ValueError("compact representative indices require exact binary masks")
    token_valid = token_pad_mask.astype(bool)
    if not np.array_equal(value == -1, ~token_valid):
        raise ValueError(
            f"{TOKEN_TO_REP_ATOM_INDEX} must use -1 exactly for padded tokens"
        )
    valid = value >= 0
    if np.any(valid):
        batch_index = np.broadcast_to(
            np.arange(value.shape[0], dtype=np.intp)[:, None], value.shape
        )
        if np.any(~atom_pad_mask.astype(bool)[batch_index[valid], value[valid]]):
            raise ValueError(
                f"{TOKEN_TO_REP_ATOM_INDEX} must reference unpadded atoms"
            )


def _validate_atom_to_token_private_representation(
    features: Mapping[str, Any],
) -> None:
    """Reject an incomplete or malformed compact atom-owner mapping."""

    has_marker = COMPACT_ATOM_TO_TOKEN in features
    has_index = ATOM_TO_TOKEN_INDEX in features
    if not has_marker or not has_index:
        raise ValueError(
            "compact atom ownership requires both its provenance marker "
            "and index payload"
        )

    marker = features[COMPACT_ATOM_TO_TOKEN]
    if not isinstance(marker, np.ndarray):
        raise TypeError(f"{COMPACT_ATOM_TO_TOKEN} must be a NumPy array")
    if marker.shape != () or marker.dtype != np.dtype(np.uint8):
        raise TypeError(f"{COMPACT_ATOM_TO_TOKEN} must be a scalar uint8")
    if int(marker.item()) != _COMPACT_VERSION:
        raise ValueError(
            f"{COMPACT_ATOM_TO_TOKEN} must contain version {_COMPACT_VERSION}"
        )

    value = features[ATOM_TO_TOKEN_INDEX]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{ATOM_TO_TOKEN_INDEX} must be a NumPy array")
    if value.dtype != np.dtype(np.int32):
        raise TypeError(f"{ATOM_TO_TOKEN_INDEX} must have dtype int32")
    token_shape = _feature_shape(features, "token_pad_mask", 2)
    atom_shape = _feature_shape(features, "atom_pad_mask", 2)
    if token_shape is None or atom_shape is None or token_shape[0] != atom_shape[0]:
        raise ValueError("compact atom ownership requires batched token/atom masks")
    if value.shape != atom_shape:
        raise ValueError(
            f"{ATOM_TO_TOKEN_INDEX} shape {value.shape} does not match "
            f"atom_pad_mask {atom_shape}"
        )
    if atom_shape[1] <= 0:
        raise ValueError("compact atom ownership requires a non-empty atom axis")
    token_count = token_shape[1]
    if token_count <= 0:
        raise ValueError("compact atom ownership requires a non-empty token axis")
    if value.size and (np.any(value < -1) or np.any(value >= token_count)):
        raise ValueError(
            f"{ATOM_TO_TOKEN_INDEX} entries must be -1 or in "
            f"[0, {token_count})"
        )
    token_pad_mask = features["token_pad_mask"]
    atom_pad_mask = features["atom_pad_mask"]
    if not _exact_unsigned_binary(token_pad_mask) or not _exact_unsigned_binary(
        atom_pad_mask
    ):
        raise ValueError("compact atom ownership requires exact binary masks")
    atom_valid = atom_pad_mask.astype(bool)
    if not np.array_equal(value == -1, ~atom_valid):
        raise ValueError(f"{ATOM_TO_TOKEN_INDEX} must use -1 exactly for padded atoms")
    valid = value >= 0
    if np.any(valid):
        batch_index = np.broadcast_to(
            np.arange(value.shape[0], dtype=np.intp)[:, None], value.shape
        )
        if np.any(~token_pad_mask.astype(bool)[batch_index[valid], value[valid]]):
            raise ValueError(f"{ATOM_TO_TOKEN_INDEX} must reference unpadded tokens")


def _without_private_fields(features: Mapping[str, Any]) -> Mapping[str, Any]:
    if not any(name in features for name in _PRIVATE_FIELDS):
        return features
    clean = dict(features)
    for name in _PRIVATE_FIELDS:
        clean.pop(name, None)
    return clean


def _without_fields(
    features: Mapping[str, Any], names: frozenset[str]
) -> Mapping[str, Any]:
    if not any(name in features for name in names):
        return features
    clean = dict(features)
    for name in names:
        clean.pop(name, None)
    return clean


def drop_token_to_rep_atom_storage(
    features: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Drop all representative ownership fields from a graph that cannot read them.

    Trunk-only managed prediction has no confidence or affinity consumer, so it
    removes dense, complete-private, and incomplete-private forms without
    scanning or validating them. The caller's public mapping is never mutated.
    Full prediction instead uses :func:`compact_token_to_rep_atom_storage` and
    rejects incomplete private representations.
    """

    names = {"token_to_rep_atom", *_PRIVATE_FIELDS}
    if not any(name in features for name in names):
        return features
    clean = dict(features)
    for name in names:
        clean.pop(name, None)
    return clean


def compact_token_to_rep_atom_storage(
    features: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replace a native dense representative map with private int32 indices.

    The public featurizer remains publisher-compatible.  Managed prediction
    calls use this helper on their private feature mapping before padding and
    device transfer.  A dense input is compacted only when it is an exact
    batched 0/1 map with at most one representative per token; zero rows use
    the ``-1`` sentinel.  Any unfamiliar dense/custom layout falls back by
    identity, preserving the direct model-call contract.

    Dense input is authoritative: stale private fields are ignored and removed
    before either generating a fresh complete representation or retaining the
    dense fallback.  Without dense input, the private representation must be a
    complete pair consisting of a scalar uint8 v1 provenance marker and its
    int32 payload. Incomplete pairs fail explicitly.
    """

    if "token_to_rep_atom" not in features:
        if any(name in features for name in _PRIVATE_FIELDS):
            _validate_private_representation(features)
        return features
    dense = features["token_to_rep_atom"]
    if not isinstance(dense, np.ndarray) or dense.ndim != 3:
        return _without_private_fields(features)
    if dense.dtype.hasobject or not (
        np.issubdtype(dense.dtype, np.bool_)
        or np.issubdtype(dense.dtype, np.integer)
        or np.issubdtype(dense.dtype, np.floating)
    ):
        return _without_private_fields(features)

    batch, tokens, atoms = (int(size) for size in dense.shape)
    token_shape = _feature_shape(features, "token_pad_mask", 2)
    atom_shape = _feature_shape(features, "atom_pad_mask", 2)
    if token_shape != (batch, tokens) or atom_shape != (batch, atoms):
        return _without_private_fields(features)
    if atoms <= 0 or atoms - 1 > np.iinfo(np.int32).max:
        return _without_private_fields(features)
    token_pad_mask = features["token_pad_mask"]
    atom_pad_mask = features["atom_pad_mask"]
    if not _exact_unsigned_binary(token_pad_mask) or not _exact_unsigned_binary(
        atom_pad_mask
    ):
        return _without_private_fields(features)
    token_valid = token_pad_mask.astype(bool)
    atom_valid = atom_pad_mask.astype(bool)

    indices = np.empty((batch, tokens), dtype=np.int32)
    rows_per_chunk = max(1, _VALIDATION_CHUNK_ELEMENTS // atoms)
    for batch_index in range(batch):
        for start in range(0, tokens, rows_per_chunk):
            stop = min(tokens, start + rows_per_chunk)
            block = dense[batch_index, start:stop]
            zeros = block == 0
            if not np.all(zeros | (block == 1)):
                return _without_private_fields(features)
            if np.issubdtype(block.dtype, np.floating) and np.any(
                np.signbit(block) & zeros
            ):
                return _without_private_fields(features)
            counts = np.count_nonzero(block, axis=-1)
            if np.any(counts > 1):
                return _without_private_fields(features)
            expected_counts = token_valid[batch_index, start:stop].astype(counts.dtype)
            if not np.array_equal(counts, expected_counts):
                return _without_private_fields(features)
            selected = np.argmax(block, axis=-1).astype(np.int32, copy=False)
            present = counts == 1
            if np.any(present) and np.any(
                ~atom_valid[batch_index, selected[present]]
            ):
                return _without_private_fields(features)
            indices[batch_index, start:stop] = np.where(counts == 1, selected, -1)

    compact = dict(_without_private_fields(features))
    del compact["token_to_rep_atom"]
    compact[COMPACT_TOKEN_TO_REP_ATOM] = np.asarray(
        _COMPACT_VERSION, dtype=np.uint8
    )
    compact[TOKEN_TO_REP_ATOM_INDEX] = indices
    return compact


def compact_atom_to_token_storage(
    features: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replace a native dense atom-owner map with private int32 token IDs.

    Managed prediction calls this before padding and device placement. Native
    publisher rows contain exactly one token owner for each real atom, while
    padded atom rows are all zero and become the ``-1`` sentinel. Unfamiliar
    dense/custom layouts retain the historical dense representation unchanged.

    Dense input is authoritative. Any stale private or former CP-only fields
    are removed before a fresh compact pair is generated or dense fallback is
    returned. Without dense input, the private marker/payload pair is validated
    exactly and incomplete pairs fail explicitly.
    """

    private_fields = frozenset(
        {*_ATOM_TO_TOKEN_PRIVATE_FIELDS, *_LEGACY_ATOM_TO_TOKEN_PRIVATE_FIELDS}
    )
    if "atom_to_token" not in features:
        if any(name in features for name in _ATOM_TO_TOKEN_PRIVATE_FIELDS):
            _validate_atom_to_token_private_representation(features)
            return _without_fields(features, _LEGACY_ATOM_TO_TOKEN_PRIVATE_FIELDS)
        if any(name in features for name in _LEGACY_ATOM_TO_TOKEN_PRIVATE_FIELDS):
            raise ValueError(
                "legacy CP atom ownership fields require dense atom_to_token"
            )
        return features

    dense = features["atom_to_token"]
    if not isinstance(dense, np.ndarray) or dense.ndim != 3:
        return _without_fields(features, private_fields)
    if dense.dtype.hasobject or not (
        np.issubdtype(dense.dtype, np.bool_)
        or np.issubdtype(dense.dtype, np.integer)
        or np.issubdtype(dense.dtype, np.floating)
    ):
        return _without_fields(features, private_fields)

    batch, atoms, tokens = (int(size) for size in dense.shape)
    token_shape = _feature_shape(features, "token_pad_mask", 2)
    atom_shape = _feature_shape(features, "atom_pad_mask", 2)
    if token_shape != (batch, tokens) or atom_shape != (batch, atoms):
        return _without_fields(features, private_fields)
    if atoms <= 0 or tokens <= 0 or tokens - 1 > np.iinfo(np.int32).max:
        return _without_fields(features, private_fields)
    token_pad_mask = features["token_pad_mask"]
    atom_pad_mask = features["atom_pad_mask"]
    if not _exact_unsigned_binary(token_pad_mask) or not _exact_unsigned_binary(
        atom_pad_mask
    ):
        return _without_fields(features, private_fields)
    token_valid = token_pad_mask.astype(bool)
    atom_valid = atom_pad_mask.astype(bool)

    indices = np.empty((batch, atoms), dtype=np.int32)
    rows_per_chunk = max(1, _VALIDATION_CHUNK_ELEMENTS // tokens)
    for batch_index in range(batch):
        for start in range(0, atoms, rows_per_chunk):
            stop = min(atoms, start + rows_per_chunk)
            block = dense[batch_index, start:stop]
            zeros = block == 0
            if not np.all(zeros | (block == 1)):
                return _without_fields(features, private_fields)
            if np.issubdtype(block.dtype, np.floating) and np.any(
                np.signbit(block) & zeros
            ):
                return _without_fields(features, private_fields)
            counts = np.count_nonzero(block, axis=-1)
            if np.any(counts > 1):
                return _without_fields(features, private_fields)
            expected_counts = atom_valid[batch_index, start:stop].astype(counts.dtype)
            if not np.array_equal(counts, expected_counts):
                return _without_fields(features, private_fields)
            selected = np.argmax(block, axis=-1).astype(np.int32, copy=False)
            present = counts == 1
            if np.any(present) and np.any(
                ~token_valid[batch_index, selected[present]]
            ):
                return _without_fields(features, private_fields)
            indices[batch_index, start:stop] = np.where(present, selected, -1)

    compact = dict(_without_fields(features, private_fields))
    del compact["atom_to_token"]
    compact[COMPACT_ATOM_TO_TOKEN] = np.asarray(_COMPACT_VERSION, dtype=np.uint8)
    compact[ATOM_TO_TOKEN_INDEX] = indices
    return compact


__all__ = [
    "ATOM_TO_TOKEN_INDEX",
    "COMPACT_ATOM_TO_TOKEN",
    "COMPACT_TOKEN_TO_REP_ATOM",
    "TOKEN_TO_REP_ATOM_INDEX",
    "compact_atom_to_token_storage",
    "compact_token_to_rep_atom_storage",
    "drop_token_to_rep_atom_storage",
]
