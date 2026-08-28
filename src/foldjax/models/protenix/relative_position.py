"""Private compact storage for Protenix-family relative-position features.

The released model consumes three one-hot blocks and one same-entity channel,
but those 139 channels contain only four categorical values per token pair.
Managed preprocessing stores those values here and lets the traced model rebuild
the historical dense feature at its first consumer.  The marker is deliberately
private: portable/custom feature archives continue to use ``relp`` unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

COMPACT_RELP_MARKER = "_foldjax_compact_relp"
COMPACT_RELP_RESIDUE_BIN = "_foldjax_relp_residue_bin"
COMPACT_RELP_TOKEN_BIN = "_foldjax_relp_token_bin"
COMPACT_RELP_SAME_ENTITY = "_foldjax_relp_same_entity"
COMPACT_RELP_CHAIN_BIN = "_foldjax_relp_chain_bin"

COMPACT_RELP_COMPONENTS = (
    COMPACT_RELP_RESIDUE_BIN,
    COMPACT_RELP_TOKEN_BIN,
    COMPACT_RELP_SAME_ENTITY,
    COMPACT_RELP_CHAIN_BIN,
)
COMPACT_RELP_FIELDS = (COMPACT_RELP_MARKER, *COMPACT_RELP_COMPONENTS)

_COMPACT_RELP_VERSION = np.uint8(1)
_RESIDUE_WIDTH = 66
_TOKEN_WIDTH = 66
_CHAIN_WIDTH = 6
_COMPONENT_WIDTHS = {
    COMPACT_RELP_RESIDUE_BIN: _RESIDUE_WIDTH,
    COMPACT_RELP_TOKEN_BIN: _TOKEN_WIDTH,
    COMPACT_RELP_CHAIN_BIN: _CHAIN_WIDTH,
}


def compact_relative_position_storage(
    *,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    entity_id: np.ndarray,
    sym_id: np.ndarray,
    token_index: np.ndarray,
    r_max: int = 32,
    s_max: int = 2,
) -> dict[str, np.ndarray]:
    """Return the four exact categorical components used by managed inputs."""

    if r_max != 32 or s_max != 2:
        raise ValueError("compact relp storage supports the released r_max=32, s_max=2")

    asym_id = np.asarray(asym_id)
    residue_index = np.asarray(residue_index)
    entity_id = np.asarray(entity_id)
    sym_id = np.asarray(sym_id)
    token_index = np.asarray(token_index)
    arrays = (asym_id, residue_index, entity_id, sym_id, token_index)
    if not arrays or any(value.ndim != 1 for value in arrays):
        raise ValueError("compact relp metadata must be one-dimensional")
    n_token = int(asym_id.shape[0])
    if any(value.shape != (n_token,) for value in arrays[1:]):
        raise ValueError("compact relp metadata must share one token axis")

    same_chain = asym_id[:, None] == asym_id[None, :]
    residue_delta = np.clip(
        residue_index[:, None] - residue_index[None, :] + r_max,
        0,
        2 * r_max,
    )
    residue_bins = np.where(
        same_chain, residue_delta, 2 * r_max + 1
    ).astype(np.uint8, copy=False)
    del residue_delta

    same_residue = residue_index[:, None] == residue_index[None, :]
    token_delta = np.clip(
        token_index[:, None] - token_index[None, :] + r_max,
        0,
        2 * r_max,
    )
    token_bins = np.where(
        same_chain & same_residue, token_delta, 2 * r_max + 1
    ).astype(np.uint8, copy=False)
    del same_chain, same_residue, token_delta

    same_entity = entity_id[:, None] == entity_id[None, :]
    chain_delta = np.clip(sym_id[:, None] - sym_id[None, :] + s_max, 0, 2 * s_max)
    chain_bins = np.where(
        same_entity, chain_delta, 2 * s_max + 1
    ).astype(np.uint8, copy=False)
    del chain_delta
    same_entity = same_entity.astype(np.uint8, copy=False)

    return {
        COMPACT_RELP_MARKER: np.asarray(_COMPACT_RELP_VERSION, dtype=np.uint8),
        COMPACT_RELP_RESIDUE_BIN: residue_bins,
        COMPACT_RELP_TOKEN_BIN: token_bins,
        COMPACT_RELP_SAME_ENTITY: same_entity,
        COMPACT_RELP_CHAIN_BIN: chain_bins,
    }


def normalize_relative_position_storage(
    features: Mapping[str, Any],
    *,
    n_token: int,
    token_padding_mask: Any | None = None,
    require: bool = False,
) -> dict[str, Any]:
    """Validate the private form or conservatively normalize to dense storage.

    Dense ``relp`` always wins.  This prevents a stale or caller-injected
    private marker from silently discarding custom data, and also gives dense
    and compact mappings stable, distinct PyTree identities.  Without dense
    data, a marker must carry every exact v1 component or the call fails.
    Marker-less private fragments are discarded; callers may then use the
    historical metadata-derived low-level path when ``require`` is false.
    """

    out = dict(features)
    dense_present = out.get("relp") is not None
    if dense_present:
        for name in COMPACT_RELP_FIELDS:
            out.pop(name, None)
        return out

    marker_present = COMPACT_RELP_MARKER in out
    component_present = [name for name in COMPACT_RELP_COMPONENTS if name in out]
    if not marker_present:
        for name in COMPACT_RELP_COMPONENTS:
            out.pop(name, None)
        if require:
            detail = (
                "private compact relp components have no provenance marker"
                if component_present
                else "missing dense or compact relp features"
            )
            raise ValueError(detail)
        return out

    missing = [name for name in COMPACT_RELP_COMPONENTS if name not in out]
    if missing:
        raise ValueError(
            "compact relp marker is incomplete; missing: " + ", ".join(missing)
        )

    marker = np.asarray(out[COMPACT_RELP_MARKER])
    if (
        marker.shape != ()
        or marker.dtype != np.dtype(np.uint8)
        or marker.item() != int(_COMPACT_RELP_VERSION)
    ):
        raise ValueError("compact relp marker must be scalar uint8 version 1")

    arrays: dict[str, np.ndarray] = {}
    for name in COMPACT_RELP_COMPONENTS:
        value = np.asarray(out[name])
        if value.shape != (n_token, n_token) or value.dtype != np.dtype(np.uint8):
            raise ValueError(
                f"{name} must have shape [{n_token}, {n_token}] and dtype uint8"
            )
        arrays[name] = value

    if token_padding_mask is None:
        valid = None
    else:
        token_mask = np.asarray(token_padding_mask).astype(bool, copy=False)
        if token_mask.shape != (n_token,):
            raise ValueError(f"token_padding_mask must have shape [{n_token}]")
        valid = (
            None
            if np.all(token_mask)
            else token_mask[:, None] & token_mask[None, :]
        )

    for name, width in _COMPONENT_WIDTHS.items():
        value = arrays[name]
        if valid is None:
            malformed = np.any(value >= width)
        else:
            malformed = np.any(value[valid] >= width) or np.any(value[~valid] != width)
        if malformed:
            raise ValueError(
                f"{name} contains an invalid compact relp bin or padding sentinel"
            )

    same_entity = arrays[COMPACT_RELP_SAME_ENTITY]
    if valid is None:
        malformed_same_entity = np.any(same_entity > 1)
    else:
        malformed_same_entity = np.any(same_entity[valid] > 1) or np.any(
            same_entity[~valid] != 0
        )
    if malformed_same_entity:
        raise ValueError(
            f"{COMPACT_RELP_SAME_ENTITY} must contain only 0/1 and zero padding"
        )

    out.pop("relp", None)
    return out


def pad_compact_relative_position_storage(
    features: Mapping[str, Any],
    *,
    storage_token: int,
    target_token: int,
) -> dict[str, Any]:
    """Right-pad a previously validated compact representation exactly."""

    out = dict(features)
    width = target_token - storage_token
    if width < 0:
        raise ValueError("compact relp target cannot be smaller than storage")
    pair_widths = ((0, width), (0, width))
    for name, sentinel in _COMPONENT_WIDTHS.items():
        out[name] = np.pad(
            np.asarray(features[name]),
            pair_widths,
            mode="constant",
            constant_values=sentinel,
        )
    out[COMPACT_RELP_SAME_ENTITY] = np.pad(
        np.asarray(features[COMPACT_RELP_SAME_ENTITY]),
        pair_widths,
        mode="constant",
        constant_values=0,
    )
    return out


__all__ = [
    "COMPACT_RELP_CHAIN_BIN",
    "COMPACT_RELP_COMPONENTS",
    "COMPACT_RELP_FIELDS",
    "COMPACT_RELP_MARKER",
    "COMPACT_RELP_RESIDUE_BIN",
    "COMPACT_RELP_SAME_ENTITY",
    "COMPACT_RELP_TOKEN_BIN",
    "compact_relative_position_storage",
    "normalize_relative_position_storage",
    "pad_compact_relative_position_storage",
]
