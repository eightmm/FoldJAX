"""Rebuild Boltz-2's dense categorical features from private compact IDs.

The managed path narrows five publisher arrays on the host
(:mod:`foldjax.models.boltz2.data.compact_categories`); this boundary rebuilds
them at the model entry so every existing consumer keeps reading the array it
has always read.

The restored arrays deliberately carry the *canonicalized publisher* dtype --
``int64`` canonicalizes to ``int32`` while ``jax_enable_x64`` is off -- and not
the consuming module's compute dtype.  ``contact_conditioning`` is read twice:
once as a slice cast to the compute dtype, and once as ``flags`` whose own
dtype drives ``one - sum(flags[..., 0:2])`` and therefore the promotion of
``encoded * (one - ...)``.  Rebuilding at a float dtype would promote that
product from BF16 to FP32 and move the result.  Reproducing the argument
exactly leaves the traced program unchanged instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.data.compact_categories import (
    COMPACT_CONTACT_CONDITIONING,
    COMPACT_REF_ATOM_CATEGORIES,
    COMPACT_TOKEN_BONDS,
    CONTACT_CONDITIONING_CLASSES,
    CONTACT_CONDITIONING_IDS,
    CONTACT_PRIVATE_FEATURES,
    REF_ATOM_NAME_CHAR_CLASSES,
    REF_ATOM_NAME_CHAR_IDS,
    REF_ATOM_PRIVATE_FEATURES,
    REF_ELEMENT_CLASSES,
    REF_ELEMENT_IDS,
    TOKEN_BOND_PRIVATE_FEATURES,
    TOKEN_BONDS_FLAGS,
    TYPE_BONDS_IDS,
    validate_compact_contact_conditioning,
    validate_compact_ref_atom_categories,
    validate_compact_token_bonds,
)


def _restore_contact(
    out: dict[str, Any], feats: Mapping[str, Any], int_dtype: jnp.dtype
) -> None:
    _require_marker(feats, COMPACT_CONTACT_CONDITIONING)
    _require_ids(feats, CONTACT_CONDITIONING_IDS, ndim=3)
    out["contact_conditioning"] = _one_hot(
        feats[CONTACT_CONDITIONING_IDS], CONTACT_CONDITIONING_CLASSES, int_dtype
    )


def _restore_token_bonds(
    out: dict[str, Any], feats: Mapping[str, Any], int_dtype: jnp.dtype
) -> None:
    _require_marker(feats, COMPACT_TOKEN_BONDS)
    flags = jnp.asarray(feats[TOKEN_BONDS_FLAGS])
    if flags.dtype != jnp.bool_ or flags.ndim != 4:
        raise TypeError(f"{TOKEN_BONDS_FLAGS} must be a rank-4 bool array")
    _require_ids(feats, TYPE_BONDS_IDS, ndim=3)
    out["token_bonds"] = flags.astype(jnp.float32)
    out["type_bonds"] = jnp.asarray(feats[TYPE_BONDS_IDS]).astype(int_dtype)


def _restore_ref_atom_categories(
    out: dict[str, Any], feats: Mapping[str, Any], int_dtype: jnp.dtype
) -> None:
    _require_marker(feats, COMPACT_REF_ATOM_CATEGORIES)
    _require_ids(feats, REF_ELEMENT_IDS, ndim=2)
    _require_ids(feats, REF_ATOM_NAME_CHAR_IDS, ndim=3)
    out["ref_element"] = _one_hot(
        feats[REF_ELEMENT_IDS], REF_ELEMENT_CLASSES, int_dtype
    )
    out["ref_atom_name_chars"] = _one_hot(
        feats[REF_ATOM_NAME_CHAR_IDS], REF_ATOM_NAME_CHAR_CLASSES, int_dtype
    )


#: ``(dense names, private names, host validator, graph restore)`` per group.
#: Every row carries its own restore function so a group can never be rebuilt
#: from another group's payload.
_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...], Any, Any], ...] = (
    (
        ("contact_conditioning",),
        CONTACT_PRIVATE_FEATURES,
        validate_compact_contact_conditioning,
        _restore_contact,
    ),
    (
        ("token_bonds", "type_bonds"),
        TOKEN_BOND_PRIVATE_FEATURES,
        validate_compact_token_bonds,
        _restore_token_bonds,
    ),
    (
        ("ref_element", "ref_atom_name_chars"),
        REF_ATOM_PRIVATE_FEATURES,
        validate_compact_ref_atom_categories,
        _restore_ref_atom_categories,
    ),
)


def _publisher_int_dtype() -> jnp.dtype:
    """The dtype an ``int64`` publisher argument reaches the graph with."""

    return jax.dtypes.canonicalize_dtype(np.dtype(np.int64))


def _one_hot(ids: Any, classes: int, dtype: jnp.dtype) -> jnp.ndarray:
    """One-hot the IDs; the ``classes`` sentinel rebuilds an all-zero row."""

    return jax.nn.one_hot(jnp.asarray(ids).astype(jnp.int32), classes, dtype=dtype)


def _validate_concrete(features: Mapping[str, Any], names: tuple[str, ...], validate):
    values = [features[name] for name in names]
    if any(isinstance(value, jax.core.Tracer) for value in values):
        return
    concrete = {name: np.asarray(features[name]) for name in names}
    for name in ("token_pad_mask", "atom_pad_mask"):
        value = features.get(name)
        if value is not None and not isinstance(value, jax.core.Tracer):
            concrete[name] = np.asarray(value)
    validate(concrete)


def restore_compact_categories(feats: Mapping[str, Any]) -> Mapping[str, Any]:
    """Restore the dense categorical features a compact managed input replaced.

    Dense input stays authoritative and drops stale private provenance.  A
    complete private group is rebuilt in graph; an incomplete or malformed
    private-only group fails before it can reach a consumer.  A mapping with
    neither form is returned unchanged, so a caller that never supplied the
    feature keeps its historical failure at the historical consumer.
    """

    if not any(name in feats for name in _PRIVATE_NAMES):
        return feats

    out = dict(feats)
    int_dtype = _publisher_int_dtype()
    for dense_names, private_names, validate, restore in _GROUPS:
        if any(name in feats for name in dense_names):
            for name in private_names:
                out.pop(name, None)
            continue
        present = [name for name in private_names if name in feats]
        if not present:
            continue
        if len(present) != len(private_names):
            missing = [name for name in private_names if name not in feats]
            raise ValueError(
                "compact Boltz-2 categories require every member of their "
                "private group; missing " + ", ".join(missing)
            )
        _validate_concrete(feats, private_names, validate)
        restore(out, feats, int_dtype)
        for name in private_names:
            out.pop(name, None)
    return out


def _require_marker(feats: Mapping[str, Any], name: str) -> None:
    marker = jnp.asarray(feats[name])
    if marker.ndim != 0 or marker.dtype != jnp.uint8:
        raise TypeError(f"{name} must be a scalar uint8")


def _require_ids(feats: Mapping[str, Any], name: str, *, ndim: int) -> None:
    ids = jnp.asarray(feats[name])
    if ids.dtype != jnp.uint8:
        raise TypeError(f"{name} must have dtype uint8")
    if ids.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")


_PRIVATE_NAMES = tuple(
    name for _, private_names, _, _ in _GROUPS for name in private_names
)


__all__ = ["restore_compact_categories"]
