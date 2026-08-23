"""Shared validation for model-native coordinate writers."""

from __future__ import annotations

from typing import Any

import numpy as np


def require_finite_coordinates(
    coordinates: Any,
    *,
    model: str,
    atom_mask: Any | None = None,
) -> np.ndarray:
    """Return host coordinates after rejecting non-finite public atoms.

    ``atom_mask`` selects the atoms that a writer will expose. Masked serving
    padding is deliberately ignored: it is not part of the public structure.
    """

    array = np.asarray(coordinates)
    public = array
    if atom_mask is not None:
        mask = np.asarray(atom_mask, dtype=bool).reshape(-1)
        if array.ndim < 2 or array.shape[-2] != mask.size:
            atom_width = array.shape[-2] if array.ndim >= 2 else None
            raise ValueError(
                f"{model} coordinate atom axis does not match its public atom mask: "
                f"{atom_width} versus {mask.size}"
            )
        public = array[..., mask, :]
    try:
        finite = bool(np.isfinite(public).all())
    except TypeError as error:
        raise ValueError(f"{model} coordinates must be numeric") from error
    if not finite:
        raise ValueError(f"{model} produced non-finite coordinates")
    return array


__all__ = ["require_finite_coordinates"]
