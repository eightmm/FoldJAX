"""Conservative schema proof shared by managed structure-output views."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_EXPLICIT_OUTPUT_ATOM_FIELDS = (
    "output_atom_name",
    "output_atom_element",
    "output_atom_res_name",
    "output_atom_chain_id",
    "output_atom_res_id",
)


def has_complete_output_atom_metadata(features: Mapping[str, Any]) -> bool:
    """Whether explicit writer metadata exactly covers the generated atoms."""

    atom_index = features.get("atom_to_token_idx")
    atom_shape = getattr(atom_index, "shape", None)
    if atom_shape is None or len(atom_shape) != 1:
        return False
    expected = tuple(atom_shape)
    return all(
        name in features and getattr(features[name], "shape", None) == expected
        for name in _EXPLICIT_OUTPUT_ATOM_FIELDS
    )
