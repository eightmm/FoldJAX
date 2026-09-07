"""Prospective coordinate gate; callers must separately prove input/RNG identity."""

import math
from collections.abc import Mapping


def coordinate_gate(entity_rmsd: Mapping, *, threshold: float = 0.05) -> dict:
    """Require five finite nonnegative RMSDs for every entity, with no averaging."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if not entity_rmsd:
        raise ValueError("no entity measurements")
    maxima = {}
    for entity, values in entity_rmsd.items():
        if str(entity) in maxima:
            raise ValueError("entity labels collide when serialized")
        if len(values) != 5:
            raise ValueError("each entity requires five paired samples")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("RMSDs must be finite nonnegative numbers")
        maxima[str(entity)] = max(values)
    return {
        "threshold_angstrom": threshold,
        "entity_max_rmsd": maxima,
        "coordinate_gate_passed": all(value <= threshold for value in maxima.values()),
        "scope": (
            "coordinate gate only; not input, RNG, confidence or universal acceptance"
        ),
    }
