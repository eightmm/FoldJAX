"""Shared shape-profile selection for optional JAX input padding.

The public contract deliberately describes semantic axes rather than array
positions.  Backends remain responsible for schema-aware padding, masks, RNG
and output cropping; this module only chooses and reports conservative target
sizes in one consistent way.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from foldjax.schema import PaddingConfig

MSA_PROFILE_DEPTH = 1024
OPENDDE_MSA_PROFILE_DEPTH = 1280

TOKEN_BUCKETS = (256, 512, 768, 1024, 1536, 2048, 3072, 4096)
ATOM_BUCKETS = (
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
)
MSA_BUCKETS = (1, 64, 128, 256, 512, 768, 1024, 1280, 2048, 4096, 8192, 16384)
TEMPLATE_BUCKETS = (1, 2, 4)
STRUCTURAL_TOKEN_BUCKETS = (
    256,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
)
LANGUAGE_MODEL_TOKEN_BUCKETS = (
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    5120,
)

DEFAULT_BUCKETS: dict[str, tuple[int, ...]] = {
    "tokens": TOKEN_BUCKETS,
    "atoms": ATOM_BUCKETS,
    "msa": MSA_BUCKETS,
    "templates": TEMPLATE_BUCKETS,
    "structural_tokens": STRUCTURAL_TOKEN_BUCKETS,
    "language_model_tokens": LANGUAGE_MODEL_TOKEN_BUCKETS,
}


def resolve_axis(
    actual: int,
    config: PaddingConfig,
    axis: str,
    *,
    buckets: tuple[int, ...] | None = None,
    minimum: int | None = None,
) -> int:
    """Resolve one semantic axis to an exact target no smaller than input.

    ``minimum`` is the already materialized array size for an archive that may
    itself contain padding.  Such an array is never shrunk even if its real mask
    count is smaller.
    """

    if actual < 0:
        raise ValueError(f"actual {axis} must be non-negative")
    floor = max(actual, actual if minimum is None else minimum)
    requested = getattr(config, axis)
    if requested is not None:
        if requested < floor:
            raise ValueError(
                f"padding.{axis}={requested} is smaller than the input size {floor}"
            )
        return requested
    candidates = DEFAULT_BUCKETS[axis] if buckets is None else buckets
    target = next((candidate for candidate in candidates if candidate >= floor), None)
    if target is not None:
        return target
    if config.overflow == "error":
        largest = candidates[-1] if candidates else 0
        raise ValueError(
            f"input {axis} size {floor} exceeds the largest standard bucket "
            f"{largest}; pin padding.{axis} or use overflow='exact'"
        )
    return floor


def resolve_token_axis(
    actual: int,
    config: PaddingConfig,
    axis: str,
    *,
    token_target: int,
    minimum: int | None = None,
    fixed_size: int | None = None,
) -> int:
    """Tie automatic capacity to the token profile without truncating storage.

    Explicit pins retain their meaning. Model-specific MSA limits and LM
    special-token overhead belong to callers, not to a shared bucket grid.
    """
    if token_target < 1:
        raise ValueError("token_target must be positive")
    if getattr(config, axis) is None:
        target = fixed_size
        if target is None:
            if axis == "atoms":
                target = ((24 * token_target + 31) // 32) * 32
            elif axis == "structural_tokens":
                target = 2 * token_target
            elif axis == "language_model_tokens":
                target = token_target
            else:
                raise ValueError(f"fixed_size is required for {axis}")
        config = replace(config, **{axis: target})
    return resolve_axis(actual, config, axis, minimum=minimum)


@dataclass(frozen=True, slots=True)
class PaddingPlan:
    """Resolved real/storage/target dimensions for one backend call."""

    actual: dict[str, int]
    target: dict[str, int]
    storage: dict[str, int] | None = None

    @property
    def changed(self) -> bool:
        source = self.storage or self.actual
        return any(self.target.get(axis) != size for axis, size in source.items())

    def summary(self) -> dict[str, Any]:
        return {
            "actual": dict(self.actual),
            "storage": dict(self.storage or self.actual),
            "target": dict(self.target),
            "changed": self.changed,
        }

    def message(self, model: str) -> str:
        source = self.storage or self.actual
        dimensions = ", ".join(
            f"{axis} {self.actual.get(axis, source[axis])}"
            + (
                f" (stored {source[axis]})"
                if source[axis] != self.actual.get(axis, source[axis])
                else ""
            )
            + f" -> {self.target[axis]}"
            for axis in self.target
        )
        return f"{model} padding profile: {dimensions}"
