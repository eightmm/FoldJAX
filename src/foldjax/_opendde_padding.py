"""Bounded auxiliary shapes shared by OpenDDE warming and prediction."""

from __future__ import annotations

COMPACT_PROFILE = "opendde-compact-v1"
# These are serving capacities, not chemistry-independent maxima or GPU limits.
# The 1024/1536 structural capacities retain the reported completed GFP profiles.
COMPACT_TARGETS = {
    128: (1024, 256),
    256: (2048, 512),
    384: (3072, 768),
    512: (4096, 1024),
    640: (6144, 1280),
    768: (6144, 1536),
    896: (8192, 1792),
    1024: (8192, 1856),
    1280: (12288, 2304),
    1536: (12288, 2752),
}


def compact_targets(tokens: int) -> tuple[int, int]:
    try:
        return COMPACT_TARGETS[tokens]
    except KeyError:
        raise ValueError(
            f"OpenDDE compact profile has no {tokens}-token bucket; "
            "supported buckets are 128..1024 by 128, then 1280 and 1536. "
            "Omit the profile and use explicit --pad-atoms/--pad-structural-tokens "
            "for a different capacity."
        ) from None
