"""Protenix-style inference chunk policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from foldjax.execution import DIFFUSION_CHUNK_SIZE

ChunkPolicyName = Literal["auto", "manual", "off"]

#: Protenix's own `infer_setting.chunk_size_thresholds`, transcribed from
#: `configs/configs_base.py`, where `-1` means "no chunking" and is written here
#: as `None`. Upstream reads it in `Protenix._get_dynamic_chunk_size`.
#:
#: This table used to carry two extra rows -- `(768, None)` and `(1024, 256)` --
#: that upstream does not have, so every job between 769 and 1024 tokens ran
#: chunked here and unchunked there, buying memory that a job that size does not
#: need. The rows had been here since the first commit and the test asserted
#: them, so nothing could report the difference; only reading upstream's config
#: found it.
PROTENIX_CHUNK_SIZE_THRESHOLDS: tuple[tuple[int, int | None], ...] = (
    (1024, None),
    (1536, 512),
    (2048, 256),
    (2560, 128),
)
PROTENIX_EXTREME_CHUNK_SIZE = 32
#: Kept as a name because upstream's config uses it, but the value now comes
#: from the one place every port reads it: a knob that means the same thing
#: should not have two constants either.
PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE = DIFFUSION_CHUNK_SIZE


@dataclass(frozen=True)
class ChunkConfig:
    """Resolved chunk knobs for the current static inference wrapper."""

    triangle_mul_chunk_size: int | None = None
    triangle_att_q_chunk_size: int | None = None
    single_att_q_chunk_size: int | None = None
    token_q_chunk_size: int | None = None
    # The outer product mean gets the same token chunk size as everything else,
    # because upstream's `MSABlock.forward` passes it the one `chunk_size` it was
    # given. Leaving it out let the `[N, N, C, C]` outer product exist at full
    # width, which is eight times the pair update it becomes.
    opm_chunk_size: int | None = None
    diffusion_chunk_size: int | None = None


def protenix_dynamic_token_chunk_size(n_token: int) -> int | None:
    """Return Protenix's threshold-based inference chunk size."""

    if n_token <= 0:
        raise ValueError("n_token must be positive")
    for threshold, chunk_size in PROTENIX_CHUNK_SIZE_THRESHOLDS:
        if n_token <= threshold:
            return chunk_size
    return PROTENIX_EXTREME_CHUNK_SIZE


def resolve_chunk_config(
    *,
    n_token: int,
    num_samples: int,
    policy: ChunkPolicyName = "auto",
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    token_q_chunk_size: int | None = None,
    opm_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
) -> ChunkConfig:
    """Resolve static inference chunk knobs from policy and explicit overrides."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if policy == "off":
        return ChunkConfig()
    if policy == "manual":
        return ChunkConfig(
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
            token_q_chunk_size=token_q_chunk_size,
            opm_chunk_size=opm_chunk_size,
            diffusion_chunk_size=diffusion_chunk_size,
        )
    if policy != "auto":
        raise ValueError(f"unknown chunk policy: {policy!r}")

    token_chunk_size = protenix_dynamic_token_chunk_size(n_token)
    auto_diffusion_chunk_size = (
        PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE
        if num_samples > PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE
        else None
    )
    return ChunkConfig(
        triangle_mul_chunk_size=_override_or_auto(
            triangle_mul_chunk_size,
            token_chunk_size,
        ),
        triangle_att_q_chunk_size=_override_or_auto(
            triangle_att_q_chunk_size,
            token_chunk_size,
        ),
        single_att_q_chunk_size=_override_or_auto(
            single_att_q_chunk_size,
            token_chunk_size,
        ),
        token_q_chunk_size=_override_or_auto(token_q_chunk_size, token_chunk_size),
        opm_chunk_size=_override_or_auto(opm_chunk_size, token_chunk_size),
        diffusion_chunk_size=_override_or_auto(
            diffusion_chunk_size,
            auto_diffusion_chunk_size,
        ),
    )


def _override_or_auto(value: int | None, auto_value: int | None) -> int | None:
    if value is not None:
        return value
    return auto_value
