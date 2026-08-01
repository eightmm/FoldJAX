"""Protenix-style inference chunk policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChunkPolicyName = Literal["auto", "manual", "off"]

PROTENIX_CHUNK_SIZE_THRESHOLDS: tuple[tuple[int, int | None], ...] = (
    (768, None),
    (1024, 256),
    (1536, 512),
    (2048, 256),
    (2560, 128),
)
PROTENIX_EXTREME_CHUNK_SIZE = 32
PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE = 5


@dataclass(frozen=True)
class ChunkConfig:
    """Resolved chunk knobs for the current static inference wrapper."""

    triangle_mul_chunk_size: int | None = None
    triangle_att_q_chunk_size: int | None = None
    single_att_q_chunk_size: int | None = None
    token_q_chunk_size: int | None = None
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
    n_sample: int,
    policy: ChunkPolicyName = "auto",
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    token_q_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
) -> ChunkConfig:
    """Resolve static inference chunk knobs from policy and explicit overrides."""

    if n_sample <= 0:
        raise ValueError("n_sample must be positive")
    if policy == "off":
        return ChunkConfig()
    if policy == "manual":
        return ChunkConfig(
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
            token_q_chunk_size=token_q_chunk_size,
            diffusion_chunk_size=diffusion_chunk_size,
        )
    if policy != "auto":
        raise ValueError(f"unknown chunk policy: {policy!r}")

    token_chunk_size = protenix_dynamic_token_chunk_size(n_token)
    auto_diffusion_chunk_size = (
        PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE
        if n_sample > PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE
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
        diffusion_chunk_size=_override_or_auto(
            diffusion_chunk_size,
            auto_diffusion_chunk_size,
        ),
    )


def _override_or_auto(value: int | None, auto_value: int | None) -> int | None:
    if value is not None:
        return value
    return auto_value
