"""Protenix-style inference chunk policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from foldjax.execution import DIFFUSION_CHUNK_SIZE

ChunkPolicyName = Literal["auto", "manual", "off"]

#: A token count ceiling paired with the chunk width to use at or below it, in
#: ascending order; `None` means "run whole". Past the last row the `extreme`
#: width applies.
ChunkSizeThresholds = tuple[tuple[int, int | None], ...]

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
#:
#: It is still the transcription and nothing else: it is what OpenDDE resolves
#: from, and it is the default of the two functions below, so a caller that says
#: nothing keeps upstream's schedule.
PROTENIX_CHUNK_SIZE_THRESHOLDS: ChunkSizeThresholds = (
    (1024, None),
    (1536, 512),
    (2048, 256),
    (2560, 128),
)
PROTENIX_EXTREME_CHUNK_SIZE = 32

#: What Protenix's `auto` policy resolves *in this port*, measured rather than
#: transcribed, for the five trunk knobs (triangle multiplication, triangle
#: attention, single attention, the token query, the outer product mean).
#:
#: At 2,096 tokens, released schedule, one GPU class (2026-09-14): chunking from
#: the table above runs 204.15 s against 207.29 s unchunked, and the peak is
#: identical to the tenth of a mebibyte, 21,253.7 MiB in both arms. At 3,012
#: tokens the chunked arm runs 569.63 s and 568.86 s against 549.27 s and
#: 549.19 s unchunked -- unchunked is 3.5% faster on both draws -- for a peak of
#: 37,539 MiB against 37,536 MiB, with coordinates moving 0.069-0.072 A against
#: an auto-vs-auto rerun floor of 0.070 A, which is to say at the floor.
#:
#: The mechanism is that Protenix's triangle multiplication and triangle
#: attention run fused cuEquivariance kernels that never form the tensors these
#: widths were bounding, so the widths only add loop overhead. **This diverges
#: from upstream's config on purpose**: the policy in this repository is
#: accuracy equivalence, not upstream fidelity, and chunking the fused trunk
#: costs time while saving no bytes.
#:
#: Above the measured range the upstream extreme width (32) still applies,
#: because nothing above 3,012 tokens was measured -- so the schedule cliffs
#: from "whole" to 32 at 3,013 tokens.
#:
#: OpenDDE shares these helpers and must not share this table: under its shipped
#: bf16 trunk `_with_cueq_triangle_defaults`
#: (`foldjax/models/opendde/models/model.py`) routes triangle multiplication to
#: the blocked xla path, where these widths bound real buffers, so its call site
#: names `PROTENIX_CHUNK_SIZE_THRESHOLDS` explicitly.
PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS: ChunkSizeThresholds = ((3012, None),)

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


def protenix_dynamic_token_chunk_size(
    n_token: int,
    *,
    thresholds: ChunkSizeThresholds = PROTENIX_CHUNK_SIZE_THRESHOLDS,
    extreme: int = PROTENIX_EXTREME_CHUNK_SIZE,
) -> int | None:
    """Return the threshold-based inference chunk size for `n_token`.

    The default is upstream's transcription, so omitting `thresholds` keeps
    upstream's schedule; the Protenix callers opt into
    `PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS` by naming it.
    """

    if n_token <= 0:
        raise ValueError("n_token must be positive")
    for threshold, chunk_size in thresholds:
        if n_token <= threshold:
            return chunk_size
    return extreme


def resolve_chunk_config(
    *,
    n_token: int,
    num_samples: int,
    policy: ChunkPolicyName = "auto",
    thresholds: ChunkSizeThresholds = PROTENIX_CHUNK_SIZE_THRESHOLDS,
    extreme: int = PROTENIX_EXTREME_CHUNK_SIZE,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    token_q_chunk_size: int | None = None,
    opm_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
) -> ChunkConfig:
    """Resolve static inference chunk knobs from policy and explicit overrides.

    `thresholds`/`extreme` default to upstream's table, which is what OpenDDE
    needs; the Protenix callers pass the measured table. `diffusion_chunk_size`
    is resolved from `num_samples` either way and no measurement touched it.
    """

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

    token_chunk_size = protenix_dynamic_token_chunk_size(
        n_token,
        thresholds=thresholds,
        extreme=extreme,
    )
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
