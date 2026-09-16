from __future__ import annotations

import inspect

import pytest

from foldjax.models.protenix.chunking import (
    PROTENIX_CHUNK_SIZE_THRESHOLDS,
    PROTENIX_EXTREME_CHUNK_SIZE,
    PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS,
    PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE,
    ChunkConfig,
    ChunkSizeThresholds,
    protenix_dynamic_token_chunk_size,
    resolve_chunk_config,
)

#: `infer_setting.chunk_size_thresholds` exactly as Protenix ships it in
#: `configs/configs_base.py`, with upstream's `-1` sentinel intact. Kept in that
#: form so a reviewer can diff it against the config by eye; the translation to
#: `None` is what the assertion below exercises.
UPSTREAM_CHUNK_SIZE_THRESHOLDS = {"1024": -1, "1536": 512, "2048": 256, "2560": 128}
UPSTREAM_EXTREME_CHUNK_SIZE = 32

#: The five knobs one resolved width fans out to. Named once so a test that
#: means "nothing is chunked" cannot pass by checking four of them.
TRUNK_CHUNK_KNOBS = (
    "triangle_mul_chunk_size",
    "triangle_att_q_chunk_size",
    "single_att_q_chunk_size",
    "token_q_chunk_size",
    "opm_chunk_size",
)


def trunk_widths(config: ChunkConfig) -> tuple[int | None, ...]:
    return tuple(getattr(config, name) for name in TRUNK_CHUNK_KNOBS)


def test_protenix_dynamic_token_chunk_size_matches_upstream_config() -> None:
    """Reproduce `Protenix._get_dynamic_chunk_size` over its own thresholds.

    This pins the transcription, not the schedule this port runs: the Protenix
    callers resolve from `PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS` instead, and
    upstream's table survives as the default of these functions because OpenDDE
    resolves from it.

    The previous version of this test listed the sizes this port happened to
    return, so the two extra rows it carried -- chunking from 769 tokens where
    upstream chunks from 1025 -- were asserted rather than caught. Deriving the
    expectation from upstream's config instead means the table cannot drift
    without a failure, which is the only thing that made the difference visible.
    """
    bands = sorted(
        (int(key), value) for key, value in UPSTREAM_CHUNK_SIZE_THRESHOLDS.items()
    )

    def upstream(n_token: int) -> int | None:
        for threshold, chunk_size in bands:
            if n_token <= threshold:
                return None if chunk_size == -1 else chunk_size
        return UPSTREAM_EXTREME_CHUNK_SIZE

    probes = [1, 768, 769, 952, 970, 1024, 1025, 1531, 1536, 1537, 2048, 2560, 2561]
    for n_token in probes:
        expected = upstream(n_token)
        assert (
            protenix_dynamic_token_chunk_size(
                n_token,
                thresholds=PROTENIX_CHUNK_SIZE_THRESHOLDS,
                extreme=PROTENIX_EXTREME_CHUNK_SIZE,
            )
            == expected
        ), n_token
        # Omitting the table has to keep meaning "upstream's table", because
        # that is what OpenDDE's own tests and its CLI rely on.
        assert protenix_dynamic_token_chunk_size(n_token) == expected, n_token

    # The band the extra rows used to sit in, named so a regression says so.
    assert protenix_dynamic_token_chunk_size(970) is None
    assert protenix_dynamic_token_chunk_size(1024) is None


def test_omitting_the_table_still_resolves_upstreams_width() -> None:
    """OpenDDE's protection, asserted here because the module is shared.

    `tests/models/opendde/test_chunk_token_count.py` and its cache-profile
    sibling call `resolve_chunk_config` without a table, and OpenDDE's trunk
    genuinely needs these widths: under its bf16 trunk the triangle product
    runs the blocked xla path, where they bound real buffers. If the default
    ever flips to the Protenix-measured table, OpenDDE stops chunking silently
    and this test is where it says so.
    """
    config = resolve_chunk_config(n_token=2096, num_samples=1)

    assert trunk_widths(config) == (128, 128, 128, 128, 128)
    assert protenix_dynamic_token_chunk_size(2096) == 128


@pytest.mark.parametrize("n_token", (1003, 2096, 3012))
def test_protenix_auto_runs_the_measured_range_whole(n_token: int) -> None:
    """The shipped Protenix `auto`: no chunking at or below 3,012 tokens.

    Measured 2026-09-14, released schedule, one GPU class. At 2,096 tokens the
    chunked arm ran 204.15 s against 207.29 s unchunked with the peak identical
    to the tenth of a mebibyte (21,253.7 MiB); at 3,012 tokens 569.63/568.86 s
    against 549.27/549.19 s -- unchunked 3.5% faster on both draws -- 37,539 vs
    37,536 MiB, coordinates moving 0.069-0.072 A against a 0.070 A rerun floor.
    The widths bound tensors the fused cuEquivariance triangle path never
    forms, so they only added loop overhead.
    """
    config = resolve_chunk_config(
        n_token=n_token,
        num_samples=1,
        thresholds=PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS,
    )

    assert trunk_widths(config) == (None,) * len(TRUNK_CHUNK_KNOBS)


def test_protenix_auto_keeps_the_extreme_width_above_the_measured_range() -> None:
    """Past 3,012 tokens nothing was measured, so upstream's 32 still applies.

    The schedule therefore cliffs from "whole" to 32 at 3,013 tokens. That is
    deliberate -- the unmeasured side keeps the upstream value -- and it is the
    first thing to revisit if a larger case is ever measured.
    """
    config = resolve_chunk_config(
        n_token=3013,
        num_samples=1,
        thresholds=PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS,
    )

    assert trunk_widths(config) == (PROTENIX_EXTREME_CHUNK_SIZE,) * len(
        TRUNK_CHUNK_KNOBS
    )
    assert PROTENIX_EXTREME_CHUNK_SIZE == UPSTREAM_EXTREME_CHUNK_SIZE


def test_the_two_callers_name_the_table_they_measured() -> None:
    """Read the call sites: the default is upstream's, so omission is a no-op.

    Every assertion above passes whether or not the Protenix CLI actually asks
    for the measured table -- that is the cost of defaulting to upstream's --
    so the change only ships if this line is there. OpenDDE is asserted in the
    same breath because the two callers must not converge.
    """
    from foldjax.models.opendde import runner as opendde_runner
    from foldjax.models.protenix import runner as protenix_runner

    protenix_source = inspect.getsource(protenix_runner)
    opendde_source = inspect.getsource(opendde_runner)

    assert "thresholds=PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS" in protenix_source
    assert "PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS" not in opendde_source
    assert "thresholds=PROTENIX_CHUNK_SIZE_THRESHOLDS" in opendde_source
    assert "extreme=PROTENIX_EXTREME_CHUNK_SIZE" in opendde_source


def test_auto_chunk_config_fans_one_width_out_to_five_knobs() -> None:
    """Upstream's table, which is the only one that still resolves a width."""
    config = resolve_chunk_config(
        n_token=2000,
        num_samples=1,
        thresholds=PROTENIX_CHUNK_SIZE_THRESHOLDS,
    )

    assert trunk_widths(config) == (256, 256, 256, 256, 256)
    assert config.diffusion_chunk_size is None


def test_auto_chunk_config_leaves_the_first_band_unchunked() -> None:
    """Upstream runs everything up to 1024 tokens whole; so does this.

    This assertion used to read 952 -> 256 under a name about bounding peak
    memory, which is what kept the extra thresholds in place. Nothing measured
    ever required them: a job this size peaks around 11 GiB.
    """
    config = resolve_chunk_config(
        n_token=952,
        num_samples=5,
        thresholds=PROTENIX_CHUNK_SIZE_THRESHOLDS,
    )

    assert trunk_widths(config) == (None,) * len(TRUNK_CHUNK_KNOBS)


@pytest.mark.parametrize(
    "thresholds, expected_triangle_mul",
    (
        (PROTENIX_CHUNK_SIZE_THRESHOLDS, 512),
        (PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS, None),
    ),
)
def test_auto_chunk_config_chunks_large_sample_batches(
    thresholds: ChunkSizeThresholds,
    expected_triangle_mul: int | None,
) -> None:
    """The sample policy is untouched by the token table, under either table.

    At 1,500 tokens upstream's table is above its first band, so the two
    policies are both live and the assertion cannot pass on one of them alone;
    the measured table stops chunking the trunk and must still chunk the
    rollout.
    """
    config = resolve_chunk_config(
        n_token=1500,
        num_samples=PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE + 1,
        thresholds=thresholds,
    )

    assert config.triangle_mul_chunk_size == expected_triangle_mul
    assert config.diffusion_chunk_size == PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE


@pytest.mark.parametrize(
    "thresholds, auto_width",
    (
        (PROTENIX_CHUNK_SIZE_THRESHOLDS, 256),
        (PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS, None),
    ),
)
def test_auto_chunk_config_preserves_explicit_overrides(
    thresholds: ChunkSizeThresholds,
    auto_width: int | None,
) -> None:
    """An explicit width wins over the policy, including over "run whole"."""
    config = resolve_chunk_config(
        n_token=2000,
        num_samples=16,
        thresholds=thresholds,
        triangle_mul_chunk_size=64,
        token_q_chunk_size=96,
        diffusion_chunk_size=2,
    )

    assert config.triangle_mul_chunk_size == 64
    assert config.triangle_att_q_chunk_size == auto_width
    assert config.single_att_q_chunk_size == auto_width
    assert config.token_q_chunk_size == 96
    assert config.opm_chunk_size == auto_width
    assert config.diffusion_chunk_size == 2


def test_manual_and_off_chunk_policies_do_not_auto_fill() -> None:
    """The three spellings stay three spellings.

    Inside the measured range `auto` and `off` now agree on the five trunk
    knobs, but not on the rollout: `off` drops the sample policy too, which is
    why the policy names are not collapsed.
    """
    manual = resolve_chunk_config(
        n_token=3000,
        num_samples=16,
        policy="manual",
        token_q_chunk_size=128,
    )
    off = resolve_chunk_config(
        n_token=3000,
        num_samples=16,
        policy="off",
        token_q_chunk_size=128,
    )
    auto = resolve_chunk_config(
        n_token=3000,
        num_samples=PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE + 1,
        thresholds=PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS,
    )

    assert manual.token_q_chunk_size == 128
    assert manual.triangle_mul_chunk_size is None
    assert manual.diffusion_chunk_size is None
    assert off.token_q_chunk_size is None
    assert off.diffusion_chunk_size is None
    assert auto.diffusion_chunk_size == PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE


def test_chunk_policy_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="n_token"):
        protenix_dynamic_token_chunk_size(0)
    with pytest.raises(ValueError, match="num_samples"):
        resolve_chunk_config(n_token=1, num_samples=0)
