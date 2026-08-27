from __future__ import annotations

import pytest

from foldjax.models.protenix.chunking import (
    PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE,
    protenix_dynamic_token_chunk_size,
    resolve_chunk_config,
)

#: `infer_setting.chunk_size_thresholds` exactly as Protenix ships it in
#: `configs/configs_base.py`, with upstream's `-1` sentinel intact. Kept in that
#: form so a reviewer can diff it against the config by eye; the translation to
#: `None` is what the assertion below exercises.
UPSTREAM_CHUNK_SIZE_THRESHOLDS = {"1024": -1, "1536": 512, "2048": 256, "2560": 128}
UPSTREAM_EXTREME_CHUNK_SIZE = 32


def test_protenix_dynamic_token_chunk_size_matches_upstream_config() -> None:
    """Reproduce `Protenix._get_dynamic_chunk_size` over its own thresholds.

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
        assert protenix_dynamic_token_chunk_size(n_token) == upstream(n_token), n_token

    # The band the extra rows used to sit in, named so a regression says so.
    assert protenix_dynamic_token_chunk_size(970) is None
    assert protenix_dynamic_token_chunk_size(1024) is None


def test_auto_chunk_config_applies_token_policy_to_attention_chunks() -> None:
    config = resolve_chunk_config(n_token=2000, num_samples=1)

    assert config.triangle_mul_chunk_size == 256
    assert config.triangle_att_q_chunk_size == 256
    assert config.single_att_q_chunk_size == 256
    assert config.token_q_chunk_size == 256
    assert config.diffusion_chunk_size is None


def test_auto_chunk_config_leaves_the_first_band_unchunked() -> None:
    """Upstream runs everything up to 1024 tokens whole; so does this.

    This assertion used to read 952 -> 256 under a name about bounding peak
    memory, which is what kept the extra thresholds in place. Nothing measured
    ever required them: a job this size peaks around 11 GiB.
    """
    config = resolve_chunk_config(n_token=952, num_samples=5)

    assert config.triangle_mul_chunk_size is None
    assert config.triangle_att_q_chunk_size is None
    assert config.single_att_q_chunk_size is None
    assert config.token_q_chunk_size is None


def test_auto_chunk_config_chunks_large_sample_batches() -> None:
    # Above the first band, so the token policy and the sample policy are both
    # live and the assertion cannot pass on one of them alone.
    config = resolve_chunk_config(
        n_token=1500,
        num_samples=PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE + 1,
    )

    assert config.triangle_mul_chunk_size == 512
    assert config.diffusion_chunk_size == PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE


def test_auto_chunk_config_preserves_explicit_overrides() -> None:
    config = resolve_chunk_config(
        n_token=2000,
        num_samples=16,
        triangle_mul_chunk_size=64,
        token_q_chunk_size=96,
        diffusion_chunk_size=2,
    )

    assert config.triangle_mul_chunk_size == 64
    assert config.triangle_att_q_chunk_size == 256
    assert config.single_att_q_chunk_size == 256
    assert config.token_q_chunk_size == 96
    assert config.diffusion_chunk_size == 2


def test_manual_and_off_chunk_policies_do_not_auto_fill() -> None:
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

    assert manual.token_q_chunk_size == 128
    assert manual.triangle_mul_chunk_size is None
    assert manual.diffusion_chunk_size is None
    assert off.token_q_chunk_size is None


def test_chunk_policy_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="n_token"):
        protenix_dynamic_token_chunk_size(0)
    with pytest.raises(ValueError, match="num_samples"):
        resolve_chunk_config(n_token=1, num_samples=0)
