from __future__ import annotations

import pytest

from foldjax.models.protenix.chunking import (
    PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE,
    protenix_dynamic_token_chunk_size,
    resolve_chunk_config,
)


def test_protenix_dynamic_token_chunk_size_matches_thresholds() -> None:
    assert protenix_dynamic_token_chunk_size(768) is None
    assert protenix_dynamic_token_chunk_size(769) == 256
    assert protenix_dynamic_token_chunk_size(952) == 256
    assert protenix_dynamic_token_chunk_size(1024) == 256
    assert protenix_dynamic_token_chunk_size(1025) == 512
    assert protenix_dynamic_token_chunk_size(1536) == 512
    assert protenix_dynamic_token_chunk_size(1537) == 256
    assert protenix_dynamic_token_chunk_size(2048) == 256
    assert protenix_dynamic_token_chunk_size(2049) == 128
    assert protenix_dynamic_token_chunk_size(2560) == 128
    assert protenix_dynamic_token_chunk_size(2561) == 32


def test_auto_chunk_config_applies_token_policy_to_attention_chunks() -> None:
    config = resolve_chunk_config(n_token=2000, n_sample=1)

    assert config.triangle_mul_chunk_size == 256
    assert config.triangle_att_q_chunk_size == 256
    assert config.single_att_q_chunk_size == 256
    assert config.token_q_chunk_size == 256
    assert config.diffusion_chunk_size is None


def test_auto_chunk_config_bounds_peak_memory_for_measured_952_token_bucket() -> None:
    config = resolve_chunk_config(n_token=952, n_sample=5)

    assert config.triangle_mul_chunk_size == 256
    assert config.triangle_att_q_chunk_size == 256
    assert config.single_att_q_chunk_size == 256
    assert config.token_q_chunk_size == 256


def test_auto_chunk_config_chunks_large_sample_batches() -> None:
    config = resolve_chunk_config(
        n_token=1000,
        n_sample=PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE + 1,
    )

    assert config.triangle_mul_chunk_size == 256
    assert config.diffusion_chunk_size == PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE


def test_auto_chunk_config_preserves_explicit_overrides() -> None:
    config = resolve_chunk_config(
        n_token=2000,
        n_sample=16,
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
        n_sample=16,
        policy="manual",
        token_q_chunk_size=128,
    )
    off = resolve_chunk_config(
        n_token=3000,
        n_sample=16,
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
    with pytest.raises(ValueError, match="n_sample"):
        resolve_chunk_config(n_token=1, n_sample=0)
