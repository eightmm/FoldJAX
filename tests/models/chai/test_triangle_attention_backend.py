from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai import inference
from foldjax.models.chai.models import pairformer, triangle_cueq


def test_triangle_attention_backend_auto_detects_cueq(monkeypatch) -> None:
    monkeypatch.delenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(triangle_cueq, "cueq_available", lambda: True)
    monkeypatch.setattr(
        pairformer.jax, "devices", lambda: [SimpleNamespace(platform="gpu")]
    )

    assert pairformer._triangle_attention_backend() == "cueq"


def test_triangle_attention_backend_auto_falls_back_to_xla(monkeypatch) -> None:
    monkeypatch.delenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(triangle_cueq, "cueq_available", lambda: False)
    monkeypatch.setattr(
        pairformer.jax, "devices", lambda: [SimpleNamespace(platform="gpu")]
    )

    assert pairformer._triangle_attention_backend() == "xla"


def test_triangle_attention_backend_auto_ignores_cueq_on_cpu(monkeypatch) -> None:
    monkeypatch.delenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(triangle_cueq, "cueq_available", lambda: True)
    monkeypatch.setattr(
        pairformer.jax, "devices", lambda: [SimpleNamespace(platform="cpu")]
    )

    assert pairformer._triangle_attention_backend() == "xla"


def test_triangle_attention_backend_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "invalid")

    with pytest.raises(ValueError, match="must be 'auto', 'xla', or 'cueq'"):
        pairformer._triangle_attention_backend()


def test_high_size_xla_host_chunk_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "xla")

    assert inference._triangle_attention_host_chunk_size(512) == 64
    assert inference._triangle_attention_host_chunk_size(32) == 32


def test_high_size_cueq_keeps_requested_host_chunk(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "cueq")
    monkeypatch.setattr(triangle_cueq, "cueq_available", lambda: True)

    assert inference._triangle_attention_host_chunk_size(512) == 512


def test_1536_xla_selects_fully_bounded_pair_paths(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "xla")

    assert inference._msa_pair_subchunk_size(1536) == 512
    assert inference._use_low_memory_pairformer(1536)


def test_1536_cueq_keeps_faster_full_pairformer(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "cueq")
    monkeypatch.setattr(triangle_cueq, "cueq_available", lambda: True)

    assert inference._msa_pair_subchunk_size(1536) is None
    assert not inference._use_low_memory_pairformer(1536)


@pytest.mark.parametrize("direction", [0, 1])
def test_direction_chunk_matches_forced_xla_full_rows(
    monkeypatch, direction: int
) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "xla")
    rng = np.random.default_rng(7)
    token_count = 6
    channels = 8
    num_heads = 2
    head_dim = 4
    params = pairformer.FusedTriangleAttentionParams(
        out_scalers=jnp.asarray(rng.normal(size=channels), jnp.float32),
        pair2b_weight=jnp.asarray(
            rng.normal(size=(2 * num_heads, channels)), jnp.float32
        ),
        pair2qkvg1_weight=jnp.asarray(
            rng.normal(size=(4 * num_heads * head_dim, channels)), jnp.float32
        ),
        pair2qkvg2_weight=jnp.asarray(
            rng.normal(size=(4 * num_heads * head_dim, channels)), jnp.float32
        ),
        linear_out_weight=jnp.asarray(
            rng.normal(size=(channels, 2 * num_heads * head_dim)), jnp.float32
        ),
    )
    pair = jnp.asarray(
        rng.normal(size=(1, token_count, token_count, channels)), jnp.float32
    )
    pair_mask = jnp.asarray([[[True, True, True, True, False, False]] * token_count])
    start, size = 2, 3
    full = pairformer.fused_triangle_attention_direction(
        pair,
        pair_mask,
        params,
        direction=direction,
        attention_backend="xla",
    )
    bias = pairformer.fused_triangle_attention_bias(pair, pair_mask, params)
    value = pair if direction == 0 else jnp.swapaxes(pair, 1, 2)
    value_chunk = jnp.take(value, jnp.arange(start, start + size), axis=-3)

    chunk = pairformer.fused_triangle_attention_direction_chunk(
        value_chunk,
        bias,
        params,
        direction=direction,
    )

    np.testing.assert_array_equal(
        np.asarray(chunk), np.asarray(full[:, start : start + size])
    )
