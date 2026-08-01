from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import foldjax.models.chai.models.confidence as confidence


def test_chunked_sdpa_matches_unchunked_fp32_softmax() -> None:
    rng = np.random.default_rng(20260715)
    q = jnp.asarray(rng.normal(size=(2, 3, 8, 4)), dtype=jnp.bfloat16)
    k = jnp.asarray(rng.normal(size=(2, 3, 8, 4)), dtype=jnp.bfloat16)
    v = jnp.asarray(rng.normal(size=(2, 3, 8, 5)), dtype=jnp.bfloat16)
    bias = jnp.asarray(rng.normal(size=(2, 1, 8, 8)), dtype=jnp.bfloat16)

    expected = confidence._sdpa(q, k, v, bias)
    actual = confidence._chunked_sdpa(
        q, k, v, bias, query_chunk_size=4, key_chunk_size=4
    )

    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        rtol=0,
        atol=1 / 128,
    )


def test_chunked_sdpa_caps_materialized_query_axis(monkeypatch) -> None:
    observed_query_sizes: list[int] = []

    def fake_logits(q, k, bias):
        del bias
        observed_query_sizes.append((q.shape[-2], k.shape[-2]))
        return jnp.zeros(q.shape[:-2] + (q.shape[-2], k.shape[-2]), jnp.float32)

    monkeypatch.setattr(confidence, "_attention_logits", fake_logits)
    q = jnp.zeros((8, 32, 32, 64), dtype=jnp.bfloat16)
    k = jnp.zeros_like(q)
    v = jnp.zeros_like(q)
    bias = jnp.zeros((8, 1, 32, 32), dtype=jnp.bfloat16)

    output = confidence._chunked_sdpa(
        q, k, v, bias, query_chunk_size=4, key_chunk_size=4
    )
    jax.block_until_ready(output)

    assert output.shape == q.shape
    assert observed_query_sizes == [(4, 4)]
