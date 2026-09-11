"""Exact-parity tests for query-axis chunking in attention_pair_bias_forward."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.models.primitives.attention import (
    attention_pair_bias_forward,
)


def _random_params(rng: np.random.Generator, c_s: int, c_z: int, num_heads: int):
    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.float32)

    return {
        "proj_q": {"kernel": w(c_s, c_s), "bias": w(c_s)},
        "proj_k": {"kernel": w(c_s, c_s)},
        "proj_v": {"kernel": w(c_s, c_s)},
        "proj_g": {"kernel": w(c_s, c_s)},
        "proj_o": {"kernel": w(c_s, c_s)},
        "proj_z_norm": {"scale": w(c_z), "bias": w(c_z)},
        "proj_z": {"kernel": w(c_z, num_heads)},
    }


def test_query_chunk_matches_single_shot() -> None:
    rng = np.random.default_rng(0)
    n, c_s, c_z, num_heads = 48, 16, 12, 4
    params = _random_params(rng, c_s, c_z, num_heads)
    s = jnp.asarray(rng.standard_normal((1, n, c_s)), dtype=jnp.float32)
    z = jnp.asarray(rng.standard_normal((1, n, n, c_z)), dtype=jnp.float32)
    mask = jnp.asarray((rng.random((1, n)) > 0.2).astype(np.float32))

    fwd = jax.jit(attention_pair_bias_forward, static_argnames=("chunk_size",))
    single = fwd(params, s, z, mask, chunk_size=None)
    chunked = fwd(params, s, z, mask, chunk_size=13)

    diff = float(jnp.max(jnp.abs(single - chunked)))
    assert diff < 1e-6, f"max abs diff={diff}"


def _token_transformer_params(rng: np.random.Generator, c_s: int):
    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.float32)

    return {
        "proj_q": {"kernel": w(c_s, c_s), "bias": w(c_s)},
        "proj_k": {"kernel": w(c_s, c_s)},
        "proj_v": {"kernel": w(c_s, c_s)},
        "proj_g": {"kernel": w(c_s, c_s)},
        "proj_o": {"kernel": w(c_s, c_s)},
    }


def _token_attention(n: int, chunk_size: int | None, *, heads: int = 4, c_s: int = 16):
    """Run the diffusion token transformer's own attention at one block width.

    The token transformer does not call ``attention_pair_bias_forward``; it
    calls the no-``proj_z`` variant with a bias its conditioning already
    projected. The rung in ``resolve_long_sequence_chunks`` steers this
    function, so this is the one whose blocking has to be characterised.
    """
    from foldjax.models.boltz2.models.diffusion.diffusion_transformer import (
        _attention_pair_bias_no_proj_z_forward,
    )

    rng = np.random.default_rng(7)
    params = _token_transformer_params(rng, c_s)
    s = jnp.asarray(rng.standard_normal((1, n, c_s)), dtype=jnp.float32)
    bias = jnp.asarray(rng.standard_normal((1, n, n, heads)), dtype=jnp.float32)
    mask = jnp.asarray((rng.random((1, n)) > 0.2).astype(np.float32))

    run = jax.jit(
        lambda p, sv, bv, mv: _attention_pair_bias_no_proj_z_forward(
            p,
            s=sv,
            bias=bv,
            mask=mv,
            k_in=sv,
            multiplicity=1,
            inf=1e6,
            attention_backend="xla",
            chunk_size=chunk_size,
        )
    )
    return np.asarray(run(params, s, bias, mask))


def test_token_attention_query_chunk_agrees_with_single_shot() -> None:
    """Blocking the query axis is exact arithmetic, so the values agree.

    Deliberately not ``array_equal``: the reduction is exact per query row,
    but XLA reschedules a narrow query axis and the words differ -- measured
    on CPU, ~90% of them at both 200 and 256 tokens, the second a divisible
    split rather than a ragged tail. Asserting either equality or inequality
    would pin a scheduling decision that is backend- and size-specific, so
    this pins the property that actually has to hold.
    """
    for n in (200, 256):
        dense = _token_attention(n, None)
        blocked = _token_attention(n, 128)
        assert np.max(np.abs(dense - blocked)) < 1e-5, n


def test_token_attention_below_the_block_is_bit_identical() -> None:
    """``N <= chunk_size`` short-circuits, so it compiles the same program.

    This is what keeps the 115-token Boltz-2 parity fixture untouched by the
    new default rung: its token axis never reaches the block width.
    """
    dense = _token_attention(115, None)
    for block in (128, 256):
        assert np.array_equal(dense, _token_attention(115, block))


def _policy(num_tokens: int, token_attention_chunk: int | None = None):
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        resolve_long_sequence_chunks,
    )

    return resolve_long_sequence_chunks(
        num_tokens,
        chunk_size=128,
        triangle_attention_chunk=None,
        triangle_attention_q_chunk=None,
        token_attention_chunk=token_attention_chunk,
    )["token_attention_chunk"]


def test_token_attention_rung_bounds_the_dense_band() -> None:
    """Which block each token count resolves to, across both rung edges.

    2,048 was the worst point of the curve: the dense score buffer cost
    2.50 GiB there, and one token later the long-shape branch's block of 64
    brought the same buffer down to 80 MiB. A very common input sat at the
    peak; a slightly longer one did not.
    """
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        TOKEN_ATTENTION_CHUNK,
        TOKEN_ATTENTION_DENSE_TOKEN_LIMIT,
    )

    assert _policy(115) is None
    assert _policy(TOKEN_ATTENTION_DENSE_TOKEN_LIMIT) is None
    assert _policy(TOKEN_ATTENTION_DENSE_TOKEN_LIMIT + 1) == TOKEN_ATTENTION_CHUNK
    assert _policy(1500) == TOKEN_ATTENTION_CHUNK
    assert _policy(2048) == TOKEN_ATTENTION_CHUNK
    assert _policy(2049) == 64


def test_token_attention_explicit_block_wins_at_every_size() -> None:
    """The sweep knob overrides the rung in both directions, including 0."""
    for n in (115, 1500, 2048, 3072):
        assert _policy(n, 256) == 256
        assert _policy(n, 64) == 64
        # 0 is the unblocked buffer: the measurement baseline for the rung.
        assert _policy(n, 0) == 0
