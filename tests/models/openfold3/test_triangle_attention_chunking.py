"""Chunked triangle attention must be exactly the unchunked result.

Chunking here is a memory transformation, not an approximation: ``I`` is a batch
axis, so each row attends over ``J`` independently and splitting the rows cannot
change any output. That makes exact equality the right assertion -- anything less
would mean the wrong axis was sliced, which is easy to do silently because the pair
representation is square and the bias's ``I`` axis is the *query* axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.triangle_attention import triangle_attention

N, C, HEADS = 23, 8, 2


def _params(seed: int = 0):
    from foldjax.models.openfold3.models.attention import AttentionParams
    from foldjax.models.openfold3.models.primitives import LayerNormParams, LinearParams
    from foldjax.models.openfold3.models.triangle_attention import (
        TriangleAttentionParams,
    )

    generator = np.random.default_rng(seed)

    def array(*shape):
        return jnp.asarray(generator.normal(size=shape) * 0.3, dtype=jnp.float32)

    c_hidden = 4
    return TriangleAttentionParams(
        layer_norm=LayerNormParams(weight=array(C), bias=array(C)),
        linear_z=LinearParams(weight=array(HEADS, C), bias=None),
        mha=AttentionParams(
            linear_q=LinearParams(weight=array(HEADS * c_hidden, C), bias=None),
            linear_k=LinearParams(weight=array(HEADS * c_hidden, C), bias=None),
            linear_v=LinearParams(weight=array(HEADS * c_hidden, C), bias=None),
            linear_o=LinearParams(weight=array(C, HEADS * c_hidden), bias=None),
            linear_g=LinearParams(weight=array(HEADS * c_hidden, C), bias=None),
        ),
    )


def _inputs(seed: int = 1, n: int = N, batch: int = 1):
    generator = np.random.default_rng(seed)
    x = jnp.asarray(generator.normal(size=(batch, n, n, C)), dtype=jnp.float32)
    mask = jnp.asarray(
        (generator.random((batch, n, n)) > 0.15).astype(np.float32)
    )
    return x, mask


@pytest.mark.parametrize("chunk_size", [1, 4, 8, N - 1, N, N + 5])
def test_chunking_changes_nothing(chunk_size: int) -> None:
    """Including chunk sizes that do not divide the row count, and one larger than
    it -- the padding path is where a silent error would live."""
    params = _params()
    x, mask = _inputs()
    reference = triangle_attention(x, params, no_heads=HEADS, mask=mask)
    chunked = triangle_attention(
        x, params, no_heads=HEADS, mask=mask, chunk_size=chunk_size
    )
    assert chunked.shape == reference.shape
    np.testing.assert_allclose(
        np.asarray(chunked, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"chunk_size={chunk_size} changed the result",
    )


@pytest.mark.parametrize("starting", [True, False])
def test_chunking_respects_the_transpose(starting: bool) -> None:
    """``starting=False`` swaps the axes before attending, so the chunked path has
    to chunk the swapped batch axis, not the original one."""
    params = _params(2)
    x, mask = _inputs(3)
    reference = triangle_attention(
        x, params, no_heads=HEADS, mask=mask, starting=starting
    )
    chunked = triangle_attention(
        x, params, no_heads=HEADS, mask=mask, starting=starting, chunk_size=5
    )
    np.testing.assert_allclose(
        np.asarray(chunked, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_chunking_works_without_a_mask() -> None:
    params = _params(4)
    x, _mask = _inputs(5)
    reference = triangle_attention(x, params, no_heads=HEADS)
    chunked = triangle_attention(x, params, no_heads=HEADS, chunk_size=6)
    np.testing.assert_allclose(
        np.asarray(chunked, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_chunking_is_jit_compatible() -> None:
    """It uses lax.map and dynamic slices, so it has to trace."""
    params = _params(6)
    x, mask = _inputs(7)
    compiled = jax.jit(
        lambda a, m: triangle_attention(
            a, params, no_heads=HEADS, mask=m, chunk_size=8
        )
    )
    np.testing.assert_allclose(
        np.asarray(compiled(x, mask), dtype=np.float64),
        np.asarray(
            triangle_attention(x, params, no_heads=HEADS, mask=mask), dtype=np.float64
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_a_fully_masked_row_does_not_produce_nan() -> None:
    """The padding path pads the mask bias with zeros for exactly this reason: a row
    masked everywhere makes softmax divide by zero."""
    params = _params(8)
    x, mask = _inputs(9)
    mask = mask.at[:, 0, :].set(0.0)
    out = triangle_attention(x, params, no_heads=HEADS, mask=mask, chunk_size=4)
    assert bool(np.isfinite(np.asarray(out)).all())


def test_extra_leading_batch_dimensions_survive() -> None:
    params = _params(10)
    x, mask = _inputs(11, batch=3)
    reference = triangle_attention(x, params, no_heads=HEADS, mask=mask)
    chunked = triangle_attention(x, params, no_heads=HEADS, mask=mask, chunk_size=7)
    assert chunked.shape == reference.shape == (3, N, N, C)
    np.testing.assert_allclose(
        np.asarray(chunked, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )
