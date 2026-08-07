"""Chunking the outer product mean must not change what it computes.

The outer product is the largest single tensor OpenFold3 builds: ``[N, N, C, C]``
before the projection that turns it into ``[N, N, C_z]``, which at the released
widths is eight times the size of its own result and 34.6 GiB at 3012 tokens.
Chunking the first token axis and projecting inside the block is how upstream
avoids it, and it is safe because that axis is a pure batch axis of the output --
the sum runs over MSA rows, which chunking never splits.

The failure this guards against is splitting the wrong axis. ``a`` and ``b`` are
both ``[..., N_token, N_seq, C]`` and the einsum is symmetric in their token axes,
so chunking ``b`` instead would still typecheck, still produce the right shape, and
silently compute each block against only its own slice.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.msa import (
    OuterProductMeanParams,
    outer_product_mean,
)
from foldjax.models.openfold3.models.primitives import LayerNormParams, LinearParams

N_SEQ, N_TOKEN, C_M, C_HIDDEN, C_Z = 7, 23, 8, 4, 6


def _params(seed: int = 0) -> OuterProductMeanParams:
    generator = np.random.default_rng(seed)

    def array(*shape):
        return jnp.asarray(generator.normal(size=shape) * 0.3, dtype=jnp.float32)

    return OuterProductMeanParams(
        layer_norm=LayerNormParams(weight=array(C_M), bias=array(C_M)),
        linear_1=LinearParams(weight=array(C_HIDDEN, C_M), bias=None),
        linear_2=LinearParams(weight=array(C_HIDDEN, C_M), bias=None),
        linear_out=LinearParams(
            weight=array(C_Z, C_HIDDEN * C_HIDDEN), bias=array(C_Z)
        ),
    )


def _inputs(seed: int = 1):
    generator = np.random.default_rng(seed)
    m = jnp.asarray(
        generator.normal(size=(1, N_SEQ, N_TOKEN, C_M)), dtype=jnp.float32
    )
    mask = jnp.asarray(
        (generator.random((1, N_SEQ, N_TOKEN)) > 0.2).astype(np.float32)
    )
    return m, mask


@pytest.mark.parametrize(
    "chunk_size", [1, 4, 8, N_TOKEN - 1, N_TOKEN, N_TOKEN + 5, None]
)
def test_chunking_changes_nothing(chunk_size: int | None) -> None:
    """Sizes that do not divide the token count exercise the padded last block."""
    m, mask = _inputs()
    params = _params()
    reference = outer_product_mean(m, params, mask=mask)
    chunked = outer_product_mean(m, params, mask=mask, chunk_size=chunk_size)
    assert chunked.shape == (1, N_TOKEN, N_TOKEN, C_Z)
    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(reference), rtol=1e-6, atol=1e-6
    )


def test_padded_rows_do_not_leak_into_the_result() -> None:
    """The pad rows carry no mask, so they divide by ``eps`` alone and come out
    large. They are sliced off before the division, but a chunk size that leaves
    a full block of padding is the case where an off-by-one would show up."""
    m, mask = _inputs()
    params = _params()
    reference = outer_product_mean(m, params, mask=mask)
    # 12 rows over 23 tokens: one full block and one block that is half padding.
    chunked = outer_product_mean(m, params, mask=mask, chunk_size=12)
    assert np.isfinite(np.asarray(chunked)).all()
    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(reference), rtol=1e-6, atol=1e-6
    )
