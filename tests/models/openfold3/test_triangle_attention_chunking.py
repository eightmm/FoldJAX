"""Chunked triangle attention must be exactly the unchunked result.

Chunking here is a memory transformation, not an approximation: ``I`` is a batch
axis, so each row attends over ``J`` independently and splitting the rows cannot
change any output. That makes exact equality the right assertion -- anything less
would mean the wrong axis was sliced, which is easy to do silently because the pair
representation is square and the bias's ``I`` axis is the *query* axis.

**Two pins make that assertion mean anything, and neither was here before.**

``backend="xla"``. The shipped default is the fused cuEquivariance kernel, and
``triangle_attention`` selects it *before* it reads ``chunk_size``; cueq never
forms the score tensor, so it has nothing to block and drops the request
silently. Unpinned, every test in this file ran cueq against cueq and passed
whatever the chunked path did -- verified by raising inside
``_chunked_attention``, which left all twelve green.
``test_the_pins_are_load_bearing`` guards that now.

The port's matmul precision. ``predict`` runs under
``_MATMUL_PRECISION`` (upstream's ``torch.set_float32_matmul_precision("high")``),
scoped rather than global, so a test that does not enter that scope measures
TF32 instead of chunking. It is a 430x difference here: chunked against
unchunked is 5.135e-05 at the process default and 1.192e-07 under the port's
own setting.

Under both pins the residual is float32 accumulation order alone, measured
1.192e-07 on GPU across five fresh processes and 5.960e-08 on CPU. The
tolerance below is 1e-6, so roughly 8x headroom -- and a mis-sliced axis moves
these outputs by order 0.1, six decades above it, so the margin cannot hide the
error the docstring is worried about.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.inference import _MATMUL_PRECISION
from foldjax.models.openfold3.models.triangle_attention import triangle_attention

N, C, HEADS = 23, 8, 2

#: The chunked path exists only under the XLA backend; see the module docstring.
BACKEND = "xla"


@pytest.fixture(autouse=True)
def _production_precision():
    """Every comparison here runs at the precision ``predict`` runs at.

    Imported rather than spelled, so pinning it somewhere else does not quietly
    leave this file measuring something the model never does.
    """
    with jax.default_matmul_precision(_MATMUL_PRECISION):
        yield


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
    mask = jnp.asarray((generator.random((batch, n, n)) > 0.15).astype(np.float32))
    return x, mask


@pytest.mark.parametrize("chunk_size", [1, 4, 8, N - 1, N, N + 5])
def test_chunking_changes_nothing(chunk_size: int) -> None:
    """Including chunk sizes that do not divide the row count, and one larger than
    it -- the padding path is where a silent error would live."""
    params = _params()
    x, mask = _inputs()
    reference = triangle_attention(
        x, params, no_heads=HEADS, backend=BACKEND, mask=mask
    )
    chunked = triangle_attention(
        x, params, no_heads=HEADS, backend=BACKEND, mask=mask, chunk_size=chunk_size
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
        x, params, no_heads=HEADS, backend=BACKEND, mask=mask, starting=starting
    )
    chunked = triangle_attention(
        x,
        params,
        no_heads=HEADS,
        backend=BACKEND,
        mask=mask,
        starting=starting,
        chunk_size=5,
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
    reference = triangle_attention(x, params, no_heads=HEADS, backend=BACKEND)
    chunked = triangle_attention(
        x, params, no_heads=HEADS, backend=BACKEND, chunk_size=6
    )
    np.testing.assert_allclose(
        np.asarray(chunked, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_chunking_is_jit_compatible() -> None:
    """It uses lax.map and dynamic slices, so it has to trace.

    Both sides are compiled deliberately. This used to compare the *compiled*
    chunked result against the *eager* unchunked one, varying two things at
    once, and it was the only test here that failed -- red on GPU, green on CPU.
    That was the fused kernel differing between dispatch modes, nothing to do
    with chunking, and chasing the tolerance kept attention away from the fact
    that no test in this file reached the code it named.
    """
    params = _params(6)
    x, mask = _inputs(7)
    chunked = jax.jit(
        lambda a, m: triangle_attention(
            a, params, no_heads=HEADS, backend=BACKEND, mask=m, chunk_size=8
        )
    )
    plain = jax.jit(
        lambda a, m: triangle_attention(
            a, params, no_heads=HEADS, backend=BACKEND, mask=m
        )
    )
    np.testing.assert_allclose(
        np.asarray(chunked(x, mask), dtype=np.float64),
        np.asarray(plain(x, mask), dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_the_pins_are_load_bearing() -> None:
    """Without ``backend="xla"`` every comparison above is vacuous.

    Two independent facts make it so, and either one returning re-hides the
    chunked path: the shipped default is the fused kernel, and that kernel is
    chosen before ``chunk_size`` is read. Assert both, so this file fails loudly
    rather than passing on a comparison it has stopped making.
    """
    from foldjax.models.openfold3.models.triangle_attention import _default_backend

    assert _default_backend() == "cueq", (
        "the default backend is no longer cueq -- recheck whether BACKEND still "
        "has to be pinned for these tests to reach _chunked_attention"
    )

    params = _params(12)
    x, mask = _inputs(13)
    ignored = triangle_attention(
        x, params, no_heads=HEADS, backend="cueq", mask=mask, chunk_size=4
    )
    whole = triangle_attention(x, params, no_heads=HEADS, backend="cueq", mask=mask)
    np.testing.assert_array_equal(
        np.asarray(ignored),
        np.asarray(whole),
        err_msg="cueq now honours chunk_size; the pin above may be unnecessary",
    )


def test_a_fully_masked_row_does_not_produce_nan() -> None:
    """The padding path pads the mask bias with zeros for exactly this reason: a row
    masked everywhere makes softmax divide by zero."""
    params = _params(8)
    x, mask = _inputs(9)
    mask = mask.at[:, 0, :].set(0.0)
    out = triangle_attention(
        x, params, no_heads=HEADS, backend=BACKEND, mask=mask, chunk_size=4
    )
    assert bool(np.isfinite(np.asarray(out)).all())


def test_extra_leading_batch_dimensions_survive() -> None:
    params = _params(10)
    x, mask = _inputs(11, batch=3)
    reference = triangle_attention(
        x, params, no_heads=HEADS, backend=BACKEND, mask=mask
    )
    chunked = triangle_attention(
        x, params, no_heads=HEADS, backend=BACKEND, mask=mask, chunk_size=7
    )
    assert chunked.shape == reference.shape == (3, N, N, C)
    np.testing.assert_allclose(
        np.asarray(chunked, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )
