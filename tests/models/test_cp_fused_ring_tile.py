"""The statistics merge the fused 2-D ring tile is built on.

The fused tile itself is a Pallas/Triton kernel and cannot run here. What can
run -- and what carries the risk -- is the arithmetic around it: a merge that
folds independently normalised key tiles into one attention, an empty-row
contract that a fused kernel reports differently from the reference, and a
rotation schedule that a one-pass ring does once where the shipped ring does it
twice.

So the gates below drive the same merge ring with the portable tile
(:data:`~foldjax.models._cp_attention.RING_TILE_BODIES`, ``xla_merge``) and
compare it against a dense softmax. What stays unproven until the GPU
experiment runs is only that tokamax's residuals *are* the tile statistics this
contract asks for, which is a property of the kernel and not of this code.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._cp_attention import (
    RING_TILE_BODIES,
    RING_TILE_KERNELS,
    RING_TOKAMAX_IMPLEMENTATION,
    _tile_terms,
    merge_softmax_statistics,
    resolve_ring_tile_kernel,
    ring_tile_kernel,
    ring_tile_kernel_available,
    ring_tile_kernel_scope,
    tile_attention_tokamax,
    tile_attention_xla,
)
from foldjax.models._tokamax_attention import tokamax_available
from tests.models.cp_probe_env import inherited_environment

ROWS, HEADS, QUERIES, KEYS, DIM = 3, 2, 5, 4, 6
SIDE = 3


def _tiles(seed: int, *, empty_row: bool = False):
    """One block's query rows and ``SIDE`` key tiles, as the ring holds them."""

    rng = np.random.default_rng(seed)

    def arr(*shape, scale=0.5):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    query = arr(ROWS, HEADS, QUERIES, DIM)
    keys = [arr(ROWS, HEADS, KEYS, DIM) for _ in range(SIDE)]
    values = [arr(ROWS, HEADS, KEYS, DIM) for _ in range(SIDE)]
    # Wide enough that the tiles' own maxima are far apart, which is what the
    # rescaling in the merge is for; a benign near-uniform bias would pass a
    # merge that never rescaled at all.
    biases = [arr(1, HEADS, QUERIES, KEYS, scale=9.0) for _ in range(SIDE)]
    masks = []
    for index in range(SIDE):
        keep = rng.random((ROWS, KEYS)) > 0.25
        keep[:, 0] = True
        if empty_row:
            # Row 0 loses every key in every tile: the globally empty query
            # row, which is the case padding produces and the one a fused
            # kernel reports as a finite maximum over absent keys.
            keep[0, :] = False
        masks.append(
            jnp.where(
                jnp.asarray(keep)[:, None, None, :],
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(-jnp.inf, dtype=jnp.float32),
            )
        )
    return query, keys, values, biases, masks


def _merge(query, keys, values, biases, masks):
    """Fold every tile through the production merge and normalise."""

    output = jnp.zeros(query.shape, dtype=jnp.float32)
    output_correction = jnp.zeros_like(output)
    normalizer = jnp.zeros(query.shape[:-1] + (1,), dtype=jnp.float32)
    normalizer_correction = jnp.zeros_like(normalizer)
    maximum = jnp.full_like(normalizer, -jnp.inf)
    for key, value, bias, mask in zip(keys, values, biases, masks, strict=True):
        tile = tile_attention_xla(query, key, value, bias, mask)
        (
            output,
            output_correction,
            normalizer,
            normalizer_correction,
            maximum,
        ) = merge_softmax_statistics(
            output,
            output_correction,
            normalizer,
            normalizer_correction,
            maximum,
            *tile,
        )
    output = output + output_correction
    normalizer = normalizer + normalizer_correction
    return jnp.where(normalizer > 0, output / jnp.maximum(normalizer, 1e-45), 0.0)


def _dense(query, keys, values, biases, masks):
    key = jnp.concatenate(list(keys), axis=-2)
    value = jnp.concatenate(list(values), axis=-2)
    bias = jnp.concatenate(list(biases), axis=-1)
    mask = jnp.concatenate(list(masks), axis=-1)
    scores = jnp.einsum("rhqd,rhkd->rhqk", query, key) + bias + mask
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
    return jnp.einsum("rhqk,rhkd->rhqd", probabilities, value)


def test_the_portable_tile_is_the_ring_s_own_tile_arithmetic() -> None:
    """Bitwise, not merely close.

    The tile callable is the shipped ring's ``_tile_terms`` evaluated against
    the tile's own maximum instead of a global one. Asserting that identity
    rather than a tolerance is what makes the merge's error attributable to the
    merge: if this drifts, the fused arm is being compared against something
    that is no longer the ring's tile.
    """

    query, keys, values, biases, masks = _tiles(20260917)
    for key, value, bias, mask in zip(keys, values, biases, masks, strict=True):
        output, maximum, normalizer = tile_attention_xla(query, key, value, bias, mask)
        expected_output, expected_normalizer = _tile_terms(
            query,
            key,
            value,
            bias,
            mask,
            maximum,
            None,
        )
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(normalizer, expected_normalizer)


def test_the_merge_matches_a_dense_softmax() -> None:
    """Three independently normalised tiles reduce to one attention."""

    query, keys, values, biases, masks = _tiles(20260918)
    merged = jax.device_get(_merge(query, keys, values, biases, masks))
    dense = jax.device_get(_dense(query, keys, values, biases, masks))
    np.testing.assert_allclose(dense, merged, atol=1e-6, rtol=1e-6)


def test_a_query_row_with_no_valid_key_merges_to_zero() -> None:
    """The empty row, which dense softmax cannot express at all.

    Every tile hands the merge ``maximum = -inf`` and a zero denominator, and
    the merge has to keep both rather than divide. Dense softmax over an
    all-``-inf`` row is ``nan``, so the reference here is the ring's own
    contract; the rows that do have keys are still checked against dense.
    """

    query, keys, values, biases, masks = _tiles(20260919, empty_row=True)
    merged = jax.device_get(_merge(query, keys, values, biases, masks))
    assert np.all(merged[0] == 0.0)
    assert np.all(np.isfinite(merged))
    dense = jax.device_get(_dense(query, keys, values, biases, masks))
    np.testing.assert_allclose(dense[1:], merged[1:], atol=1e-6, rtol=1e-6)


def test_the_merge_carries_the_low_bits_a_plain_sum_drops() -> None:
    """The compensated accumulation path, on terms chosen so it matters.

    One large tile term and many tiny ones: in fp32 each tiny term is half an
    ulp of the running total, rounds to nothing on its own, and the plain sum
    returns the large term unchanged however many of them arrive. Their sum is
    four ulps, so the compensated total does represent it. The maxima are held
    equal so this measures the accumulation rather than the rescaling.
    """

    shape = (2, 1)
    tail = 256
    large = jnp.full(shape, 1.0, dtype=jnp.float32)
    tiny = jnp.full(shape, 2.0**-30, dtype=jnp.float32)
    terms = [large] + [tiny] * tail
    maxima = jnp.zeros(shape, dtype=jnp.float32)

    output = jnp.zeros(shape, dtype=jnp.float32)
    output_correction = jnp.zeros_like(output)
    normalizer = jnp.zeros(shape, dtype=jnp.float32)
    normalizer_correction = jnp.zeros_like(normalizer)
    maximum = jnp.full(shape, -jnp.inf, dtype=jnp.float32)
    plain = jnp.zeros(shape, dtype=jnp.float32)
    for term in terms:
        plain = plain + term
        (
            output,
            output_correction,
            normalizer,
            normalizer_correction,
            maximum,
        ) = merge_softmax_statistics(
            output,
            output_correction,
            normalizer,
            normalizer_correction,
            maximum,
            term,
            maxima,
            term,
        )
    exact = float(np.float64(1.0) + tail * np.float64(2.0**-30))
    compensated = float(jax.device_get(output + output_correction)[0, 0])
    assert float(jax.device_get(plain)[0, 0]) == 1.0
    assert abs(compensated - exact) < abs(float(jax.device_get(plain)[0, 0]) - exact)
    np.testing.assert_allclose(compensated, exact, rtol=0, atol=2.0**-40)


def test_the_option_vocabulary_is_a_subset_of_the_bodies() -> None:
    """``xla_merge`` is a body and not a request, and the code says which."""

    assert set(RING_TILE_KERNELS) < set(RING_TILE_BODIES)
    assert "xla_merge" not in RING_TILE_KERNELS


@pytest.mark.parametrize("kernel", ["", "triton", "cueq", "xla_merge"])
def test_a_kernel_outside_the_vocabulary_is_refused(kernel: str) -> None:
    with pytest.raises(ValueError, match="triangle_attention_ring_kernel"):
        resolve_ring_tile_kernel(kernel)


def test_the_fused_tile_is_refused_rather_than_downgraded_off_gpu() -> None:
    """No silent fallback: the whole point of the option is which kernel ran."""

    if jax.default_backend() == "gpu":
        pytest.skip("the refusal under test is the non-GPU one")
    with pytest.raises(RuntimeError, match="GPU backend"):
        resolve_ring_tile_kernel("tokamax")


def test_the_availability_probe_answers_the_refusals_two_halves() -> None:
    """One boolean for the caller that decides, two messages for the one that
    validates -- and they must be the same question on this host.

    `backends/boltz2._realised_ring_tile_kernel` resolves an omitted option
    against the probe; `resolve_ring_tile_kernel` refuses an explicit request
    against the two halves. A probe that answered yes where the refusal fires
    would let an omitted option resolve to a body the very next call rejects.
    """

    available = ring_tile_kernel_available()
    assert available == (tokamax_available() and jax.default_backend() == "gpu")
    if available:
        assert resolve_ring_tile_kernel("tokamax") == "tokamax"
    else:
        with pytest.raises(RuntimeError):
            resolve_ring_tile_kernel("tokamax")
    # The portable body never consults it, on either host.
    assert resolve_ring_tile_kernel(None) == "xla"
    assert resolve_ring_tile_kernel("xla") == "xla"


def test_the_scope_defaults_to_the_shipped_kernel() -> None:
    assert ring_tile_kernel() == "xla"
    with ring_tile_kernel_scope("tokamax") as name:
        assert name == "tokamax"
        assert ring_tile_kernel() == "tokamax"
    assert ring_tile_kernel() == "xla"
    with ring_tile_kernel_scope(None) as name:
        assert name == "xla"
    with pytest.raises(ValueError, match="triangle_attention_ring_kernel"):
        with ring_tile_kernel_scope("triton"):
            pass


_MERGE_RING_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import ring_triangle_attention_2d

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    assert jax.device_count() == DEVICES, jax.devices()

    rng = np.random.default_rng(20260920)
    BATCH, HEADS, DIM = 2, 3, 5
    N = int(os.environ["FOLDJAX_CP_PROBE_TOKENS"])

    def arr(*shape, scale=0.4):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    q = arr(BATCH, N, HEADS, N, DIM)
    k = arr(BATCH, N, HEADS, N, DIM)
    v = arr(BATCH, N, HEADS, N, DIM)
    bias = arr(BATCH, 1, HEADS, N, N, scale=7.0)
    keep = rng.random((BATCH, N, N)) > 0.2
    keep[..., 0] = True
    mask = jnp.where(
        jnp.asarray(keep)[:, :, None, None, :],
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(-1.0e9, dtype=jnp.float32),
    )

    def dense(q_in, k_in, v_in, b_in, m_in):
        scores = jnp.einsum("...hqd,...hkd->...hqk", q_in, k_in)
        scores = scores + b_in + m_in
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        return jnp.einsum("...hqk,...hkd->...hqd", probs, v_in)

    reference = jax.device_get(jax.jit(dense)(q, k, v, bias, mask))

    def run(kernel, block):
        def ring(q_in, k_in, v_in, b_in, m_in):
            return ring_triangle_attention_2d(
                q_in, k_in, v_in, b_in, m_in, tile_kernel=kernel, q_block=block
            )

        return jax.device_get(jax.jit(ring)(q, k, v, bias, mask))

    with context_parallel(DEVICES, layout="2d"):
        shipped = run("xla", None)
        # Unblocked and blocked: the merge carry crosses the `lax.scan`
        # boundary only in the second, which is the ring's real program.
        merged = run("xla_merge", None)
        merged_blocked = run("xla_merge", 1)

    np.testing.assert_allclose(reference, shipped, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(reference, merged, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(reference, merged_blocked, atol=3e-5, rtol=3e-5)
    print("MERGE_RING_OK")
    """
)


def _run(source: str, *, devices: int, tokens: int) -> str:
    env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "FOLDJAX_CP_PROBE_DEVICES": str(devices),
        "FOLDJAX_CP_PROBE_TOKENS": str(tokens),
        **inherited_environment(),
    }
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


@pytest.mark.parametrize(("devices", "tokens"), [(4, 8), (9, 9)])
def test_the_merge_ring_matches_dense_on_a_real_mesh(devices: int, tokens: int) -> None:
    """The merge ring's own rotation schedule, not just its arithmetic.

    The one-pass body rotates V from the first step where the shipped body
    leaves it at its initial owner for a whole cycle, so a schedule that is
    right for the two-pass ring can be wrong here. 3x3 is included because a
    2x2 mesh cannot distinguish a hop from its inverse
    (``cannon-sign-errors-need-side-3``).
    """

    assert "MERGE_RING_OK" in _run(_MERGE_RING_PROBE, devices=devices, tokens=tokens)


_PALLAS_TILE_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import pallas as pl

    import foldjax.models._cp_attention as cp_attention
    from foldjax.models._cp import context_parallel

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    assert jax.device_count() == DEVICES, jax.devices()

    rng = np.random.default_rng(20260926)
    BATCH, HEADS, DIM = 1, 2, 4
    N = int(os.environ["FOLDJAX_CP_PROBE_TOKENS"])

    def copy_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    def through_pallas(x):
        # The narrowest Pallas kernel there is: the arithmetic stays the
        # portable tile's and only the output *declaration* is the fused
        # tile's -- a `jax.ShapeDtypeStruct` that names no varying axes.
        return pl.pallas_call(
            copy_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            interpret=True,
        )(x)

    def pallas_resolve(tile_kernel, precision):
        def tile(*operands):
            terms = cp_attention.tile_attention_xla(*operands, precision=precision)
            return tuple(through_pallas(term) for term in terms)

        return tile

    def arr(*shape, scale=0.4):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    q = arr(BATCH, N, HEADS, N, DIM)
    k = arr(BATCH, N, HEADS, N, DIM)
    v = arr(BATCH, N, HEADS, N, DIM)
    pair = arr(BATCH, N, N, 2 * HEADS * DIM)
    weight = arr(2 * HEADS * DIM, 3 * HEADS * DIM)
    bias = arr(BATCH, 1, HEADS, N, N, scale=7.0)
    keep = rng.random((BATCH, N, N)) > 0.2
    keep[..., 0] = True
    mask = jnp.where(
        jnp.asarray(keep)[:, :, None, None, :],
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(-1.0e9, dtype=jnp.float32),
    )

    def dense(q_in, k_in, v_in, b_in, m_in):
        scores = jnp.einsum("...hqd,...hkd->...hqk", q_in, k_in)
        scores = scores + b_in + m_in
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        return jnp.einsum("...hqk,...hkd->...hqd", probs, v_in)

    def split_heads(array):
        array = array.reshape(array.shape[:-1] + (HEADS, DIM))
        return jnp.swapaxes(array, -2, -3)

    def project(params, rows):
        projected = jnp.matmul(rows, params)
        query, key, value = jnp.split(projected, 3, axis=-1)
        query = split_heads(query) / jnp.sqrt(jnp.asarray(DIM, dtype=query.dtype))
        return query, split_heads(key), split_heads(value), None

    reference = jax.device_get(jax.jit(dense)(q, k, v, bias, mask))
    pair_q, pair_k, pair_v, _ = project(weight, pair)
    pair_reference = jax.device_get(
        jax.jit(dense)(pair_q, pair_k, pair_v, bias, mask)
    )

    def run(block):
        def ring(q_in, k_in, v_in, b_in, m_in):
            return cp_attention.ring_triangle_attention_2d(
                q_in, k_in, v_in, b_in, m_in, tile_kernel="xla_merge", q_block=block
            )

        return jax.device_get(jax.jit(ring)(q, k, v, bias, mask))

    def run_from_pair(block):
        def ring(p_in, b_in, m_in, w_in):
            return cp_attention.ring_triangle_attention_2d_from_pair(
                p_in,
                b_in,
                m_in,
                w_in,
                project=project,
                tile_kernel="xla_merge",
                q_block=block,
            )

        return jax.device_get(jax.jit(ring)(pair, bias, mask, weight))

    cp_attention._resolve_tile_attention = pallas_resolve
    with context_parallel(DEVICES, layout="2d"):
        # Blocked as well as unblocked: the merge carry crosses a `lax.scan`
        # boundary only in the second, and the initial-carry rule is one of
        # the things the varying-axis check enforces.
        np.testing.assert_allclose(reference, run(None), atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(reference, run(1), atol=3e-5, rtol=3e-5)
        print("PALLAS_TILE_OK")

        # Both entry points, because each builds its own `shard_map` and the
        # GPU failure came through this one.
        for block in (None, 1):
            np.testing.assert_allclose(
                pair_reference, run_from_pair(block), atol=3e-5, rtol=3e-5
            )
        print("PALLAS_TILE_FROM_PAIR_OK")

        # The tripwire. Without it a passing probe could mean the Pallas
        # declaration is accepted everywhere and the arm proves nothing.
        spec = cp_attention._two_axis_spec(q.ndim, -4, -2)
        checked = jax.shard_map(
            through_pallas,
            mesh=cp_attention.cp_mesh(),
            in_specs=(spec,),
            out_specs=spec,
        )
        try:
            jax.jit(checked).lower(q)
        except ValueError as error:
            assert "manual_axis_type" in str(error), error
            print("VMA_CHECK_REFUSES")

    print("PROBE_DONE")
    """
)


def test_the_fused_ring_runs_a_tile_the_varying_axis_check_refuses() -> None:
    """A Pallas tile inside the fused arm's `shard_map`, which cannot check it.

    A Pallas kernel declares its outputs as `jax.ShapeDtypeStruct`s carrying no
    `manual_axis_type`, and a checked `shard_map` requires one of every output
    produced inside it: on four GPUs the tokamax arm raised out of
    `pl.pallas_call` before computing anything. Here a one-line copy kernel
    stands in for the Triton one -- same arithmetic as the portable tile, the
    same output declaration as the fused one -- so the arm the GPU refused is
    the arm this runs, through both ring entry points, and the tripwire shows
    that declaration still being refused by a `shard_map` the change did not
    touch.
    """

    output = _run(_PALLAS_TILE_PROBE, devices=4, tokens=8)
    assert "PALLAS_TILE_OK" in output, output
    assert "PALLAS_TILE_FROM_PAIR_OK" in output, output
    assert "VMA_CHECK_REFUSES" in output, output
    assert "PROBE_DONE" in output, output


# ---------------------------------------------------------------------------
# The tokamax side of the contract, as far as a CPU can take it
# ---------------------------------------------------------------------------
#
# The fused tile is a Triton kernel, but the *contract* it is trusted for is
# not Triton's: `normalize_output=False` plus `return_residuals=True` returning
# `(softmax maximum, softmax denominator)` is `base.DotProductAttention`'s, and
# every registered implementation goes through the same `__call__`. So the
# claim "tokamax's residuals are the tile statistics the merge needs" is
# checkable here, against tokamax's own XLA implementation, and what stays
# unchecked until the GPU experiment runs is only whether the Triton kernel
# agrees with its own base class.

requires_tokamax = pytest.mark.skipif(
    not tokamax_available(),
    reason="tokamax did not import in this process",
)


@requires_tokamax
def test_the_pinned_implementation_takes_the_residual_keywords() -> None:
    """A private path, so the assertion is that it still exists and still takes.

    `tokamax.dot_product_attention` does not expose `normalize_output` or
    `return_residuals`, so the option addresses the implementation object
    under `tokamax._src`. If that moves, this fails here rather than at trace
    time on four allocated cards.
    """

    import inspect

    from tokamax._src.ops.attention.api import IMPLEMENTATIONS

    assert RING_TOKAMAX_IMPLEMENTATION in IMPLEMENTATIONS, sorted(IMPLEMENTATIONS)
    for name in ("xla", RING_TOKAMAX_IMPLEMENTATION):
        signature = inspect.signature(type(IMPLEMENTATIONS[name])._fwd)
        for keyword in ("normalize_output", "return_residuals"):
            assert keyword in signature.parameters, (name, keyword)


@requires_tokamax
def test_tokamax_residuals_are_the_tile_statistics_the_merge_needs() -> None:
    """The load-bearing assumption, proved on the implementation that runs here.

    `normalize_output=False` must leave the numerator unnormalised -- an
    implementation that ignored the flag would return a normalised output and
    the merge would be wrong by the denominator, silently and everywhere.
    """

    from tokamax._src.ops.attention.api import IMPLEMENTATIONS

    rng = np.random.default_rng(20260921)
    rows, heads, queries, keys, dim = 2, 2, 4, 6, 8

    def arr(*shape, scale=0.5):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    query = arr(rows, queries, heads, dim)
    key = arr(rows, keys, heads, dim)
    value = arr(rows, keys, heads, dim)
    bias = arr(1, heads, queries, keys, scale=6.0)

    output, (maximum, normalizer) = IMPLEMENTATIONS["xla"](
        query,
        key,
        value,
        bias=bias,
        logits_scale=1.0,
        logits_dtype=jnp.float32,
        normalize_output=False,
        return_residuals=True,
    )
    logits = jnp.einsum("rqhd,rkhd->rhqk", query, key) + bias
    expected_maximum = jnp.max(logits, axis=-1)
    unnormalized = jnp.exp(logits - expected_maximum[..., None])
    expected_normalizer = jnp.sum(unnormalized, axis=-1)
    expected_output = jnp.einsum("rhqk,rkhd->rqhd", unnormalized, value)

    np.testing.assert_allclose(maximum, expected_maximum, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        normalizer, expected_normalizer, atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(output, expected_output, atol=1e-4, rtol=1e-5)
    # And the flag is honoured rather than accepted: a normalised output would
    # be this divided by the denominator, which is a different array wherever
    # the denominator is not one.
    normalised = expected_output / expected_normalizer.mT[..., None]
    assert not np.allclose(output, normalised, atol=1e-3)


@requires_tokamax
def test_the_fused_tile_wiring_reproduces_the_portable_tile() -> None:
    """Every conversion in the fused callable, run on the CPU implementation.

    The layout swaps, the additive mask becoming a boolean one, the residual
    reshape and the empty-row forcing are all shared with the Triton path and
    are where a wiring bug would live. Driving them through tokamax's XLA
    implementation puts them under a CPU gate; only the kernel itself is left
    to the GPU experiment.
    """

    query, keys, values, biases, masks = _tiles(20260922, empty_row=True)
    for key, value, bias, mask in zip(keys, values, biases, masks, strict=True):
        reference = tile_attention_xla(
            query, key, value, bias, mask, precision=jax.lax.Precision.HIGHEST
        )
        got = tile_attention_tokamax(
            query,
            key,
            value,
            bias,
            mask,
            # The precision the Boltz-2 layer pins. Passed so tokamax's
            # canonicalisation of the kwarg runs under a CPU gate rather than
            # first running on four allocated cards.
            precision=jax.lax.Precision.HIGHEST,
            implementation="xla",
        )
        for name, left, right in zip(
            ("output", "maximum", "normalizer"), reference, got, strict=True
        ):
            left = jax.device_get(left)
            right = jax.device_get(right)
            empty = np.isneginf(left)
            # The empty rows must agree exactly: both sides force them, and a
            # tolerance there would hide the case the forcing exists for.
            np.testing.assert_array_equal(
                np.isneginf(right), empty, err_msg=f"{name}: empty rows differ"
            )
            finite = ~np.isneginf(left) & ~np.isneginf(right)
            np.testing.assert_allclose(
                right[finite],
                left[finite],
                atol=1e-4,
                rtol=1e-4,
                err_msg=name,
            )


@requires_tokamax
def test_an_unregistered_implementation_is_refused_not_substituted() -> None:
    query, keys, values, biases, masks = _tiles(20260923)
    with pytest.raises(ValueError, match="attention implementation"):
        tile_attention_tokamax(
            query,
            keys[0],
            values[0],
            biases[0],
            masks[0],
            implementation="flash",
        )


@requires_tokamax
def test_the_mask_biased_row_is_where_the_two_tiles_diverge() -> None:
    """The one documented semantic difference, asserted rather than described.

    A real run's mask bias is finite (`-1e9`), not `-inf`: an absent key is
    `-inf` only where padding put it. The portable tile therefore reduces a
    row whose keys are all merely mask-biased over exactly those keys, as the
    serial path does, while the fused tile routes them through tokamax's
    boolean mask, finds no valid key, and returns zeros.

    Both are defensible and the rows are the ones the model masks away
    downstream, so this pins the divergence instead of removing it -- and
    catches the day it stops being confined to those rows.
    """

    query, keys, values, biases, masks = _tiles(20260924)
    # The ring's real mask bias: finite, which is what makes the two tiles
    # differ at all. `_tiles` uses -inf, where they agree.
    finite_masks = [
        jnp.where(jnp.isneginf(mask), jnp.asarray(-1.0e9, jnp.float32), mask)
        for mask in masks
    ]
    # Row 0 keeps no key in tile 0, by mask bias alone.
    blocked = finite_masks[0].at[0].set(-1.0e9)
    portable = tile_attention_xla(query, keys[0], values[0], biases[0], blocked)
    fused = tile_attention_tokamax(
        query, keys[0], values[0], biases[0], blocked, implementation="xla"
    )
    assert np.all(jax.device_get(fused[0])[0] == 0.0)
    assert np.all(np.isneginf(jax.device_get(fused[1])[0]))
    assert np.all(jax.device_get(fused[2])[0] == 0.0)
    # The portable tile does not: its maximum is finite and its denominator is
    # positive, because -1e9 is a number.
    assert np.all(np.isfinite(jax.device_get(portable[1])[0]))
    assert np.all(jax.device_get(portable[2])[0] > 0.0)
    # And every other row still agrees, which is what confines the divergence.
    for name, left, right in zip(
        ("output", "maximum", "normalizer"), portable, fused, strict=True
    ):
        np.testing.assert_allclose(
            jax.device_get(right)[1:],
            jax.device_get(left)[1:],
            atol=1e-4,
            rtol=1e-4,
            err_msg=name,
        )


@requires_tokamax
def test_the_fused_tile_keeps_the_numerator_in_f32_under_bf16_operands() -> None:
    """The released `compute_dtype` is bfloat16, so this is the deployed case.

    Both tokamax implementations end their forward with
    `out.astype(q.dtype)`, so bf16 operands would round the *unnormalised*
    numerator -- a sum whose magnitude is the tile's denominator, up to a
    thousand keys -- to bf16 before the merge saw it, and a cast on the way
    out could not undo it. The tile casts before the call for that reason, and
    this is what would notice if that cast were removed: the outputs stay f32
    and stay on top of the portable tile, which has always been f32 inside.
    """

    query, keys, values, biases, masks = _tiles(20260925)
    cast = [leaf.astype(jnp.bfloat16) for leaf in (query, keys[0], values[0])]
    reference = tile_attention_xla(
        *cast, biases[0], masks[0], precision=jax.lax.Precision.HIGHEST
    )
    got = tile_attention_tokamax(
        *cast,
        biases[0],
        masks[0],
        precision=jax.lax.Precision.HIGHEST,
        implementation="xla",
    )
    for name, left, right in zip(
        ("output", "maximum", "normalizer"), reference, got, strict=True
    ):
        assert right.dtype == jnp.float32, (name, right.dtype)
        assert left.dtype == jnp.float32, (name, left.dtype)
        np.testing.assert_allclose(
            jax.device_get(right),
            jax.device_get(left),
            atol=1e-3,
            rtol=1e-3,
            err_msg=name,
        )
