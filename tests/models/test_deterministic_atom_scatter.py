"""Atom-to-token sums without scatter-add.

GPU scatter-add accumulates with atomics in no fixed order, so the Boltz-2,
Protenix and OpenDDE samplers aggregate through ``segment_sum`` instead. The
reference here is the scatter-add each site used before, which on CPU is
sequential; the gather loop adds in the same atom order, so the two agree
bit-for-bit on CPU, in the input dtype as the scatter-add accumulated.
Boltz-2's serial mean alone now accumulates bfloat16 in float32, as upstream's
`bmm` and the context-parallel path do, so its bfloat16 reference is the
scatter-add on the float32 upcast. The lowered graphs must carry no scatter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models import _segment as segment_module
from foldjax.models._cp_atom import scatter_atoms_to_tokens_mean_cp
from foldjax.models._segment import segment_plan, segment_sum
from foldjax.models.boltz2.models.diffusion.atom import (
    _one_hot_index,
    atom_to_token_plan,
    scatter_atoms_to_tokens_mean,
)
from foldjax.models.protenix.models.diffusion.atom import aggregate_atom_to_token

N_TOKEN = 37
N_ATOM = 211
DTYPES = [jnp.float32, jnp.bfloat16]


def _owners(rng: np.random.Generator, n_atom: int = N_ATOM) -> np.ndarray:
    """Out-of-order owners plus wrapped (-1, -2) and dropped (-40, >= N) ones."""

    owners = rng.integers(0, N_TOKEN, n_atom)
    owners[:5] = [-1, -2, -40, N_TOKEN, N_TOKEN + 3]
    owners[5:30] = 4  # one owner far larger than the rest
    return owners.astype(np.int32)


def _scatter_sum(values, owners, n_segments, axis):
    shape = list(values.shape)
    shape[axis] = n_segments
    index = (slice(None),) * (axis % values.ndim) + (jnp.asarray(owners),)
    return jnp.zeros(shape, values.dtype).at[index].add(values)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [0, -2, -1])
def test_segment_sum_matches_scatter_add(dtype, axis) -> None:
    rng = np.random.default_rng(0)
    owners = _owners(rng)
    shape = {0: (N_ATOM, 5), -2: (3, N_ATOM, 5), -1: (3, N_ATOM)}[axis]
    values = jnp.asarray(rng.normal(size=shape), dtype)

    expected = _scatter_sum(values, owners, N_TOKEN, axis)
    actual = jax.jit(lambda v: segment_sum(v, owners, N_TOKEN, axis=axis))(values)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    )


def test_segment_sum_gradient_and_vmap_match_scatter_add() -> None:
    rng = np.random.default_rng(1)
    owners = np.stack([_owners(rng) for _ in range(3)])
    values = jnp.asarray(rng.normal(size=(3, N_ATOM, 4)), jnp.float32)

    def loss(fn, v):
        return jnp.sum(jnp.sin(jax.vmap(fn)(owners, v)))

    new = jax.jit(jax.grad(lambda v: loss(lambda o, x: segment_sum(x, o, N_TOKEN), v)))
    old = jax.grad(lambda v: loss(lambda o, x: _scatter_sum(x, o, N_TOKEN, 0), v))
    np.testing.assert_allclose(new(values), old(values), rtol=1e-6, atol=1e-6)


def test_boltz2_mean_accumulates_bfloat16_in_float32() -> None:
    values = jnp.asarray([1.0] + [0.001] * 22, jnp.bfloat16)[None, :, None]
    index = (jnp.zeros((1, 23), jnp.int32), jnp.ones((1, 23), bool))

    mean = scatter_atoms_to_tokens_mean(None, values, index=index, num_tokens=1)

    # Re-rounding to bfloat16 after every atom would sum to exactly 1.0.
    assert mean.dtype == jnp.bfloat16
    assert float(mean[0, 0, 0]) == float(jnp.bfloat16(1.022 / (23 + 1e-6)))
    assert float(mean[0, 0, 0]) != float(jnp.bfloat16(1.0 / 23))


def test_segment_sum_rejects_mismatched_owners() -> None:
    with pytest.raises(ValueError, match="1-D"):
        segment_sum(jnp.ones((4, 2)), jnp.zeros((2, 4), jnp.int32), 3)


def _old_boltz2_mean(token_idx, valid, values, eps=1e-6):
    def one(idx, ok, v):
        v = v * ok[:, None].astype(v.dtype)
        acc = jnp.promote_types(v.dtype, jnp.float32)
        summed = jnp.zeros((N_TOKEN, v.shape[-1]), acc).at[idx].add(v.astype(acc))
        counts = jnp.zeros((N_TOKEN,), acc).at[idx].add(ok.astype(acc))
        return (summed / (counts[:, None] + eps)).astype(v.dtype)

    return jax.vmap(one)(token_idx, valid, values)


def _old_cp_mean(token_idx, valid, values, eps=1e-6):
    def one(idx, ok, v):
        f = v.astype(jnp.float32) * ok[:, None].astype(jnp.float32)
        sums = jnp.zeros((N_TOKEN, v.shape[-1]), jnp.float32).at[idx].add(f)
        counts = jnp.zeros((N_TOKEN,), jnp.float32).at[idx].add(ok.astype(jnp.float32))
        return (sums / (counts[:, None] + eps)).astype(v.dtype)

    return jax.vmap(one)(token_idx, valid, values)


def _boltz_inputs(dtype):
    rng = np.random.default_rng(2)
    token_idx = np.stack([_owners(rng) for _ in range(2)])
    valid = rng.random(token_idx.shape) > 0.2
    # Compact ownership marks padded atoms -1 and leaves their values live.
    token_idx = np.where(valid, token_idx, -1).astype(np.int32)
    values = jnp.asarray(rng.normal(size=(2, N_ATOM, 6)), dtype)
    return jnp.asarray(token_idx), jnp.asarray(valid), values


@pytest.mark.parametrize("dtype", DTYPES)
def test_boltz2_scatter_mean_matches_previous_scatter(dtype) -> None:
    token_idx, valid, values = _boltz_inputs(dtype)
    actual = jax.jit(
        lambda v: scatter_atoms_to_tokens_mean(
            None, v, index=(token_idx, valid), num_tokens=N_TOKEN
        )
    )(values)
    expected = _old_boltz2_mean(token_idx, valid, values)
    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_serial_cp_scatter_mean_matches_previous_scatter(dtype) -> None:
    token_idx, valid, values = _boltz_inputs(dtype)
    actual = jax.jit(
        lambda v: scatter_atoms_to_tokens_mean_cp(
            v, token_idx, valid, num_tokens=N_TOKEN
        )
    )(values)
    expected = _old_cp_mean(token_idx, valid, values)
    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("masked", [False, True])
def test_protenix_aggregate_matches_previous_scatter(dtype, reduce, masked) -> None:
    rng = np.random.default_rng(3)
    owners = jnp.asarray(_owners(rng))
    values = jnp.asarray(rng.normal(size=(2, N_ATOM, 6)), dtype)
    mask = jnp.asarray(rng.random(N_ATOM) > 0.2) if masked else None

    actual = jax.jit(
        lambda v: aggregate_atom_to_token(
            v, owners, n_token=N_TOKEN, reduce=reduce, atom_mask=mask
        )
    )(values)

    weights = None if mask is None else mask.astype(dtype)
    weighted = values if weights is None else values * weights[:, None]
    expected = jnp.zeros((2, N_TOKEN, 6), dtype).at[..., owners, :].add(weighted)
    if reduce == "mean":
        counts = (
            jnp.zeros((N_TOKEN,), dtype)
            .at[owners]
            .add(1.0 if weights is None else weights)
        )
        expected = expected / jnp.maximum(counts[..., None], 1.0)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    )


def test_masked_atoms_never_reach_the_token_their_index_names() -> None:
    """Padding gives masked atoms a real owner; their values must not matter.

    Protenix pads atom ownership with token 0 when the token axis is already
    on a bucket, and an invalid Boltz-2 index (-1 here) would wrap to the last
    token. Non-finite garbage there would survive a multiply-by-zero mask.
    """
    rng = np.random.default_rng(4)
    owners = _owners(rng)
    mask = rng.random(N_ATOM) > 0.3
    values = rng.normal(size=(2, N_ATOM, 6)).astype(np.float32)
    garbage = values.copy()
    garbage[:, ~mask] = np.nan
    dummy = jnp.asarray(np.where(mask, owners, 0))

    def protenix(v, reduce):
        return aggregate_atom_to_token(
            jnp.asarray(v), dummy, n_token=N_TOKEN, reduce=reduce, atom_mask=mask
        )

    for reduce in ("sum", "mean"):
        np.testing.assert_array_equal(
            protenix(garbage, reduce),
            protenix(np.where(mask[:, None], values, 0), reduce),
        )

    token_idx = jnp.asarray(np.stack([np.where(mask, owners, -1)] * 2))
    valid = jnp.asarray(np.stack([mask] * 2))
    actual = scatter_atoms_to_tokens_mean(
        None, jnp.asarray(garbage), index=(token_idx, valid), num_tokens=N_TOKEN
    )
    expected = _old_boltz2_mean(token_idx, valid, jnp.asarray(values))
    np.testing.assert_array_equal(actual, expected)
    cp = scatter_atoms_to_tokens_mean_cp(
        jnp.asarray(garbage), token_idx, valid, num_tokens=N_TOKEN
    )
    np.testing.assert_array_equal(cp, _old_cp_mean(token_idx, valid, values))


def test_padded_atoms_do_not_lengthen_the_loop(monkeypatch) -> None:
    """A padded run sums as fast as its real atoms, to the same bits.

    Every padded atom names one owner (an all-zero one-hot row argmaxes to 0;
    Protenix pads with a real token), and the loop runs once per atom of the
    largest owner. Padded atoms must go to the dropped owner instead.
    """
    rng = np.random.default_rng(5)
    owners = _owners(rng)
    # Every path's largest real owner is at most this (the one-hot one keeps
    # the out-of-range owners too, modulo the token count).
    largest = int(np.bincount(owners % N_TOKEN, minlength=N_TOKEN).max())
    n_pad = 4 * N_ATOM
    padded_owners = np.concatenate([owners, np.zeros(n_pad, np.int32)])
    mask = np.arange(N_ATOM + n_pad) < N_ATOM
    values = rng.normal(size=(2, N_ATOM + n_pad, 6)).astype(np.float32)
    trip_counts: list[int] = []
    traced = segment_module._segment_sum

    def recording(values, plan, num_segments, axis):
        jax.debug.callback(
            lambda m: trip_counts.append(int(np.max(m))), jnp.max(plan.counts)
        )
        return traced(values, plan, num_segments, axis)

    monkeypatch.setattr(segment_module, "_segment_sum", recording)

    protenix = aggregate_atom_to_token(
        jnp.asarray(values),
        jnp.asarray(padded_owners),
        n_token=N_TOKEN,
        atom_mask=jnp.asarray(mask),
    )
    unpadded = aggregate_atom_to_token(
        jnp.asarray(values[:, :N_ATOM]), jnp.asarray(owners), n_token=N_TOKEN
    )
    np.testing.assert_array_equal(protenix, unpadded)

    one_hot = jax.nn.one_hot(np.where(mask, padded_owners % N_TOKEN, -1), N_TOKEN)
    index = _one_hot_index(jnp.stack([one_hot] * 2))
    token_idx = jnp.stack([jnp.asarray(np.where(mask, padded_owners, 0))] * 2)
    valid = jnp.stack([jnp.asarray(mask)] * 2)
    boltz = scatter_atoms_to_tokens_mean(
        None, jnp.asarray(values), index=index, num_tokens=N_TOKEN
    )
    hoisted = scatter_atoms_to_tokens_mean(
        None,
        jnp.asarray(values),
        index=index,
        num_tokens=N_TOKEN,
        plan=atom_to_token_plan(index, N_TOKEN),
    )
    old = _old_boltz2_mean(index[0], index[1], jnp.asarray(values))
    np.testing.assert_array_equal(boltz, old)
    np.testing.assert_array_equal(hoisted, old)
    cp = scatter_atoms_to_tokens_mean_cp(
        jnp.asarray(values), token_idx, valid, num_tokens=N_TOKEN
    )
    np.testing.assert_array_equal(cp, _old_cp_mean(token_idx, valid, values))

    jax.effects_barrier()
    assert trip_counts and max(trip_counts) <= largest, (trip_counts, largest)


def test_a_reused_plan_sums_to_the_same_bits() -> None:
    rng = np.random.default_rng(6)
    owners = jnp.asarray(_owners(rng))
    values = jnp.asarray(rng.normal(size=(N_ATOM, 5)), jnp.float32)
    plan = segment_plan(owners, N_TOKEN)

    np.testing.assert_array_equal(
        segment_sum(values, None, N_TOKEN, plan=plan),
        segment_sum(values, owners, N_TOKEN),
    )
    with pytest.raises(ValueError, match="plan does not match"):
        segment_sum(values[:-1], None, N_TOKEN, plan=plan)


def test_aggregation_graphs_carry_no_scatter() -> None:
    token_idx, valid, values = _boltz_inputs(jnp.float32)
    owners = token_idx[0]
    graphs = {
        "boltz2": lambda v: scatter_atoms_to_tokens_mean(
            None, v, index=(token_idx, valid), num_tokens=N_TOKEN
        ),
        "cp_serial": lambda v: scatter_atoms_to_tokens_mean_cp(
            v, token_idx, valid, num_tokens=N_TOKEN
        ),
        "protenix": lambda v: aggregate_atom_to_token(v, owners, n_token=N_TOKEN),
        "protenix_grad": jax.grad(
            lambda v: jnp.sum(aggregate_atom_to_token(v, owners, n_token=N_TOKEN))
        ),
    }
    for name, fn in graphs.items():
        assert "scatter" not in jax.jit(fn).lower(values).as_text(), name
