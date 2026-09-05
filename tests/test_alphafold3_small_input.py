"""Regression gates for FoldJAX's managed AlphaFold 3 source patch."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
import pytest


def _generic_chain_pair_pae_reference(
    confidences,
    *,
    num_tokens,
    asym_ids,
    full_pae,
    mask=None,
    contact_probs=None,
):
    """The pre-specialisation chain loop, retained as an independent oracle."""
    if mask is None:
        mask = np.ones(shape=full_pae.shape[1:], dtype=bool)
    if contact_probs is None:
        contact_probs = np.ones(shape=full_pae.shape[1:], dtype=float)
    full_pae = full_pae[:, :num_tokens, :num_tokens]
    mask = mask[:num_tokens, :num_tokens]
    asym_ids = asym_ids[:num_tokens]
    contact_probs = contact_probs[:num_tokens, :num_tokens]
    unique_asym_ids = np.unique(asym_ids)
    num_samples = full_pae.shape[0]
    num_chains = len(unique_asym_ids)
    means = np.zeros((num_samples, num_chains, num_chains))
    minima = np.zeros((num_samples, num_chains, num_chains))

    for row, asym_id_1 in enumerate(unique_asym_ids):
        subset = full_pae[:, asym_ids == asym_id_1, :]
        subset_mask = mask[asym_ids == asym_id_1, :]
        subset_contact = contact_probs[asym_ids == asym_id_1, :]
        for column, asym_id_2 in enumerate(unique_asym_ids):
            subsubset = subset[:, :, asym_ids == asym_id_2]
            submask = subset_mask[:, asym_ids == asym_id_2]
            subcontact = subset_contact[:, asym_ids == asym_id_2]
            (valid,) = np.where(submask.flatten() > 0)
            values = subsubset.reshape([num_samples, -1])
            weights = subcontact.flatten()
            if not valid.size:
                means[:, row, column] = np.nan
                minima[:, row, column] = np.nan
            else:
                minima[:, row, column] = np.min(values[:, valid], axis=1)
                means[:, row, column] = confidences.weighted_mean(
                    mask=weights[valid], value=values[:, valid], axis=-1
                )
    return means, minima, unique_asym_ids


def _assert_nan_exact(actual, expected) -> None:
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(
        np.nan_to_num(actual, nan=np.inf), np.nan_to_num(expected, nan=np.inf)
    )


def _generic_chain_pair_pde_reference(
    *, num_tokens, asym_ids, full_pde
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-specialisation chain loop, including both axis copies."""
    full_pde = full_pde[:, :num_tokens, :num_tokens]
    asym_ids = asym_ids[:num_tokens]
    unique_asym_ids = np.unique(asym_ids)
    num_samples = full_pde.shape[0]
    means = np.zeros((num_samples, len(unique_asym_ids), len(unique_asym_ids)))
    minima = np.zeros_like(means)
    for row, asym_id_1 in enumerate(unique_asym_ids):
        subset = full_pde[:, asym_ids == asym_id_1, :]
        for column, asym_id_2 in enumerate(unique_asym_ids):
            subsubset = subset[:, :, asym_ids == asym_id_2]
            means[:, row, column] = np.mean(subsubset, axis=(1, 2))
            minima[:, row, column] = np.min(subsubset, axis=(1, 2))
    return means, minima


def _generic_pde_single_reference(
    confidences, *, num_tokens, asym_ids, full_pde, contact_probs
):
    """The historical vectorised products used by ``pde_single``."""
    full_pde = full_pde[:, :num_tokens, :num_tokens]
    contact_probs = contact_probs[:num_tokens, :num_tokens]
    asym_ids = asym_ids[:num_tokens]
    unique_asym_ids = np.unique(asym_ids)
    samples = full_pde.shape[0]
    expanded_asym_ids = asym_ids[None]
    expanded_contacts = contact_probs[None]
    ichain = np.zeros((samples, len(unique_asym_ids)))
    xchain = np.zeros_like(ichain)
    for index, asym_id in enumerate(unique_asym_ids):
        mine = expanded_asym_ids == asym_id
        imask = mine[:, :, None] * mine[:, None, :]
        xmask = mine[:, :, None] * ~mine[:, None, :]
        imask = imask * expanded_contacts
        xmask = xmask * expanded_contacts
        ichain[:, index] = confidences.weighted_mean(
            mask=imask, value=full_pde, axis=(-2, -1)
        )
        xchain[:, index] = confidences.weighted_mean(
            mask=xmask, value=full_pde, axis=(-2, -1)
        )
    full_chain = confidences.weighted_mean(
        mask=expanded_contacts, value=full_pde, axis=(-1,)
    )
    return ichain, xchain, full_chain


def test_pde_single_streams_standard_sample_products(monkeypatch) -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(41)
    padded = rng.normal(size=(5, 13, 13)).astype(np.float32)
    padded[0, 1, 2] = np.nan
    padded[1, 3, 4] = np.inf
    padded[2, 5, 6] = -np.inf
    padded[3, 7, 1] = -0.0
    contact_probs = rng.random((13, 13), dtype=np.float32)
    asym_ids = np.ones(13, dtype=np.int32)
    with np.errstate(invalid="ignore"):
        expected = _generic_pde_single_reference(
            confidences,
            num_tokens=9,
            asym_ids=asym_ids,
            full_pde=padded,
            contact_probs=contact_probs,
        )
    real_samplewise = confidences._samplewise_weighted_mean
    calls = []
    first_pair_mask = None

    def tracked_samplewise(mask, value, axis):
        nonlocal first_pair_mask
        calls.append((value.shape, axis))
        if axis == (-2, -1):
            if first_pair_mask is None:
                first_pair_mask = weakref.ref(mask)
            else:
                assert first_pair_mask() is None
        return real_samplewise(mask, value, axis)

    monkeypatch.setattr(confidences, "_samplewise_weighted_mean", tracked_samplewise)
    with np.errstate(invalid="ignore"):
        actual = confidences.pde_single(
            9,
            asym_ids,
            padded,
            contact_probs,
        )
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        _assert_nan_exact(actual_leaf, expected_leaf)
        np.testing.assert_array_equal(
            np.signbit(actual_leaf), np.signbit(expected_leaf)
        )
    assert calls == [
        ((5, 9, 9), (-2, -1)),
        ((5, 9, 9), (-2, -1)),
        ((5, 9, 9), (-1,)),
    ]


def test_pde_single_keeps_fortran_reduction_order() -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(42)
    full_pde = np.asfortranarray(rng.normal(size=(4, 11, 11)).astype(np.float32))
    contact_probs = rng.random((11, 11), dtype=np.float32)
    asym_ids = np.asarray([1] * 6 + [2] * 5, dtype=np.int32)
    expected = _generic_pde_single_reference(
        confidences,
        num_tokens=11,
        asym_ids=asym_ids,
        full_pde=full_pde,
        contact_probs=contact_probs,
    )
    actual = confidences.pde_single(11, asym_ids, full_pde, contact_probs)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        _assert_nan_exact(actual_leaf, expected_leaf)
        np.testing.assert_array_equal(
            np.signbit(actual_leaf), np.signbit(expected_leaf)
        )


def test_monomer_chain_pair_pde_uses_one_layout_preserving_copy(monkeypatch) -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(39)
    real_asfortranarray = confidences.np.asfortranarray
    calls = []

    def tracked_asfortranarray(value):
        calls.append(np.shape(value))
        return real_asfortranarray(value)

    monkeypatch.setattr(confidences.np, "asfortranarray", tracked_asfortranarray)
    for dtype in (np.float32, np.float64):
        full_pde = rng.normal(size=(5, 11, 11)).astype(dtype)
        expected = _generic_chain_pair_pde_reference(
            num_tokens=8,
            asym_ids=np.ones(11, dtype=np.int32),
            full_pde=full_pde,
        )
        actual = confidences.chain_pair_pde(
            num_tokens=8,
            asym_ids=np.ones(11, dtype=np.int32),
            full_pde=full_pde,
        )
        for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
            _assert_nan_exact(actual_leaf, expected_leaf)

    assert calls == [(5, 8, 8), (5, 8, 8)]

    calls.clear()
    full_pde = rng.normal(size=(3, 8, 8)).astype(np.float32)
    asym_ids = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    expected = _generic_chain_pair_pde_reference(
        num_tokens=8, asym_ids=asym_ids, full_pde=full_pde
    )
    actual = confidences.chain_pair_pde(8, asym_ids, full_pde)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        _assert_nan_exact(actual_leaf, expected_leaf)
    assert calls == []


def test_multimer_chain_pair_pde_bounds_selection_scratch(monkeypatch) -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(40)
    full_pde = np.asfortranarray(rng.normal(size=(4, 13, 13)).astype(np.float32))
    full_pde[0, 1, 8] = np.nan
    full_pde[1, 7, 2] = np.inf
    full_pde[2, 9, 4] = -np.inf
    full_pde[3, 3, 11] = -0.0
    asym_ids = np.asarray([1, 2, 1, 1, 3, 2, 1, 3, 2, 3, 1, 2, 3])
    with np.errstate(invalid="ignore"):
        expected = _generic_chain_pair_pde_reference(
            num_tokens=13,
            asym_ids=asym_ids,
            full_pde=full_pde,
        )
    real_submatrix = confidences._chain_pair_submatrix
    calls = []
    previous = None

    def tracked_submatrix(values, rows, columns):
        nonlocal previous
        if previous is not None:
            assert previous() is None
        selected = real_submatrix(values, rows, columns)
        assert selected.flags.f_contiguous
        calls.append((tuple(rows), tuple(columns)))
        previous = weakref.ref(selected)
        return selected

    monkeypatch.setattr(confidences, "_chain_pair_submatrix", tracked_submatrix)
    with np.errstate(invalid="ignore"):
        actual = confidences.chain_pair_pde(13, asym_ids, full_pde)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        _assert_nan_exact(actual_leaf, expected_leaf)
        np.testing.assert_array_equal(
            np.signbit(actual_leaf), np.signbit(expected_leaf)
        )
    assert len(calls) == 9


def test_empty_chain_pair_pde_keeps_the_empty_shapes() -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    mean, minimum = confidences.chain_pair_pde(
        0,
        np.zeros(0, dtype=np.int32),
        np.zeros((3, 0, 0), dtype=np.float32),
    )
    assert mean.shape == (3, 0, 0)
    assert minimum.shape == (3, 0, 0)


def test_monomer_chain_pair_pae_avoids_the_duplicate_axis_copy() -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(31)
    for dtype in (np.float32, np.float64):
        for samples in (1, 5):
            num_tokens, padding = 7, 3
            full_pae = rng.normal(
                size=(samples, num_tokens + padding, num_tokens + padding)
            ).astype(dtype)
            asym_ids = np.ones(num_tokens + padding, dtype=np.int32)
            mask = rng.random((num_tokens + padding, num_tokens + padding)) > 0.2
            contact_probs = rng.random(
                (num_tokens + padding, num_tokens + padding)
            ).astype(dtype)
            for contacts in (contact_probs, None):
                for frame_mask in (mask, np.ones_like(mask)):
                    expected = _generic_chain_pair_pae_reference(
                        confidences,
                        num_tokens=num_tokens,
                        asym_ids=asym_ids,
                        full_pae=full_pae,
                        mask=frame_mask,
                        contact_probs=contacts,
                    )
                    actual = confidences.chain_pair_pae(
                        num_tokens=num_tokens,
                        asym_ids=asym_ids,
                        full_pae=full_pae,
                        mask=frame_mask,
                        contact_probs=contacts,
                    )
                    for actual_leaf, expected_leaf in zip(
                        actual, expected, strict=True
                    ):
                        _assert_nan_exact(actual_leaf, expected_leaf)

            empty_mask = np.zeros_like(mask)
            actual = confidences.chain_pair_pae(
                num_tokens=num_tokens,
                asym_ids=asym_ids,
                full_pae=full_pae,
                mask=empty_mask,
                contact_probs=contact_probs,
            )
            assert np.isnan(actual[0]).all()
            assert np.isnan(actual[1]).all()


def test_monomer_chain_pair_pae_routes_through_one_flat_selection(
    monkeypatch,
) -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(35)
    full_pae = rng.random((2, 5, 5), dtype=np.float32)
    asym_ids = np.ones(5, dtype=np.int32)
    expected = _generic_chain_pair_pae_reference(
        confidences,
        num_tokens=5,
        asym_ids=asym_ids,
        full_pae=full_pae,
        mask=None,
        contact_probs=None,
    )
    real_divmod = confidences.np.divmod
    calls = []

    def tracked_divmod(*args, **kwargs):
        calls.append(np.shape(args[0]))
        return real_divmod(*args, **kwargs)

    monkeypatch.setattr(confidences.np, "divmod", tracked_divmod)
    actual = confidences.chain_pair_pae(
        num_tokens=5,
        asym_ids=asym_ids,
        full_pae=full_pae,
    )
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        _assert_nan_exact(actual_leaf, expected_leaf)
    assert calls == [(25,)]

    calls.clear()
    confidences.chain_pair_pae(
        num_tokens=5,
        asym_ids=np.asarray([1, 1, 1, 2, 2], dtype=np.int32),
        full_pae=full_pae,
    )
    assert calls == []


def test_empty_chain_pair_pae_keeps_the_empty_shapes() -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    mean, minimum, chain_ids = confidences.chain_pair_pae(
        num_tokens=0,
        asym_ids=np.zeros(0, dtype=np.int32),
        full_pae=np.zeros((3, 0, 0), dtype=np.float32),
    )
    assert mean.shape == (3, 0, 0)
    assert minimum.shape == (3, 0, 0)
    assert chain_ids.shape == (0,)


def test_multimer_chain_pair_pae_keeps_the_generic_path(monkeypatch) -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(32)
    full_pae = rng.random((3, 8, 8), dtype=np.float32)
    asym_ids = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    mask = rng.random((8, 8)) > 0.15
    contact_probs = rng.random((8, 8), dtype=np.float32)
    expected = _generic_chain_pair_pae_reference(
        confidences,
        num_tokens=8,
        asym_ids=asym_ids,
        full_pae=full_pae,
        mask=mask,
        contact_probs=contact_probs,
    )
    real_submatrix = confidences._chain_pair_submatrix
    calls = []
    previous_group = []

    def tracked_submatrix(values, rows, columns):
        nonlocal previous_group
        if calls and len(calls) % 3 == 0:
            assert all(reference() is None for reference in previous_group)
            previous_group = []
        selected = real_submatrix(values, rows, columns)
        calls.append((values.ndim, selected.flags.f_contiguous))
        previous_group.append(weakref.ref(selected))
        return selected

    monkeypatch.setattr(confidences, "_chain_pair_submatrix", tracked_submatrix)
    actual = confidences.chain_pair_pae(
        num_tokens=8,
        asym_ids=asym_ids,
        full_pae=full_pae,
        mask=mask,
        contact_probs=contact_probs,
    )
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        _assert_nan_exact(actual_leaf, expected_leaf)
    assert calls == [(3, True), (2, True), (2, True)] * 4
    assert all(reference() is None for reference in previous_group)


def test_numpy_rank_metric_reduces_each_sample_separately(monkeypatch) -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(33)
    full_pde = rng.random((5, 17, 17), dtype=np.float32)
    contact_probs = rng.random((17, 17), dtype=np.float32)
    expected = -np.sum(full_pde * contact_probs[None], axis=(-2, -1)) / (
        np.sum(contact_probs) + 1e-6
    )

    real_sum = np.sum
    reduced_shapes = []

    def tracked_sum(value, *args, **kwargs):
        reduced_shapes.append(np.shape(value))
        return real_sum(value, *args, **kwargs)

    monkeypatch.setattr(confidences.np, "sum", tracked_sum)
    actual = confidences.rank_metric(full_pde, contact_probs)

    np.testing.assert_array_equal(actual, expected)
    assert full_pde.shape not in reduced_shapes
    assert reduced_shapes.count(contact_probs.shape) == full_pde.shape[0] + 1


def test_rank_metric_keeps_single_sample_and_jax_semantics(monkeypatch) -> None:
    import jax.numpy as jnp

    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import confidences

    rng = np.random.default_rng(34)
    full_pde = rng.random((1, 9, 9), dtype=np.float32)
    contact_probs = rng.random((9, 9), dtype=np.float32)
    expected = -np.sum(full_pde * contact_probs[None], axis=(-2, -1)) / (
        np.sum(contact_probs) + 1e-6
    )
    np.testing.assert_array_equal(
        confidences.rank_metric(full_pde, contact_probs), expected
    )

    jax_full_pde = jnp.asarray(np.repeat(full_pde, 3, axis=0))
    jax_contacts = jnp.asarray(contact_probs)
    jax_expected = -jnp.sum(jax_full_pde * jax_contacts[None], axis=(-2, -1)) / (
        jnp.sum(jax_contacts) + 1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(confidences.rank_metric(jax_full_pde, jax_contacts)),
        np.asarray(jax_expected),
    )

    # NumPy can choose a different summation traversal for a Fortran-order
    # sample stack. Such uncommon direct inputs retain the literal historical
    # vectorised expression rather than taking the C-order streaming path.
    fortran_full_pde = np.asfortranarray(np.repeat(full_pde, 3, axis=0))
    fortran_expected = -np.sum(
        fortran_full_pde * contact_probs[None], axis=(-2, -1)
    ) / (np.sum(contact_probs) + 1e-6)
    real_sum = np.sum
    reduced_shapes = []

    def tracked_sum(value, *args, **kwargs):
        reduced_shapes.append(np.shape(value))
        return real_sum(value, *args, **kwargs)

    monkeypatch.setattr(confidences.np, "sum", tracked_sum)
    np.testing.assert_array_equal(
        confidences.rank_metric(fortran_full_pde, contact_probs), fortran_expected
    )
    assert fortran_full_pde.shape in reduced_shapes


def test_pae_single_mask_is_an_exact_zero_copy_broadcast() -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import model as af3_model

    frame_mask = np.asarray([True, False, True, True], dtype=bool)
    expected = np.tile(frame_mask[:, None], [1, frame_mask.shape[0]])

    actual = af3_model._broadcast_pae_single_mask(frame_mask)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (4, 4)
    assert actual.dtype == frame_mask.dtype
    assert np.shares_memory(actual, frame_mask)
    assert not actual.flags.owndata
    assert not actual.flags.writeable

    rng = np.random.default_rng(0)
    asym_id = np.asarray([1, 1, 2, 2], dtype=np.int32)
    tm_adjusted_pae = rng.random((2, 4, 4), dtype=np.float32)
    full_pae = rng.random((2, 4, 4), dtype=np.float32)
    contact_probs = rng.random((4, 4), dtype=np.float32)
    for interface in (False, True):
        tiled_score = af3_model.confidences.predicted_tm_score(
            tm_adjusted_pae[0], expected, asym_id, interface=interface
        )
        broadcast_score = af3_model.confidences.predicted_tm_score(
            tm_adjusted_pae[0], actual, asym_id, interface=interface
        )
        np.testing.assert_array_equal(broadcast_score, tiled_score)

    tiled_metrics = af3_model.confidences.pae_metrics(
        num_tokens=4,
        asym_ids=asym_id,
        full_pae=full_pae,
        mask=expected,
        contact_probs=contact_probs,
        tm_adjusted_pae=tm_adjusted_pae,
    )
    broadcast_metrics = af3_model.confidences.pae_metrics(
        num_tokens=4,
        asym_ids=asym_id,
        full_pae=full_pae,
        mask=actual,
        contact_probs=contact_probs,
        tm_adjusted_pae=tm_adjusted_pae,
    )
    assert broadcast_metrics.keys() == tiled_metrics.keys()
    for name in broadcast_metrics:
        np.testing.assert_array_equal(
            np.isnan(broadcast_metrics[name]), np.isnan(tiled_metrics[name])
        )
        np.testing.assert_array_equal(
            np.nan_to_num(broadcast_metrics[name], nan=np.inf),
            np.nan_to_num(tiled_metrics[name], nan=np.inf),
        )


def test_monomer_skips_interface_tm_but_multimer_keeps_it(monkeypatch) -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import model as af3_model

    calls = []

    def fake_ptm(*, interface, **unused):
        calls.append(interface)
        return np.asarray([0.7 if interface else 0.8], dtype=np.float32)

    monkeypatch.setattr(af3_model, "_compute_ptm", fake_ptm)
    mask = np.ones((2, 2), dtype=bool)

    ptm, iptm, is_multimer = af3_model._compute_global_ptm_metrics(
        result={},
        num_tokens=2,
        asym_ids=np.asarray([1, 1], dtype=np.int32),
        pae_single_mask=mask,
    )
    assert calls == [False]
    assert is_multimer is False
    np.testing.assert_array_equal(ptm, np.asarray([0.8], dtype=np.float32))
    assert iptm.shape == ptm.shape
    assert iptm.dtype == ptm.dtype
    assert np.isnan(iptm).all()

    calls.clear()
    ptm, iptm, is_multimer = af3_model._compute_global_ptm_metrics(
        result={},
        num_tokens=2,
        asym_ids=np.asarray([1, 2], dtype=np.int32),
        pae_single_mask=mask,
    )
    assert calls == [False, True]
    assert is_multimer is True
    np.testing.assert_array_equal(ptm, np.asarray([0.8], dtype=np.float32))
    np.testing.assert_array_equal(iptm, np.asarray([0.7], dtype=np.float32))


@pytest.mark.parametrize("padded_atoms", [96, 160])
def test_atom_cross_attention_pads_a_shorter_than_key_window_layout(
    padded_atoms,
) -> None:
    """Four-residue jobs work without forcing the released 256-token bucket."""
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import features
    from alphafold3.model.atom_layout import atom_layout

    shape = (4, 24)
    atom_name = np.full(shape, "", dtype=object)
    atom_element = np.full(shape, "", dtype=object)
    res_name = np.full(shape, "", dtype=object)
    res_id = np.zeros(shape, dtype=int)
    chain_id = np.full(shape, "", dtype=object)
    chain_type = np.full(shape, "", dtype=object)
    for residue_index in range(4):
        atom_name[residue_index, :5] = ("N", "CA", "C", "O", "CB")
        atom_element[residue_index, :5] = ("N", "C", "C", "O", "C")
        res_name[residue_index, :5] = "ALA"
        res_id[residue_index, :5] = residue_index + 1
        chain_id[residue_index, :5] = "A"
        chain_type[residue_index, :5] = "polypeptide(L)"

    layout = atom_layout.AtomLayout(
        atom_name=atom_name,
        atom_element=atom_element,
        res_name=res_name,
        res_id=res_id,
        chain_id=chain_id,
        chain_type=chain_type,
    )
    result = features.AtomCrossAtt.compute_features(
        all_token_atoms_layout=layout,
        queries_subset_size=32,
        keys_subset_size=128,
        padding_shapes=features.PaddingShapes(
            num_tokens=4,
            msa_size=1,
            num_chains=1,
            num_templates=0,
            num_atoms=padded_atoms,
        ),
    )

    rows = padded_atoms // 32
    assert result.tokens_to_keys.gather_idxs.shape == (rows, 128)
    assert result.tokens_to_keys.gather_mask.shape == (rows, 128)
    assert np.count_nonzero(result.tokens_to_keys.gather_mask) == rows * 20
    if padded_atoms >= 128:
        # Native negative gather indices wrap into existing padded storage.
        # The tiny-storage extension must not reorder these ordinary windows.
        assert not result.tokens_to_keys.gather_mask[2, :-20].any()
        assert result.tokens_to_keys.gather_mask[2, -20:].all()


def test_diffusion_padding_preserves_multisample_noise_across_every_step(
    monkeypatch,
) -> None:
    """Serving buckets keep AF3's exact compact random stream on real tokens."""
    import haiku as hk
    import jax
    import jax.numpy as jnp

    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model.network import diffusion_head

    monkeypatch.setattr(
        diffusion_head,
        "random_augmentation",
        lambda rng_key, positions, mask: positions * mask[..., None],
    )
    config = diffusion_head.SampleConfig(
        steps=4,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        num_samples=2,
    )

    def forward(atom_mask, token_mask, sample_key, prefix_stable_noise):
        batch = type("Batch", (), {})()
        batch.predicted_structure_info = type("Pred", (), {"atom_mask": atom_mask})()
        batch.token_features = type("Tokens", (), {"mask": token_mask})()
        return diffusion_head.sample(
            denoising_step=lambda positions, noise_level: positions,
            batch=batch,
            key=sample_key,
            config=config,
            prefix_stable_noise=prefix_stable_noise,
        )

    transformed = hk.transform(forward)
    sample_key = jax.random.PRNGKey(23)
    exact_mask = jnp.ones((3, 4), dtype=jnp.float32)
    padded_mask = jnp.pad(exact_mask, ((0, 5), (0, 0)))
    exact_token_mask = jnp.ones((3,), dtype=jnp.float32)
    padded_token_mask = jnp.pad(exact_token_mask, (0, 5))

    run_key, initial_noise_key = jax.random.split(sample_key)
    exact_initial_noise = jax.random.normal(
        initial_noise_key,
        (config.num_samples, *exact_mask.shape, 3),
    )
    padded_initial_noise = diffusion_head._prefix_stable_atom_noise(
        initial_noise_key,
        jnp.broadcast_to(
            padded_token_mask[None, :],
            (config.num_samples, padded_token_mask.shape[0]),
        ),
        trailing_shape=exact_mask.shape[1:] + (3,),
    )
    np.testing.assert_array_equal(
        np.asarray(padded_initial_noise[:, :3]),
        np.asarray(exact_initial_noise),
    )
    assert not np.any(np.asarray(padded_initial_noise[:, 3:]))

    step_keys = jax.random.split(run_key, config.num_samples)
    for _ in range(config.steps):
        split_keys = jax.vmap(lambda key: jax.random.split(key, 3))(step_keys)
        step_keys = split_keys[:, 0]
        noise_keys = split_keys[:, 1]
        exact_step_noise = jax.vmap(
            lambda key: jax.random.normal(key, (*exact_mask.shape, 3))
        )(noise_keys)
        padded_step_noise = jax.vmap(
            lambda key: diffusion_head._prefix_stable_atom_noise(
                key,
                padded_token_mask,
                trailing_shape=exact_mask.shape[1:] + (3,),
            )
        )(noise_keys)
        np.testing.assert_array_equal(
            np.asarray(padded_step_noise[:, :3]),
            np.asarray(exact_step_noise),
        )
        assert not np.any(np.asarray(padded_step_noise[:, 3:]))

    params = transformed.init(
        sample_key,
        exact_mask,
        exact_token_mask,
        sample_key,
        False,
    )
    exact = transformed.apply(
        params,
        sample_key,
        exact_mask,
        exact_token_mask,
        sample_key,
        False,
    )["atom_positions"]
    padded = transformed.apply(
        params,
        sample_key,
        padded_mask,
        padded_token_mask,
        sample_key,
        True,
    )["atom_positions"]

    # The stochastic tape above is bitwise identical. The full padded scan can
    # still choose a shape-specific fused arithmetic schedule on CPU, so compare
    # its real-token rollout at a tight float32 tolerance.
    np.testing.assert_allclose(
        np.asarray(padded[:, :3]),
        np.asarray(exact),
        rtol=3e-6,
        atol=1e-3,
    )
    assert not np.any(np.asarray(padded[:, 3:]))


def test_diffusion_sample_sharding_preserves_the_explicit_rng_tape(monkeypatch) -> None:
    """Raised sample counts bound denoiser width without changing samples."""
    import haiku as hk
    import jax
    import jax.numpy as jnp

    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model.network import diffusion_head

    monkeypatch.setattr(
        diffusion_head,
        "random_augmentation",
        lambda rng_key, positions, mask: positions * mask[..., None],
    )
    config = diffusion_head.SampleConfig(
        steps=3,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        num_samples=7,
    )

    def forward(atom_mask, token_mask, sample_key, shard_size):
        batch = type("Batch", (), {})()
        batch.predicted_structure_info = type("Pred", (), {"atom_mask": atom_mask})()
        batch.token_features = type("Tokens", (), {"mask": token_mask})()
        return diffusion_head.sample(
            denoising_step=lambda positions, noise_level: (
                positions / (1.0 + noise_level**2)
            ),
            batch=batch,
            key=sample_key,
            config=config,
            sample_shard_size=shard_size,
        )

    transformed = hk.transform(forward)
    key = jax.random.PRNGKey(20260831)
    atom_mask = jnp.ones((4, 3), dtype=jnp.float32)
    token_mask = jnp.ones((4,), dtype=jnp.float32)
    params = transformed.init(key, atom_mask, token_mask, key, None)
    unsharded = transformed.apply(params, key, atom_mask, token_mask, key, None)[
        "atom_positions"
    ]
    sharded = transformed.apply(params, key, atom_mask, token_mask, key, 5)[
        "atom_positions"
    ]

    assert sharded.shape == (7, 4, 3, 3)
    # The keys and random draws are identical. A different mapped batch width
    # can select a different fused float32 schedule, so retain the same tight
    # numerical contract used by the padding sampler gate above.
    np.testing.assert_allclose(
        np.asarray(sharded),
        np.asarray(unsharded),
        rtol=4e-6,
        atol=2e-4,
    )


def test_af3_only_shards_sample_counts_above_the_released_batch() -> None:
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import model as af3_model

    diffusion_config = SimpleNamespace(
        eval_batch_size=5,
        eval_batch_dim_shard_size=5,
    )
    assert (
        af3_model._diffusion_sample_shard_size(
            diffusion_config,
            SimpleNamespace(num_samples=5),
        )
        is None
    )
    assert (
        af3_model._diffusion_sample_shard_size(
            diffusion_config,
            SimpleNamespace(num_samples=6),
        )
        == 5
    )


def test_inference_result_reuses_pae_metric_chain_aggregates(monkeypatch) -> None:
    """Host postprocessing does not recompute aggregates from ``pae_metrics``."""
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import model as af3_model

    batch = SimpleNamespace(
        token_features=SimpleNamespace(
            seq_length=np.asarray(2),
            asym_id=np.asarray([1, 2]),
            residue_index=np.asarray([10, 20]),
        ),
        frames=SimpleNamespace(mask=np.ones(2, dtype=np.float32)),
    )
    predicted_sample = object()
    predicted_structure = SimpleNamespace(
        chains=("A", "B"),
        unstack=lambda: [predicted_sample],
    )
    monkeypatch.setattr(
        af3_model.feat_batch.Batch,
        "from_data_dict",
        lambda unused: batch,
    )
    monkeypatch.setattr(
        af3_model,
        "get_predicted_structure",
        lambda **unused: predicted_structure,
    )

    def fake_ptm(*, interface, **unused):
        return np.asarray([0.7 if interface else 0.8], dtype=np.float32)

    monkeypatch.setattr(af3_model, "_compute_ptm", fake_ptm)
    monkeypatch.setattr(
        af3_model.confidences,
        "chain_pair_pde",
        lambda **unused: (
            np.zeros((1, 2, 2), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        af3_model.confidences,
        "pde_single",
        lambda *unused: (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    sentinels = {
        "chain_pair_pae_min": np.asarray([[[11.0, 12.0], [13.0, 14.0]]]),
        "chain_pair_iptm": np.asarray([[[21.0, 22.0], [23.0, 24.0]]]),
        "iptm_ichain": np.asarray([[31.0, 32.0]]),
        "iptm_xchain": np.asarray([[41.0, 42.0]]),
        "pae_ichain": np.asarray([[51.0, 52.0]]),
        "pae_xchain": np.asarray([[61.0, 62.0]]),
    }
    monkeypatch.setattr(
        af3_model.confidences,
        "pae_metrics",
        lambda **unused: sentinels,
    )
    monkeypatch.setattr(
        af3_model.confidences,
        "rank_metric",
        lambda *unused: np.asarray([0.6], dtype=np.float32),
    )
    monkeypatch.setattr(af3_model.confidences, "has_clash", lambda unused: False)
    monkeypatch.setattr(
        af3_model.confidences,
        "fraction_disordered",
        lambda unused: 0.1,
    )
    monkeypatch.setattr(
        af3_model.confidences,
        "get_ranking_score",
        lambda **unused: 0.9,
    )

    def fail_duplicate(*args, **kwargs):
        del args, kwargs
        raise AssertionError("aggregate was recomputed instead of reused")

    monkeypatch.setattr(af3_model.confidences, "chain_pair_pae", fail_duplicate)
    monkeypatch.setattr(af3_model, "_compute_chain_pair_iptm", fail_duplicate)
    monkeypatch.setattr(af3_model.confidences, "get_iptm_xchain", fail_duplicate)

    result = {
        "distogram": {"contact_probs": np.ones((2, 2), dtype=np.float32)},
        "full_pae": np.zeros((1, 2, 2), dtype=np.float32),
        "full_pde": np.zeros((1, 2, 2), dtype=np.float32),
        "tmscore_adjusted_pae_interface": np.zeros((1, 2, 2), dtype=np.float32),
        "average_pde": np.asarray([0.25], dtype=np.float32),
        "__identifier__": b"test-model",
    }
    [inference_result] = af3_model.Model.get_inference_result({}, result)

    assert inference_result.predicted_structure is predicted_sample
    for name in (
        "chain_pair_pae_min",
        "chain_pair_iptm",
        "iptm_ichain",
        "iptm_xchain",
    ):
        np.testing.assert_array_equal(
            inference_result.metadata[name],
            sentinels[name][0],
        )
