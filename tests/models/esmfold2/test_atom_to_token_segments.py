"""Sparse atom-to-token reductions preserve the historical dense result."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.data import chemistry
from foldjax.models.esmfold2.data.features import build_features, pad_features
from foldjax.models.esmfold2.models import heads
from foldjax.models.esmfold2.models.segments import (
    MAX_ATOMS_PER_TOKEN,
    sum_by_token,
)


def _legacy_sum(values, atom_to_token, n_tokens):
    owners = jax.nn.one_hot(atom_to_token, n_tokens, dtype=values.dtype)
    if values.ndim == 2:
        return jnp.einsum("ba,bat->bt", values, owners)
    return jnp.einsum("bnt,bnd->btd", owners, values)


def _legacy_mean(values, atom_to_token, n_tokens, mask):
    weights = mask.astype(values.dtype)[..., None]
    owners = jax.nn.one_hot(atom_to_token, n_tokens, dtype=values.dtype)
    totals = jnp.einsum("bnt,bnd->btd", owners, values * weights)
    counts = jnp.einsum(
        "bnt,bnd->btd", owners, jnp.broadcast_to(weights, values.shape)
    )
    return jnp.where(counts > 0, totals / jnp.maximum(counts, 1e-12), 0.0)


def _fixture(dtype):
    rng = np.random.default_rng(17)
    counts = np.asarray([1, 7, 3, 23, 5, 11], dtype=np.int32)
    owners = np.repeat(np.arange(counts.size, dtype=np.int32), counts)
    # Padding is assigned to token zero by the real featurizer and ignored by
    # its suffix mask. A final invalid owner also pins one_hot's drop rule.
    owners = np.concatenate([owners, np.zeros(5, np.int32), [-1]])[None]
    mask = np.ones_like(owners, dtype=bool)
    mask[:, -6:] = False
    values = rng.standard_normal((1, owners.shape[1], 37)).astype(np.float32)
    return jnp.asarray(values, dtype), jnp.asarray(owners), jnp.asarray(mask)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_sparse_mean_matches_the_dense_one_hot_reduction(dtype) -> None:
    values, owners, mask = _fixture(dtype)
    expected = np.asarray(_legacy_mean(values, owners, 6, mask))
    weights = mask.astype(values.dtype)
    totals = sum_by_token(values * weights[..., None], owners, 6, mask)
    counts = sum_by_token(weights, owners, 6, mask)[..., None]
    actual = np.asarray(
        jnp.where(counts > 0, totals / jnp.maximum(counts, 1e-12), 0.0)
    )

    if dtype == jnp.bfloat16:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual,
            expected,
            atol=np.finfo(np.float32).eps,
            rtol=np.finfo(np.float32).eps,
        )
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_confidence_mean_matches_the_dense_one_hot_reduction(dtype) -> None:
    values, owners, mask = _fixture(dtype)
    # pLDDT is non-negative and the head exposes the per-token mean, not the
    # unreduced numerator. Exercise that actual numerical boundary.
    values = jax.nn.sigmoid(values[..., 0]) * mask
    weights = mask.astype(values.dtype)
    expected = np.asarray(
        _legacy_sum(values, owners, 6)
        / jnp.clip(_legacy_sum(weights, owners, 6), min=1e-6)
    )
    actual = np.asarray(
        heads._scatter_sum(
            values, owners, 6, mask, contiguous_atom_groups=True
        )
        / jnp.clip(
            heads._scatter_sum(
                weights, owners, 6, mask, contiguous_atom_groups=True
            ),
            min=1e-6,
        )
    )

    if dtype == jnp.bfloat16:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual,
            expected,
            atol=2 * np.finfo(np.float32).eps,
            rtol=2 * np.finfo(np.float32).eps,
        )


def test_sparse_reduction_has_no_atom_by_token_dot() -> None:
    atoms, tokens = 8192, 1024
    values = jnp.zeros((1, atoms), jnp.bfloat16)
    owners = jnp.repeat(jnp.arange(tokens, dtype=jnp.int32), atoms // tokens)[None]

    mask = jnp.ones_like(owners, dtype=bool)
    lowered = jax.jit(
        lambda x, i, m: heads._scatter_sum(
            x,
            i,
            tokens,
            m,
            contiguous_atom_groups=True,
        )
    ).lower(values, owners, mask)
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiled = lowered.compile()

    assert "stablehlo.gather" in hlo
    assert "stablehlo.scatter" not in hlo
    assert "stablehlo.dot_general" not in hlo
    # The dense owner matrix alone is 16 MiB at bfloat16. The fixed-rank loop
    # streams one token-wide gather, so its complete arena stays far below it.
    dense_owner_bytes = atoms * tokens * jnp.dtype(jnp.bfloat16).itemsize
    assert compiled.memory_analysis().temp_size_in_bytes < dense_owner_bytes


def test_shape_contract_fails_before_tracing_a_wrong_owner_array() -> None:
    with pytest.raises(ValueError, match="batch and atom axes"):
        sum_by_token(
            jnp.zeros((1, 4, 3)),
            jnp.zeros((1, 3), jnp.int32),
            2,
            jnp.ones((1, 3), bool),
        )


def test_grouped_scatter_accepts_an_empty_atom_axis() -> None:
    result = heads._scatter_sum(
        jnp.zeros((2, 0), dtype=jnp.float32),
        jnp.zeros((2, 0), dtype=jnp.int32),
        3,
        jnp.zeros((2, 0), dtype=bool),
        contiguous_atom_groups=True,
    )
    np.testing.assert_array_equal(np.asarray(result), np.zeros((2, 3), np.float32))


def test_released_chemistry_fits_the_fixed_deterministic_group() -> None:
    assert max(map(len, chemistry.PROTEIN_HEAVY_ATOMS.values())) <= (
        MAX_ATOMS_PER_TOKEN
    )


def test_production_features_have_contiguous_prefix_atom_groups() -> None:
    features = build_features(
        [
            ("ACDEFGHIKLMNPQRSTVWY", "A", 0, 0),
            ("YWVTSRQPNMLKIHGFEDCA", "B", 1, 0),
        ]
    )
    padded = pad_features(
        features,
        n_token=features["token_attention_mask"].shape[-1] + 3,
        n_atom=features["atom_attention_mask"].shape[-1] + 32,
        n_msa=features["msa_attention_mask"].shape[-2] + 2,
    )

    for candidate in (features, padded):
        mask = candidate["atom_attention_mask"][0].astype(bool)
        owners = candidate["atom_to_token"][0]
        n_tokens = candidate["token_attention_mask"].shape[-1]
        n_active = int(mask.sum())

        np.testing.assert_array_equal(mask, np.arange(mask.size) < n_active)
        assert np.all((owners[mask] >= 0) & (owners[mask] < n_tokens))
        assert np.all(np.diff(owners[mask]) >= 0)
        counts = np.bincount(owners[mask], minlength=n_tokens)
        assert counts.max() <= MAX_ATOMS_PER_TOKEN
        assert inference._has_contiguous_atom_groups(candidate)


def test_default_scatter_keeps_generic_direct_call_semantics() -> None:
    values = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    owners = jnp.asarray([[0, 1, 0]], dtype=jnp.int32)
    mask = jnp.ones_like(owners, dtype=bool)
    np.testing.assert_array_equal(
        np.asarray(heads._scatter_sum(values, owners, 2, mask)),
        np.asarray([[4.0, 2.0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "values, mask",
    [
        ([1.0, np.nan, 3.0, 4.0], [1, 1, 1, 1]),
        ([1.0, np.inf, 3.0, 4.0], [1, 1, 1, 1]),
        ([1.0, np.inf, -np.inf, 4.0], [1, 1, 1, 1]),
        ([1.0, 2.0, 3.0, np.nan], [1, 1, 1, 0]),
    ],
)
def test_grouped_scatter_preserves_dense_nonfinite_semantics(values, mask) -> None:
    values = jnp.asarray([values], dtype=jnp.float32)
    owners = jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32)
    mask = jnp.asarray([mask], dtype=bool)
    expected = np.asarray(_legacy_sum(values, owners, 2))
    actual = np.asarray(
        heads._scatter_sum(
            values,
            owners,
            2,
            mask,
            contiguous_atom_groups=True,
        )
    )
    np.testing.assert_array_equal(actual, expected)


def test_grouped_nonfinite_semantics_accept_unsigned_owners() -> None:
    values = jnp.asarray([[np.inf, 1.0, 3.0, 4.0]], dtype=jnp.float32)
    owners = jnp.asarray([[0, 0, 1, 1]], dtype=jnp.uint32)
    mask = jnp.ones_like(owners, dtype=bool)
    expected = np.asarray(_legacy_sum(values, owners, 2))
    actual = np.asarray(
        heads._scatter_sum(
            values,
            owners,
            2,
            mask,
            contiguous_atom_groups=True,
        )
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "failure", ["hole", "unsorted", "overflow", "range", "nonbinary"]
)
def test_external_feature_contract_falls_back_to_generic_reduction(failure) -> None:
    owners = np.repeat(np.arange(2, dtype=np.int32), 23)[None]
    mask = np.ones_like(owners, dtype=bool)
    token_mask = np.ones((1, 2), dtype=bool)
    if failure == "hole":
        mask[0, 3] = False
    elif failure == "unsorted":
        owners[0, 1] = 1
    elif failure == "overflow":
        owners = np.zeros((1, 24), dtype=np.int32)
        mask = np.ones_like(owners, dtype=bool)
    elif failure == "nonbinary":
        mask = mask.astype(np.float32)
        mask[0, 0] = 0.5
    else:
        owners[0, 0] = 2
    features = {
        "atom_to_token": owners,
        "atom_attention_mask": mask,
        "token_attention_mask": token_mask,
    }
    assert not inference._has_contiguous_atom_groups(features)


def test_unsigned_external_owners_are_checked_without_wraparound() -> None:
    token_mask = np.ones((1, 2), dtype=bool)
    valid = {
        "atom_to_token": np.asarray([[0, 0, 1]], dtype=np.uint64),
        "atom_attention_mask": np.ones((1, 3), dtype=bool),
        "token_attention_mask": token_mask,
    }
    assert inference._has_contiguous_atom_groups(valid)

    unsorted = {
        **valid,
        "atom_to_token": np.asarray([[1, 0, 1]], dtype=np.uint32),
    }
    assert not inference._has_contiguous_atom_groups(unsorted)


@pytest.mark.parametrize(
    ("dtype", "n_tokens"),
    [(np.int8, 128), (np.uint8, 256)],
)
def test_narrow_owner_dtype_falls_back_when_the_sentinel_would_wrap(
    dtype, n_tokens
) -> None:
    features = {
        "atom_to_token": np.asarray([[0, 0, 1, 1, 0]], dtype=dtype),
        "atom_attention_mask": np.asarray([[1, 1, 1, 1, 0]], dtype=bool),
        "token_attention_mask": np.ones((1, n_tokens), dtype=bool),
    }
    assert not inference._has_contiguous_atom_groups(features)


def test_inference_routes_the_atom_contract_into_the_compile_identity(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from foldjax.models.esmfold2.models import model as structure_model

    features = build_features([("AG", "A", 0, 0)])
    loaded = SimpleNamespace(
        settings=structure_model.ModelSettings(),
        parameters={},
        esmc_parameters=None,
        esmc_settings=None,
    )
    routed = []

    def fake_compiled_predict(*identity):
        routed.append(identity[-4])
        assert identity[-2] is True
        assert identity[-1] is False

        def run(*args):
            assert args[-4] is identity[-4]
            assert args[-2] is True
            assert args[-1] is False
            return {}

        return run

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)
    inference.predict(jax.random.key(0), features, loaded)

    unsorted = {name: value.copy() for name, value in features.items()}
    unsorted["atom_to_token"][0, 0] = 1
    inference.predict(jax.random.key(1), unsorted, loaded)
    inference.predict(jax.random.key(2), features, loaded, stop_after_trunk=True)
    assert routed == [True, False, False]
