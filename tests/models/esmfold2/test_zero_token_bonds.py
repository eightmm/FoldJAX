"""The protein adapter's verified-zero token-bond graph specialization."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.data.features import build_features, pad_features
from foldjax.models.esmfold2.models import model as structure_model


def _contract(
    *, n_tokens: int = 4, pair_width: int = 8
) -> tuple[dict[str, np.ndarray], dict[str, jax.Array]]:
    features = {
        "token_attention_mask": np.ones((1, n_tokens), dtype=bool),
        "token_bonds": np.zeros((1, n_tokens, n_tokens, 1), dtype=np.float32),
    }
    parameters = {
        "token_bonds.weight": jnp.arange(pair_width, dtype=jnp.bfloat16).reshape(
            pair_width, 1
        )
    }
    return features, parameters


def test_host_contract_accepts_only_a_proven_unbiased_zero_projection() -> None:
    features, parameters = _contract()
    assert inference._has_compact_token_bond_encoding(
        features, parameters, pair_width=8, compute_dtype=jnp.bfloat16
    )
    unchanneled = {
        **features,
        "token_bonds": features["token_bonds"][..., 0],
    }
    assert inference._has_compact_token_bond_encoding(
        unchanneled, parameters, pair_width=8, compute_dtype=jnp.bfloat16
    )

    rejected: list[tuple[dict[str, np.ndarray], dict[str, jax.Array]]] = []
    for value in (1.0, np.nan, np.inf):
        changed = {name: array.copy() for name, array in features.items()}
        changed["token_bonds"][0, 0, 0, 0] = value
        rejected.append((changed, parameters))

    negative_zero = {name: array.copy() for name, array in features.items()}
    negative_zero["token_bonds"][0, 0, 0, 0] = np.float32(-0.0)
    rejected.append((negative_zero, parameters))

    wrong_shape = {name: array.copy() for name, array in features.items()}
    wrong_shape["token_bonds"] = np.zeros((1, 4, 4, 2), dtype=np.float32)
    rejected.append((wrong_shape, parameters))
    wrong_mask = {name: array.copy() for name, array in features.items()}
    wrong_mask["token_attention_mask"] = np.ones(4, dtype=bool)
    rejected.append((wrong_mask, parameters))
    nonnumeric = {name: array.copy() for name, array in features.items()}
    nonnumeric["token_bonds"] = np.full((1, 4, 4, 1), "0")
    rejected.append((nonnumeric, parameters))

    biased = {**parameters, "token_bonds.bias": jnp.zeros(8)}
    rejected.append((features, biased))
    wrong_weight = {"token_bonds.weight": jnp.zeros((7, 1))}
    rejected.append((features, wrong_weight))
    integer_weight = {"token_bonds.weight": jnp.zeros((8, 1), dtype=jnp.int32)}
    rejected.append((features, integer_weight))
    nonfinite_weight = {
        "token_bonds.weight": jnp.full((8, 1), jnp.inf, dtype=jnp.float32)
    }
    rejected.append((features, nonfinite_weight))
    rejected.append((features, {}))
    rejected.append(
        ({"token_attention_mask": features["token_attention_mask"]}, parameters)
    )

    for candidate_features, candidate_parameters in rejected:
        assert not inference._has_compact_token_bond_encoding(
            candidate_features,
            candidate_parameters,
            pair_width=8,
            compute_dtype=jnp.bfloat16,
        )


def test_padded_protein_features_keep_the_verified_zero_contract() -> None:
    features = build_features([("ACDE", "A", 0, 0)])
    padded = pad_features(
        features,
        n_token=8,
        n_atom=64,
        n_msa=4,
    )
    parameters = {"token_bonds.weight": jnp.ones((256, 1), dtype=jnp.bfloat16)}

    assert padded["token_bonds"].shape == (1, 8, 8, 1)
    assert inference._has_compact_token_bond_encoding(
        padded, parameters, pair_width=256, compute_dtype=jnp.bfloat16
    )


@pytest.mark.parametrize(
    ("source_dtype", "compute_dtype", "expected"),
    [
        (np.float32, jnp.float32, True),
        (np.float32, jnp.bfloat16, True),
        (jnp.bfloat16, jnp.bfloat16, True),
        (jnp.bfloat16, jnp.float32, False),
        (np.float16, jnp.bfloat16, False),
        (np.float64, jnp.bfloat16, False),
        (np.float32, jnp.float16, False),
        (np.float32, jnp.float64, False),
    ],
)
def test_host_contract_matches_the_supported_source_compute_matrix(
    source_dtype, compute_dtype, expected
) -> None:
    features, _ = _contract()
    weight = (
        jnp.ones((8, 1), dtype=jnp.bfloat16)
        if source_dtype is jnp.bfloat16
        else np.ones((8, 1), dtype=source_dtype)
    )
    parameters = {"token_bonds.weight": weight}

    assert (
        inference._has_compact_token_bond_encoding(
            features,
            parameters,
            pair_width=8,
            compute_dtype=compute_dtype,
        )
        is expected
    )


def test_finite_fp32_weight_that_casts_to_bf16_infinity_falls_back() -> None:
    features, _ = _contract(pair_width=2)
    maximum = np.finfo(np.float32).max
    parameters = {
        "token_bonds.weight": jnp.asarray(
            [[maximum], [-maximum]], dtype=jnp.float32
        )
    }
    assert not inference._has_compact_token_bond_encoding(
        features,
        parameters,
        pair_width=2,
        compute_dtype=jnp.bfloat16,
    )

    cast = structure_model._cast(
        parameters, structure_model.TRUNK_PREFIXES, jnp.bfloat16
    )
    assert jnp.all(jnp.isinf(cast["token_bonds.weight"]))
    generic = structure_model._token_bonds_encoding(
        jnp.asarray(features["token_bonds"]), cast, jnp.dtype(jnp.bfloat16)
    )
    assert jnp.all(jnp.isnan(generic))


def test_inference_routes_the_zero_contract_into_the_compile_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = build_features([("AG", "A", 0, 0)])
    settings = structure_model.ModelSettings()
    parameters = {
        "token_bonds.weight": jnp.ones(
            (settings.d_pair, 1), dtype=jnp.bfloat16
        )
    }
    routed: list[bool] = []

    def fake_compiled_predict(*identity):
        routed.append(identity[-3])
        assert identity[-2] is True
        assert identity[-1] is False

        def run(*args):
            assert args[-3] is identity[-3]
            assert args[-2] is True
            assert args[-1] is False
            return {}

        return run

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)

    def loaded(params=parameters, *, model_settings=settings):
        return SimpleNamespace(
            settings=model_settings,
            parameters=params,
            esmc_parameters=None,
            esmc_settings=None,
        )

    inference.predict(jax.random.key(0), features, loaded())
    nonzero = {name: value.copy() for name, value in features.items()}
    nonzero["token_bonds"][0, 0, 0, 0] = 1.0
    inference.predict(jax.random.key(1), nonzero, loaded())
    inference.predict(
        jax.random.key(2),
        features,
        loaded({**parameters, "token_bonds.bias": jnp.zeros(settings.d_pair)}),
    )
    negative_zero = {name: value.copy() for name, value in features.items()}
    negative_zero["token_bonds"][0, 0, 0, 0] = np.float32(-0.0)
    inference.predict(jax.random.key(3), negative_zero, loaded())
    inference.predict(
        jax.random.key(4),
        features,
        loaded(
            {
                "token_bonds.weight": jnp.zeros(
                    (settings.d_pair, 1), dtype=jnp.int32
                )
            }
        ),
    )
    missing = dict(features)
    missing.pop("token_bonds")
    inference.predict(jax.random.key(5), missing, loaded())
    inference.predict(
        jax.random.key(6),
        features,
        loaded(
            {
                "token_bonds.weight": jnp.ones(
                    (settings.d_pair, 1), dtype=jnp.float32
                )
            }
        ),
    )
    inference.predict(
        jax.random.key(7),
        features,
        loaded(
            model_settings=dataclasses.replace(settings, trunk_dtype="float32")
        ),
    )

    assert routed == [True, False, False, False, False, False, True, False]


def test_verified_zero_bonds_are_removed_only_from_the_model_bound_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = build_features([("AG", "A", 0, 0)])
    settings = structure_model.ModelSettings()
    parameters = {
        "token_bonds.weight": jnp.ones(
            (settings.d_pair, 1), dtype=jnp.bfloat16
        )
    }
    model = SimpleNamespace(
        settings=settings,
        parameters=parameters,
        esmc_parameters=None,
        esmc_settings=None,
    )
    transferred: list[dict[str, jax.Array]] = []

    def fake_compiled_predict(*identity):
        def run(*args):
            transferred.append(args[1])
            return {}

        return run

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)

    inference.predict(jax.random.key(0), features, model)
    assert "token_bonds" not in transferred[-1]
    assert "token_bonds" in features

    nonzero = {name: value.copy() for name, value in features.items()}
    nonzero["token_bonds"][0, 0, 0, 0] = 1.0
    inference.predict(jax.random.key(1), nonzero, model)
    assert "token_bonds" in transferred[-1]


def test_model_bound_zero_bond_compaction_removes_quadratic_storage() -> None:
    n_tokens = 1003
    features, _ = _contract(n_tokens=n_tokens)

    compact = inference._model_bound_features(
        features, compact_token_bond_encoding=True
    )
    generic = inference._model_bound_features(
        features, compact_token_bond_encoding=False
    )

    assert "token_bonds" not in compact
    assert generic["token_bonds"] is features["token_bonds"]
    assert features["token_bonds"].nbytes == 4 * n_tokens * n_tokens


def test_compact_encoding_does_not_need_the_removed_dense_leaf() -> None:
    weight = jnp.asarray([[1.0], [-1.0], [0.0], [-0.0]], dtype=jnp.bfloat16)
    params = {"token_bonds.weight": weight}
    bonds = jnp.zeros((1, 3, 3, 1), dtype=jnp.float32)

    dense = structure_model._token_bonds_encoding(
        bonds, params, jnp.dtype(jnp.bfloat16)
    )
    compact = structure_model._token_bonds_encoding(
        None,
        params,
        jnp.dtype(jnp.bfloat16),
        compact_token_bond_encoding=True,
    )

    assert np.asarray(dense).tobytes() == np.broadcast_to(
        np.asarray(compact), np.asarray(dense).shape
    ).tobytes()
    with pytest.raises(KeyError, match="token_bonds"):
        structure_model._token_bonds_encoding(
            None, params, jnp.dtype(jnp.bfloat16)
        )


def test_zero_contract_is_part_of_the_compiled_factory_cache_key() -> None:
    settings = dataclasses.replace(
        structure_model.ModelSettings(), trunk_n_layers=0, coda_n_layers=0
    )
    inference.compiled_predict.cache_clear()
    try:
        generic = inference.compiled_predict(
            settings, 1, False, 1, (), False, False, False
        )
        specialized = inference.compiled_predict(
            settings, 1, False, 1, (), False, False, True
        )
        assert generic is not specialized
        assert inference.compiled_predict.cache_info().currsize == 2
    finally:
        inference.compiled_predict.cache_clear()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_compact_verified_projection_is_bitwise_exact(dtype) -> None:
    rng = np.random.default_rng(9)
    base = jnp.asarray(rng.normal(size=(1, 7, 7, 8)), dtype=dtype)
    bonds = np.zeros((1, 7, 7, 1), dtype=np.float32)
    weight = jnp.asarray(rng.normal(size=(8, 1)), dtype=dtype)
    params = {"token_bonds.weight": weight}

    encoding = structure_model._token_bonds_encoding(
        jnp.asarray(bonds), params, jnp.dtype(dtype)
    )
    compact = structure_model._token_bonds_encoding(
        jnp.asarray(bonds),
        params,
        jnp.dtype(dtype),
        compact_token_bond_encoding=True,
    )

    assert encoding is not None
    assert compact.shape == (8,)
    assert np.asarray(base + encoding).tobytes() == np.asarray(
        base + compact
    ).tobytes()


@pytest.mark.parametrize(
    ("source_dtype", "compute_dtype"),
    [
        (jnp.float32, jnp.float32),
        (jnp.float32, jnp.bfloat16),
        (jnp.bfloat16, jnp.bfloat16),
    ],
)
def test_compact_signed_zeros_match_supported_post_cast_finite_edges(
    source_dtype, compute_dtype
) -> None:
    finfo = jnp.finfo(compute_dtype)
    values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        float(finfo.tiny),
        -float(finfo.tiny),
        float(finfo.max),
        -float(finfo.max),
        1e-40,
        -1e-40,
    ]
    weight = jnp.asarray(values, dtype=source_dtype).reshape(-1, 1)
    bonds = jnp.zeros((1, 2, 2, 1), dtype=jnp.float32)
    params = {"token_bonds.weight": weight}
    features = {
        "token_attention_mask": np.ones((1, 2), dtype=bool),
        "token_bonds": np.zeros((1, 2, 2, 1), dtype=np.float32),
    }
    assert inference._has_compact_token_bond_encoding(
        features,
        params,
        pair_width=len(values),
        compute_dtype=compute_dtype,
    )
    cast = structure_model._cast(
        params, structure_model.TRUNK_PREFIXES, compute_dtype
    )

    generic = structure_model._token_bonds_encoding(
        bonds, cast, jnp.dtype(compute_dtype)
    )
    compact = structure_model._token_bonds_encoding(
        bonds,
        cast,
        jnp.dtype(compute_dtype),
        compact_token_bond_encoding=True,
    )
    expanded = np.broadcast_to(np.asarray(compact), np.asarray(generic).shape)

    assert np.asarray(generic).tobytes() == expanded.tobytes()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_compact_signed_zeros_preserve_both_left_associated_adds(dtype) -> None:
    weight = jnp.asarray([[1.0], [-1.0], [0.0], [-0.0]], dtype=dtype)
    params = {"token_bonds.weight": weight}
    bonds = jnp.zeros((1, 2, 2, 1), dtype=jnp.float32)
    generic = structure_model._token_bonds_encoding(
        bonds, params, jnp.dtype(dtype)
    )
    compact = structure_model._token_bonds_encoding(
        bonds,
        params,
        jnp.dtype(dtype),
        compact_token_bond_encoding=True,
    )
    row_signs = jnp.asarray([0.0, -0.0], dtype=dtype).reshape(1, 2, 1, 1)
    column_signs = jnp.asarray([-0.0, 0.0], dtype=dtype).reshape(1, 1, 2, 1)
    z_init = jnp.broadcast_to(row_signs, (1, 2, 2, 4))
    confidence_pair = jnp.broadcast_to(-row_signs, (1, 2, 2, 4))
    relative = jnp.broadcast_to(column_signs, (1, 2, 2, 4))

    # Spell both call sites separately and keep the original left association:
    # `(base + relative) + token_bonds_encoding`.
    old_z = (z_init + relative) + generic
    compact_z = (z_init + relative) + compact
    old_confidence = (confidence_pair + relative) + generic
    compact_confidence = (confidence_pair + relative) + compact

    assert np.asarray(old_z).tobytes() == np.asarray(compact_z).tobytes()
    assert np.asarray(old_confidence).tobytes() == np.asarray(
        compact_confidence
    ).tobytes()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_negative_zero_falls_back_when_the_dense_add_changes_its_sign(dtype) -> None:
    features = {
        "token_attention_mask": np.ones((1, 1), dtype=bool),
        "token_bonds": np.full((1, 1, 1, 1), -0.0, dtype=np.float32),
    }
    weight = jnp.asarray([[1.0], [-1.0]], dtype=dtype)
    parameters = {"token_bonds.weight": weight}
    assert not inference._has_compact_token_bond_encoding(
        features, parameters, pair_width=2, compute_dtype=dtype
    )

    base = jnp.full((1, 1, 1, 2), -0.0, dtype=dtype)
    encoding = structure_model._token_bonds_encoding(
        jnp.asarray(features["token_bonds"]), parameters, jnp.dtype(dtype)
    )
    assert encoding is not None
    dense = np.asarray(base + encoding)
    unadded = np.asarray(base)
    np.testing.assert_array_equal(dense, unadded)
    np.testing.assert_array_equal(np.signbit(dense), [[[[True, False]]]])
    np.testing.assert_array_equal(np.signbit(unadded), [[[[True, True]]]])


def test_verified_compact_graph_omits_the_token_bond_dot() -> None:
    base = jnp.zeros((1, 8, 8, 16), dtype=jnp.bfloat16)
    bonds = jnp.zeros((1, 8, 8, 1), dtype=jnp.float32)
    weight = jnp.ones((16, 1), dtype=jnp.bfloat16)

    def stage(base, bonds, weight, *, verified):
        encoding = structure_model._token_bonds_encoding(
            bonds,
            {"token_bonds.weight": weight},
            base.dtype,
            compact_token_bond_encoding=verified,
        )
        return base + encoding

    generic = jax.jit(lambda x, b, w: stage(x, b, w, verified=False)).lower(
        base, bonds, weight
    )
    specialized = jax.jit(
        lambda x, b, w: stage(x, b, w, verified=True)
    ).lower(base, bonds, weight)

    assert generic.as_text().count("stablehlo.dot_general") == 1
    assert specialized.as_text().count("stablehlo.dot_general") == 0
