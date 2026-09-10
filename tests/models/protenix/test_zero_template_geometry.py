"""Dropping all-zero Protenix template geometry from the device arguments.

A query with no template hits still carries the four quadratic geometry
tensors -- ``template_distogram``, ``template_unit_vector`` and the two pair
masks -- and every one of them is bitwise ``+0.0``.  At 4,100 tokens over the
two ``dedup_templates`` survivors that is 5.9 GB of arguments the host copies
onto the device so the trunk can multiply it by a mask and get zero back.

``template_aatype`` is *not* part of that: the survivors carry restype 31 and
restype 0, which give different embeddings, so the template tower is nonzero
even for a template-free query.  Only the geometry is dropped, and the model
rebuilds it from a dynamic scalar zero so ``0 * NaN`` still yields ``NaN``
exactly as the dense path did.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.bridge.torch_mapping import (
    map_template_embedder_state_dict,
)
from foldjax.models.protenix.data.template_features import (
    ZERO_TEMPLATE_GEOMETRY_FIELDS,
    ZERO_TEMPLATE_GEOMETRY_MARKER,
    compact_zero_template_geometry,
    dedup_templates,
    has_compact_zero_template_geometry,
)
from foldjax.models.protenix.models import model as model_impl
from foldjax.models.protenix.models.trunk_blocks.template import (
    template_embedder,
    template_pair_features,
)
from tests.models.protenix.test_template import _template_state

N_TOKEN, BINS, N_ATOM = 3, 39, 24


def _zero_templates(n_template: int = 2) -> dict[str, np.ndarray]:
    """What the featurizer produces for a query with no template hits."""
    features = {
        "template_aatype": np.zeros((n_template, N_TOKEN), np.int32),
        "template_pseudo_beta_mask": np.zeros(
            (n_template, N_TOKEN, N_TOKEN), np.float32
        ),
        "template_distogram": np.zeros(
            (n_template, N_TOKEN, N_TOKEN, BINS), np.float32
        ),
        "template_unit_vector": np.zeros((n_template, N_TOKEN, N_TOKEN, 3), np.float32),
        "template_backbone_frame_mask": np.zeros(
            (n_template, N_TOKEN, N_TOKEN), np.float32
        ),
    }
    features["template_aatype"][0] = 31
    return features


def _asym_id() -> jnp.ndarray:
    """Two chains, so the multichain mask is not all ones."""
    return jnp.asarray([0, 0, 1], dtype=jnp.int32)


def _pair_mask() -> jnp.ndarray:
    return jnp.asarray(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=jnp.float32,
    )


def _as_model_features(features: dict[str, np.ndarray]) -> dict[str, object]:
    out: dict[str, object] = {
        name: jnp.asarray(value) for name, value in features.items()
    }
    out["asym_id"] = _asym_id()
    return out


# --------------------------------------------------------------------------
# host boundary
# --------------------------------------------------------------------------


def test_a_template_free_query_drops_the_four_geometry_leaves() -> None:
    compact = compact_zero_template_geometry(_zero_templates())

    assert has_compact_zero_template_geometry(compact)
    for name in ZERO_TEMPLATE_GEOMETRY_FIELDS:
        assert name not in compact, name
    # The restypes discriminate the survivors and must survive.
    np.testing.assert_array_equal(compact["template_aatype"][0], 31)
    marker = np.asarray(compact[ZERO_TEMPLATE_GEOMETRY_MARKER])
    assert marker.shape == () and marker.dtype == np.dtype(np.float32)
    assert marker.view(np.uint32) == 0


def test_the_marker_is_four_bytes_beside_the_retained_restypes() -> None:
    dense = _zero_templates()
    compact = compact_zero_template_geometry(dense)

    dropped = sum(dense[name].nbytes for name in ZERO_TEMPLATE_GEOMETRY_FIELDS)
    added = np.asarray(compact[ZERO_TEMPLATE_GEOMETRY_MARKER]).nbytes
    assert added == 4
    # 2 rows x (39 + 3 + 1 + 1) channels x 4 B x N_token^2.
    assert dropped == 352 * N_TOKEN * N_TOKEN
    assert dropped > added


def test_the_deduplicated_survivors_are_what_gets_compacted() -> None:
    """The wiring order: deduplicate first, then compact what is left."""
    padded = {
        "template_aatype": np.zeros((4, N_TOKEN), np.int32),
        "template_atom_positions": np.zeros((4, N_TOKEN, N_ATOM, 3), np.float32),
        "template_atom_mask": np.zeros((4, N_TOKEN, N_ATOM), bool),
        "template_pseudo_beta_mask": np.zeros((4, N_TOKEN, N_TOKEN), np.float32),
        "template_distogram": np.zeros((4, N_TOKEN, N_TOKEN, BINS), np.float32),
        "template_unit_vector": np.zeros((4, N_TOKEN, N_TOKEN, 3), np.float32),
        "template_backbone_frame_mask": np.zeros((4, N_TOKEN, N_TOKEN), np.float32),
    }
    padded["template_aatype"][0] = 31

    compact = compact_zero_template_geometry(dedup_templates(padded))

    assert has_compact_zero_template_geometry(compact)
    assert compact["template_aatype"].shape == (2, N_TOKEN)
    np.testing.assert_array_equal(compact["template_multiplicity"], [1.0, 3.0])


@pytest.mark.parametrize("name", sorted(ZERO_TEMPLATE_GEOMETRY_FIELDS))
def test_any_nonzero_geometry_keeps_the_dense_representation(name: str) -> None:
    dense = _zero_templates()
    dense[name].reshape(-1)[0] = 1.0

    compact = compact_zero_template_geometry(dense)

    assert not has_compact_zero_template_geometry(compact)
    assert ZERO_TEMPLATE_GEOMETRY_MARKER not in compact
    for field in ZERO_TEMPLATE_GEOMETRY_FIELDS:
        np.testing.assert_array_equal(compact[field], dense[field])


@pytest.mark.parametrize("name", sorted(ZERO_TEMPLATE_GEOMETRY_FIELDS))
def test_negative_zero_is_not_the_featurizer_zero(name: str) -> None:
    """``-0.0 == 0.0`` compares equal; the raw word does not."""
    dense = _zero_templates()
    dense[name].reshape(-1)[0] = -0.0

    compact = compact_zero_template_geometry(dense)

    assert not has_compact_zero_template_geometry(compact)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_geometry_keeps_the_dense_representation(value: float) -> None:
    dense = _zero_templates()
    dense["template_unit_vector"].reshape(-1)[0] = value

    assert not has_compact_zero_template_geometry(compact_zero_template_geometry(dense))


def test_a_wider_float_is_not_the_released_storage() -> None:
    dense = _zero_templates()
    dense["template_distogram"] = dense["template_distogram"].astype(np.float64)

    assert not has_compact_zero_template_geometry(compact_zero_template_geometry(dense))


def test_shape_drift_keeps_the_dense_representation() -> None:
    dense = _zero_templates()
    dense["template_distogram"] = np.zeros((2, N_TOKEN, N_TOKEN, 38), np.float32)

    assert not has_compact_zero_template_geometry(compact_zero_template_geometry(dense))


def test_features_without_templates_are_returned_unchanged() -> None:
    features = {"asym_id": np.zeros((3,), np.int32)}

    assert compact_zero_template_geometry(features) == features


def test_a_missing_geometry_field_keeps_what_is_there() -> None:
    dense = _zero_templates()
    del dense["template_unit_vector"]

    compact = compact_zero_template_geometry(dense)

    assert ZERO_TEMPLATE_GEOMETRY_MARKER not in compact
    assert "template_distogram" in compact


def test_a_stale_marker_beside_dense_geometry_is_dropped_not_trusted() -> None:
    """Provenance is produced here, never accepted from a caller."""
    dense = _zero_templates()
    dense["template_distogram"][0, 0, 0, 0] = 1.0
    dense[ZERO_TEMPLATE_GEOMETRY_MARKER] = np.asarray(7.0, np.float32)

    compact = compact_zero_template_geometry(dense)

    assert ZERO_TEMPLATE_GEOMETRY_MARKER not in compact
    np.testing.assert_array_equal(
        compact["template_distogram"], dense["template_distogram"]
    )


def test_a_stale_marker_is_replaced_by_a_freshly_emitted_zero() -> None:
    dense = _zero_templates()
    dense[ZERO_TEMPLATE_GEOMETRY_MARKER] = np.asarray(7.0, np.float32)

    compact = compact_zero_template_geometry(dense)

    assert np.asarray(compact[ZERO_TEMPLATE_GEOMETRY_MARKER]) == 0.0


def test_the_predicate_rejects_a_marker_carried_beside_dense_geometry() -> None:
    dense = dict(_zero_templates())
    dense[ZERO_TEMPLATE_GEOMETRY_MARKER] = np.zeros((), np.float32)

    assert not has_compact_zero_template_geometry(dense)


# --------------------------------------------------------------------------
# bit-exactness
# --------------------------------------------------------------------------


def _pair_outputs(features, pair_mask, n_template: int = 2):
    return [
        np.asarray(template_pair_features(features, index, pair_mask))
        for index in range(n_template)
    ]


def test_the_compact_pair_features_are_bitwise_the_dense_ones() -> None:
    dense = _as_model_features(_zero_templates())
    compact = _as_model_features(compact_zero_template_geometry(_zero_templates()))
    pair_mask = _pair_mask()

    for expected, actual in zip(
        _pair_outputs(dense, pair_mask),
        _pair_outputs(compact, pair_mask),
        strict=True,
    ):
        assert expected.shape == (N_TOKEN, N_TOKEN, 108)
        assert np.array_equal(expected, actual)


def test_the_compact_pair_features_are_bitwise_equal_without_a_pair_mask() -> None:
    dense = _as_model_features(_zero_templates())
    compact = _as_model_features(compact_zero_template_geometry(_zero_templates()))

    for expected, actual in zip(
        _pair_outputs(dense, None), _pair_outputs(compact, None), strict=True
    ):
        assert np.array_equal(expected, actual)


def _assert_the_comparison_has_power(z, pair_mask, params, expected) -> None:
    """This embedder can be blind to its geometry; prove this draw is not.

    Random template weights and the ``max(u, 0)`` before ``linear_u`` can leave
    the tower emitting an all-zero tensor that is equal on every path,
    including the wrong ones. Move one geometry element and require the output
    to move with it before believing an equality.
    """
    assert np.any(expected != 0.0)
    perturbed = _zero_templates()
    perturbed["template_distogram"][0, 0, 0, 0] = 1.0
    moved = np.asarray(
        template_embedder(_as_model_features(perturbed), z, pair_mask, params)
    )
    assert not np.array_equal(expected, moved)


def test_the_compact_embedder_output_is_bitwise_the_dense_one() -> None:
    rng = np.random.default_rng(2)
    params = map_template_embedder_state_dict(_template_state(rng))
    z = jnp.asarray(rng.normal(size=(N_TOKEN, N_TOKEN, 4)).astype(np.float32))
    dense = _as_model_features(_zero_templates())
    compact = _as_model_features(compact_zero_template_geometry(_zero_templates()))
    pair_mask = _pair_mask()

    expected = np.asarray(template_embedder(dense, z, pair_mask, params))
    actual = np.asarray(template_embedder(compact, z, pair_mask, params))

    _assert_the_comparison_has_power(z, pair_mask, params, expected)
    assert np.array_equal(expected, actual)


def test_the_compact_embedder_is_bitwise_equal_under_jit() -> None:
    """Two pytrees are two traces, so this is not the same program twice."""
    rng = np.random.default_rng(12)
    params = map_template_embedder_state_dict(_template_state(rng))
    z = jnp.asarray(rng.normal(size=(N_TOKEN, N_TOKEN, 4)).astype(np.float32))
    dense = _as_model_features(_zero_templates())
    compact = _as_model_features(compact_zero_template_geometry(_zero_templates()))
    pair_mask = _pair_mask()

    run = jax.jit(lambda features, pair: template_embedder(features, z, pair, params))
    expected = np.asarray(run(dense, pair_mask))
    actual = np.asarray(run(compact, pair_mask))

    _assert_the_comparison_has_power(z, pair_mask, params, expected)
    assert np.array_equal(expected, actual)


def test_a_multiplicity_weighted_sum_stays_bitwise_equal() -> None:
    rng = np.random.default_rng(13)
    params = map_template_embedder_state_dict(_template_state(rng))
    z = jnp.asarray(rng.normal(size=(N_TOKEN, N_TOKEN, 4)).astype(np.float32))
    dense = _zero_templates()
    dense["template_multiplicity"] = np.asarray([1.0, 3.0], np.float32)
    compact = compact_zero_template_geometry(dense)
    pair_mask = _pair_mask()

    expected = np.asarray(
        template_embedder(_as_model_features(dense), z, pair_mask, params)
    )
    actual = np.asarray(
        template_embedder(_as_model_features(compact), z, pair_mask, params)
    )

    _assert_the_comparison_has_power(z, pair_mask, params, expected)
    assert np.array_equal(expected, actual)


def test_the_rebuilt_zero_keeps_the_dense_non_finite_mask_behaviour() -> None:
    """``0 * NaN`` is ``NaN``; a folded constant zero would give ``0``."""
    dense = _as_model_features(_zero_templates())
    compact = _as_model_features(compact_zero_template_geometry(_zero_templates()))
    pair_mask = _pair_mask().at[0, 1].set(jnp.nan)

    expected, actual = (
        np.asarray(template_pair_features(features, 0, pair_mask))
        for features in (dense, compact)
    )

    assert np.isnan(expected[0, 1, 0])
    assert np.array_equal(expected, actual, equal_nan=True)


def test_nonzero_templates_never_take_the_compact_path() -> None:
    """Real geometry reaches the trunk unchanged, marker or no marker."""
    rng = np.random.default_rng(24)
    params = map_template_embedder_state_dict(_template_state(rng))
    z = jnp.asarray(rng.normal(size=(N_TOKEN, N_TOKEN, 4)).astype(np.float32))
    dense = _zero_templates()
    dense["template_distogram"][:] = rng.normal(
        size=dense["template_distogram"].shape
    ).astype(np.float32)
    dense["template_unit_vector"][:] = rng.normal(
        size=dense["template_unit_vector"].shape
    ).astype(np.float32)
    dense["template_pseudo_beta_mask"][:] = 1.0
    dense["template_backbone_frame_mask"][:] = 1.0
    pair_mask = _pair_mask()

    compact = compact_zero_template_geometry(dense)
    assert not has_compact_zero_template_geometry(compact)

    expected = np.asarray(
        template_embedder(_as_model_features(dense), z, pair_mask, params)
    )
    actual = np.asarray(
        template_embedder(_as_model_features(compact), z, pair_mask, params)
    )
    assert np.array_equal(expected, actual)
    # A real template moves the tower away from the all-zero-geometry value.
    zero_geometry = np.asarray(
        template_embedder(
            _as_model_features(compact_zero_template_geometry(_zero_templates())),
            z,
            pair_mask,
            params,
        )
    )
    assert not np.array_equal(expected, zero_geometry)


def test_a_marker_beside_dense_geometry_still_uses_that_geometry() -> None:
    """The marker is provenance for one representation, not a switch."""
    rng = np.random.default_rng(15)
    features = _as_model_features(_zero_templates())
    features["template_distogram"] = jnp.asarray(
        rng.normal(size=(2, N_TOKEN, N_TOKEN, BINS)).astype(np.float32)
    )
    with_marker = dict(features)
    with_marker[ZERO_TEMPLATE_GEOMETRY_MARKER] = jnp.zeros((), jnp.float32)
    pair_mask = _pair_mask()

    assert np.array_equal(
        np.asarray(template_pair_features(features, 0, pair_mask)),
        np.asarray(template_pair_features(with_marker, 0, pair_mask)),
    )


def test_a_nonzero_marker_is_rejected_before_it_becomes_geometry() -> None:
    features = _as_model_features(compact_zero_template_geometry(_zero_templates()))
    features[ZERO_TEMPLATE_GEOMETRY_MARKER] = jnp.asarray(1.0, jnp.float32)

    with pytest.raises(ValueError, match="zero"):
        template_pair_features(features, 0, _pair_mask())


def test_a_misshapen_marker_is_rejected() -> None:
    features = _as_model_features(compact_zero_template_geometry(_zero_templates()))
    features[ZERO_TEMPLATE_GEOMETRY_MARKER] = jnp.zeros((2,), jnp.float32)

    with pytest.raises(ValueError, match="scalar"):
        template_pair_features(features, 0, _pair_mask())


# --------------------------------------------------------------------------
# what reaches the device
# --------------------------------------------------------------------------


def _capture_model_features(monkeypatch, features, **kwargs) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_compiled(input_feature_dict, *_args, **_kwargs):
        captured.update(input_feature_dict)
        return {}

    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", fake_compiled)
    model_impl.protenix_infer_compiled(
        {
            "asym_id": np.asarray([0, 0, 1], dtype=np.int32),
            "token_padding_mask": np.asarray([1, 1, 1], dtype=np.float32),
            **features,
        },
        (),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        **kwargs,
    )
    return captured


def test_the_compact_geometry_never_reaches_the_device(monkeypatch) -> None:
    captured = _capture_model_features(
        monkeypatch, compact_zero_template_geometry(_zero_templates())
    )

    for name in ZERO_TEMPLATE_GEOMETRY_FIELDS:
        assert name not in captured, name
    assert ZERO_TEMPLATE_GEOMETRY_MARKER in captured
    assert "template_aatype" in captured


def test_dense_geometry_still_reaches_the_device(monkeypatch) -> None:
    captured = _capture_model_features(monkeypatch, _zero_templates())

    for name in ZERO_TEMPLATE_GEOMETRY_FIELDS:
        assert name in captured, name
    assert ZERO_TEMPLATE_GEOMETRY_MARKER not in captured


def test_the_padded_graph_allowlist_keeps_the_marker(monkeypatch) -> None:
    captured = _capture_model_features(
        monkeypatch,
        compact_zero_template_geometry(_zero_templates()),
        padded_generated_schema=True,
    )

    assert ZERO_TEMPLATE_GEOMETRY_MARKER in captured
    assert "template_aatype" in captured


def test_the_compiled_host_gate_rejects_a_nonzero_marker(monkeypatch) -> None:
    called = False

    def unexpected_compiled(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", unexpected_compiled)
    features = compact_zero_template_geometry(_zero_templates())
    features[ZERO_TEMPLATE_GEOMETRY_MARKER] = np.asarray(1.0, np.float32)

    with pytest.raises(ValueError, match="zero"):
        model_impl.protenix_infer_compiled(
            {
                "asym_id": np.asarray([0, 0, 1], dtype=np.int32),
                **features,
            },
            (),
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        )
    assert called is False


def _run_cli(tmp_path, monkeypatch, *, extra: list[str]) -> dict[str, object]:
    from foldjax.models.protenix.cli import predict as predict_impl

    input_path = tmp_path / "input.json"
    input_path.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "A"}}]}]',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_predict(_params, features, **_kwargs):
        captured.update(features)
        n_atom = len(features["atom_to_token_idx"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    predict_impl.main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(tmp_path / "unused-weights.jax"),
            "--input-json",
            str(input_path),
            "--out",
            str(tmp_path / "unused-output.npz"),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--trunk-dtype",
            "fp32",
            "--prewarm-only",
            "--cpu-only",
            "--no-compile-cache",
            *extra,
        ],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )
    return captured


def test_the_cli_compacts_a_template_free_query(tmp_path, monkeypatch) -> None:
    captured = _run_cli(tmp_path, monkeypatch, extra=[])

    assert has_compact_zero_template_geometry(captured)
    assert "template_aatype" in captured
    np.testing.assert_array_equal(captured["template_multiplicity"], [1.0, 3.0])


def test_the_cli_compacts_after_serving_padding(tmp_path, monkeypatch) -> None:
    captured = _run_cli(
        tmp_path,
        monkeypatch,
        extra=[
            "--padding",
            "--pad-tokens",
            "8",
            "--pad-atoms",
            "8",
            "--pad-msa",
            "4",
            "--pad-templates",
            "4",
        ],
    )

    assert has_compact_zero_template_geometry(captured)
    assert np.asarray(captured["template_aatype"]).shape[-1] == 8
