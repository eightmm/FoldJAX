"""Collapsing a template axis whose rows are all the same template.

A query with no templates is featurized as the released fixed-width axis of four
identical empty ones, and the template stack embeds every one of them before
averaging. These cover the host-side reduction that keeps one of them, and check
that the value the model computes survives it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.data import (
    collapse_identical_templates,
    compact_zero_template_pair_features,
)
from foldjax.models.openfold3.data.featurize import _ZERO_TEMPLATE_PAIR_MARKER
from foldjax.models.openfold3.models.attention import AttentionParams
from foldjax.models.openfold3.models.pair_block import PairBlockParams
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
)
from foldjax.models.openfold3.models.template_module import (
    TemplateEmbedderParams,
    TemplatePairEmbedderParams,
    TemplatePairStackParams,
    template_embedder,
    template_pair_embedder,
)
from foldjax.models.openfold3.models.triangle import TriangleMultiplicationParams
from foldjax.models.openfold3.models.triangle_attention import TriangleAttentionParams
from tests.models.cp_probe_env import inherited_environment

N_TOKEN, N_TEMPLATE = 5, 4
DISTOGRAM_BINS, RESTYPE = 39, 32


def _empty_templates(n_template: int = N_TEMPLATE) -> dict[str, np.ndarray]:
    """The shape `_empty_template_features` produces for a template-free query."""
    restype = np.zeros((1, n_template, N_TOKEN, RESTYPE), dtype=np.int32)
    restype[..., -1] = 1  # every position is the same gap one-hot
    return {
        "template_restype": restype,
        "template_pseudo_beta_mask": np.zeros(
            (1, n_template, N_TOKEN), dtype=np.float32
        ),
        "template_backbone_frame_mask": np.zeros(
            (1, n_template, N_TOKEN), dtype=np.float32
        ),
        "template_distogram": np.zeros(
            (1, n_template, N_TOKEN, N_TOKEN, DISTOGRAM_BINS), dtype=np.float32
        ),
        "template_unit_vector": np.zeros(
            (1, n_template, N_TOKEN, N_TOKEN, 3), dtype=np.float32
        ),
    }


def _template_rows(features: dict[str, np.ndarray]) -> int:
    return features["template_distogram"].shape[1]


def test_identical_templates_collapse_to_one_row() -> None:
    features = _empty_templates()
    collapsed = collapse_identical_templates(features)

    assert _template_rows(collapsed) == 1
    for name, array in collapsed.items():
        assert array.shape[1] == 1, name
        np.testing.assert_array_equal(array[:, 0], features[name][:, 0])


def test_distinct_templates_are_left_alone() -> None:
    features = _empty_templates()
    # One real contact is enough: the rows are no longer interchangeable.
    features["template_distogram"][0, 2, 1, 3, 7] = 1.0

    collapsed = collapse_identical_templates(features)

    assert _template_rows(collapsed) == N_TEMPLATE
    for name, array in collapsed.items():
        np.testing.assert_array_equal(array, features[name])


def test_a_single_template_is_returned_unchanged() -> None:
    features = _empty_templates(n_template=1)
    assert _template_rows(collapse_identical_templates(features)) == 1


def test_features_without_templates_pass_through() -> None:
    features = {"token_mask": np.ones((1, N_TOKEN), dtype=np.float32)}
    collapsed = collapse_identical_templates(features)
    assert list(collapsed) == ["token_mask"]


def test_a_padding_mask_that_deactivates_rows_blocks_the_collapse() -> None:
    """Padded storage rows are excluded from the average, so they are not spares."""
    features = _empty_templates()
    padding = np.ones((1, N_TEMPLATE), dtype=np.float32)
    padding[0, -1] = 0.0
    features["template_padding_mask"] = padding

    assert _template_rows(collapse_identical_templates(features)) == N_TEMPLATE


def test_an_all_active_padding_mask_collapses_with_the_rest() -> None:
    features = _empty_templates()
    features["template_padding_mask"] = np.ones((1, N_TEMPLATE), dtype=np.float32)

    collapsed = collapse_identical_templates(features)

    assert _template_rows(collapsed) == 1
    assert collapsed["template_padding_mask"].shape == (1, 1)


def test_disagreeing_template_row_counts_are_rejected() -> None:
    features = _empty_templates()
    features["template_restype"] = features["template_restype"][:, :2]
    with pytest.raises(ValueError, match="row count"):
        collapse_identical_templates(features)


_ZERO_PAIR_FEATURES = (
    "template_distogram",
    "template_pseudo_beta_mask",
    "template_unit_vector",
    "template_backbone_frame_mask",
)


def test_exact_positive_zero_pair_features_compact_to_one_scalar() -> None:
    dense = collapse_identical_templates(_empty_templates())
    compact = compact_zero_template_pair_features(dense)

    assert set(compact) == {"template_restype", _ZERO_TEMPLATE_PAIR_MARKER}
    assert set(dense).issuperset(_ZERO_PAIR_FEATURES)
    marker = compact[_ZERO_TEMPLATE_PAIR_MARKER]
    assert marker.shape == () and marker.dtype == np.dtype(np.float32)
    assert marker.view(np.uint32) == 0


@pytest.mark.parametrize("name", _ZERO_PAIR_FEATURES)
@pytest.mark.parametrize("value", [1.0, np.nan, np.inf, -0.0])
def test_compaction_requires_every_pair_value_to_be_exact_positive_zero(
    name: str, value: float
) -> None:
    features = _empty_templates(n_template=1)
    features[name].reshape(-1)[0] = value

    compact = compact_zero_template_pair_features(features)

    assert _ZERO_TEMPLATE_PAIR_MARKER not in compact
    assert set(compact) == set(features)
    for feature_name in _ZERO_PAIR_FEATURES:
        assert compact[feature_name] is features[feature_name]


def test_compaction_rejects_wrong_dtype_and_untrusted_marker() -> None:
    features = _empty_templates(n_template=1)
    features["template_unit_vector"] = features["template_unit_vector"].astype(
        np.float64
    )
    features[_ZERO_TEMPLATE_PAIR_MARKER] = np.zeros((), dtype=np.float32)

    compact = compact_zero_template_pair_features(features)

    assert _ZERO_TEMPLATE_PAIR_MARKER not in compact
    assert set(compact) == set(features).difference({_ZERO_TEMPLATE_PAIR_MARKER})


@pytest.mark.parametrize("malformation", ["missing", "shape"])
def test_malformed_direct_inputs_retain_the_dense_mapping(
    malformation: str,
) -> None:
    features = _empty_templates(n_template=1)
    if malformation == "missing":
        del features["template_unit_vector"]
    else:
        features["template_unit_vector"] = features["template_unit_vector"][
            ..., :-1, :, :
        ]

    compact = compact_zero_template_pair_features(features)

    assert _ZERO_TEMPLATE_PAIR_MARKER not in compact
    assert set(compact) == set(features)


def _linear(key: jax.Array, out_features: int, in_features: int) -> LinearParams:
    return LinearParams(
        weight=jax.random.normal(key, (out_features, in_features)) * 0.1
    )


def _layer_norm(channels: int) -> LayerNormParams:
    return LayerNormParams(
        weight=jnp.ones((channels,)), bias=jnp.zeros((channels,))
    )


def _pair_block(
    keys: Iterator[jax.Array], channels: int, heads: int
) -> PairBlockParams:
    """One template pair block at toy widths, with the released layout."""

    def tri_mul() -> TriangleMultiplicationParams:
        return TriangleMultiplicationParams(
            layer_norm_in=_layer_norm(channels),
            layer_norm_out=_layer_norm(channels),
            linear_a_p=_linear(next(keys), channels, channels),
            linear_a_g=_linear(next(keys), channels, channels),
            linear_b_p=_linear(next(keys), channels, channels),
            linear_b_g=_linear(next(keys), channels, channels),
            linear_g=_linear(next(keys), channels, channels),
            linear_z=_linear(next(keys), channels, channels),
        )

    def tri_att() -> TriangleAttentionParams:
        return TriangleAttentionParams(
            layer_norm=_layer_norm(channels),
            linear_z=_linear(next(keys), heads, channels),
            mha=AttentionParams(
                linear_q=_linear(next(keys), channels, channels),
                linear_k=_linear(next(keys), channels, channels),
                linear_v=_linear(next(keys), channels, channels),
                linear_o=_linear(next(keys), channels, channels),
                linear_g=_linear(next(keys), channels, channels),
            ),
        )

    return PairBlockParams(
        tri_mul_out=tri_mul(),
        tri_mul_in=tri_mul(),
        tri_att_start=tri_att(),
        tri_att_end=tri_att(),
        pair_transition=SwiGLUTransitionParams(
            layer_norm=_layer_norm(channels),
            swiglu=SwiGLUParams(
                linear_a=_linear(next(keys), 2 * channels, channels),
                linear_b=_linear(next(keys), 2 * channels, channels),
            ),
            linear_out=_linear(next(keys), channels, 2 * channels),
        ),
    )


def _embedder_params(channels: int = 8, heads: int = 2) -> TemplateEmbedderParams:
    keys = iter(jax.random.split(jax.random.key(0), 128))
    return TemplateEmbedderParams(
        template_pair_embedder=TemplatePairEmbedderParams(
            dgram_linear=_linear(next(keys), channels, DISTOGRAM_BINS),
            aatype_linear_1=_linear(next(keys), channels, RESTYPE),
            aatype_linear_2=_linear(next(keys), channels, RESTYPE),
            pseudo_beta_mask_linear=_linear(next(keys), channels, 1),
            x_linear=_linear(next(keys), channels, 1),
            y_linear=_linear(next(keys), channels, 1),
            z_linear=_linear(next(keys), channels, 1),
            backbone_mask_linear=_linear(next(keys), channels, 1),
            layer_norm_z=_layer_norm(channels),
            linear_z=_linear(next(keys), channels, channels),
        ),
        template_pair_stack=TemplatePairStackParams(
            blocks=tuple(_pair_block(keys, channels, heads) for _ in range(2)),
            layer_norm=_layer_norm(channels),
        ),
        linear_t=_linear(next(keys), channels, channels),
    )


def _model_batch(source: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {
        "asym_id": jnp.zeros((1, N_TOKEN), dtype=jnp.int32),
        **{name: jnp.asarray(value) for name, value in source.items()},
    }


def test_compact_pair_projection_and_full_two_block_update_are_bitwise_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "xla")
    params = _embedder_params()
    dense = collapse_identical_templates(_empty_templates())
    compact = compact_zero_template_pair_features(dense)
    z = jax.random.normal(
        jax.random.key(7), (1, N_TOKEN, N_TOKEN, 8), dtype=jnp.float32
    )
    pair_mask = jnp.ones((1, N_TOKEN, N_TOKEN), dtype=jnp.float32)

    pair_run = jax.jit(
        lambda batch: template_pair_embedder(
            batch, z, params.template_pair_embedder
        )
    )
    full_run = jax.jit(
        lambda batch: template_embedder(
            batch,
            z,
            params,
            pair_mask=pair_mask,
            no_heads=2,
        )
    )
    dense_pair = np.asarray(pair_run(_model_batch(dense)))
    compact_pair = np.asarray(pair_run(_model_batch(compact)))
    dense_full = np.asarray(full_run(_model_batch(dense)))
    compact_full = np.asarray(full_run(_model_batch(compact)))

    assert dense_pair.tobytes() == compact_pair.tobytes()
    assert dense_full.tobytes() == compact_full.tobytes()


def test_direct_dense_batch_ignores_an_extra_private_marker() -> None:
    dense = collapse_identical_templates(_empty_templates())
    dense["template_distogram"][0, 0, 1, 3, 7] = 1.0
    marked = {**dense, _ZERO_TEMPLATE_PAIR_MARKER: np.zeros((), np.float32)}
    params = _embedder_params().template_pair_embedder
    z = jax.random.normal(
        jax.random.key(9), (1, N_TOKEN, N_TOKEN, 8), dtype=jnp.float32
    )
    run = jax.jit(lambda batch: template_pair_embedder(batch, z, params))

    expected = np.asarray(run(_model_batch(dense)))
    actual = np.asarray(run(_model_batch(marked)))

    assert expected.tobytes() == actual.tobytes()


@pytest.mark.parametrize(
    ("projection", "special"),
    [
        ("dgram_linear", np.nan),
        ("x_linear", np.inf),
        ("backbone_mask_linear", -np.inf),
    ],
)
def test_compact_projection_preserves_zero_times_nonfinite_semantics(
    projection: str, special: float
) -> None:
    dense = collapse_identical_templates(_empty_templates())
    compact = compact_zero_template_pair_features(dense)
    params = _embedder_params().template_pair_embedder
    selected = getattr(params, projection)
    selected = selected._replace(weight=selected.weight.at[0, 0].set(special))
    params = params._replace(**{projection: selected})
    z = jax.random.normal(
        jax.random.key(11), (1, N_TOKEN, N_TOKEN, 8), dtype=jnp.float32
    )
    run = jax.jit(lambda batch: template_pair_embedder(batch, z, params))

    dense_value = np.asarray(run(_model_batch(dense)))
    compact_value = np.asarray(run(_model_batch(compact)))

    assert np.isnan(dense_value).any(), "the non-finite probe is vacuous"
    np.testing.assert_array_equal(np.isnan(compact_value), np.isnan(dense_value))
    np.testing.assert_array_equal(np.isposinf(compact_value), np.isposinf(dense_value))
    np.testing.assert_array_equal(np.isneginf(compact_value), np.isneginf(dense_value))
    finite = np.isfinite(dense_value)
    assert np.array_equal(finite, np.isfinite(compact_value))
    assert (
        dense_value[finite].view(np.uint32).tobytes()
        == compact_value[finite].view(np.uint32).tobytes()
    )


def test_compact_hlo_removes_quadratic_inputs_and_pairwise_distogram_dot() -> None:
    dense = collapse_identical_templates(_empty_templates())
    compact = compact_zero_template_pair_features(dense)
    params = _embedder_params().template_pair_embedder
    z = jnp.zeros((1, N_TOKEN, N_TOKEN, 8), dtype=jnp.float32)

    def run(batch, pair, weights):
        return template_pair_embedder(batch, pair, weights)

    dense_hlo = jax.jit(run).lower(_model_batch(dense), z, params).as_text()
    compact_hlo = jax.jit(run).lower(_model_batch(compact), z, params).as_text()

    quadratic_input = f"tensor<1x1x{N_TOKEN}x{N_TOKEN}x39xf32>"
    unit_vector_input = f"tensor<1x1x{N_TOKEN}x{N_TOKEN}x3xf32>"
    mask_input = f"tensor<1x1x{N_TOKEN}xf32>"
    dense_dot = (
        f"({quadratic_input}, tensor<39x8xf32>) -> "
        f"tensor<1x1x{N_TOKEN}x{N_TOKEN}x8xf32>"
    )
    assert quadratic_input in dense_hlo
    assert unit_vector_input in dense_hlo
    assert mask_input in dense_hlo
    assert dense_dot in dense_hlo
    assert quadratic_input not in compact_hlo
    assert unit_vector_input not in compact_hlo
    assert mask_input not in compact_hlo
    assert dense_dot not in compact_hlo
    # The compact graph still performs the width-39 reduction once. This is
    # what retains 0 * NaN/Inf semantics instead of deleting the term outright.
    assert "(tensor<39xf32>, tensor<39x8xf32>) -> tensor<8xf32>" in compact_hlo
    assert len(compact_hlo) < len(dense_hlo)


def test_the_model_computes_the_same_update_from_the_collapsed_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four copies averaged, against the one copy that survives the collapse.

    The two are the same value up to the rounding of the sum -- ``4x`` divided by
    four against ``x`` -- and they take different branches getting there: four
    rows scan the stack, one row does not.

    The tolerance is absolute and exactly one float32 machine epsilon, not a
    comfortable ``rtol``. CPU lowerings have put the largest difference at one
    epsilon (two representable values below 1.0), so this still confines the
    change to rounding instead of permitting magnitude-scaled drift.
    """
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "xla")
    channels, heads = 8, 2
    params = _embedder_params(channels, heads)
    features = _empty_templates()
    collapsed = collapse_identical_templates(features)
    assert _template_rows(collapsed) == 1

    key = jax.random.key(7)
    z = jax.random.normal(key, (1, N_TOKEN, N_TOKEN, channels))
    batch = {"asym_id": jnp.zeros((1, N_TOKEN), dtype=jnp.int32)}
    pair_mask = jnp.ones((1, N_TOKEN, N_TOKEN))

    def run(source: dict[str, np.ndarray]) -> jnp.ndarray:
        return template_embedder(
            {**batch, **{k: jnp.asarray(v) for k, v in source.items()}},
            z,
            params,
            pair_mask=pair_mask,
            no_heads=heads,
        )

    np.testing.assert_allclose(
        run(features),
        run(collapsed),
        rtol=0,
        atol=np.finfo(np.float32).eps,
    )


_COMPACT_TEMPLATE_CP_PROBE = textwrap.dedent(
    """
    import os
    os.environ["OPENFOLD3_TRIANGLE_BACKEND"] = "xla"

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.openfold3.data import (
        collapse_identical_templates,
        compact_zero_template_pair_features,
    )
    from foldjax.models.openfold3.models.template_module import template_embedder
    from tests.models.openfold3.test_template_collapse import (
        N_TOKEN,
        _embedder_params,
        _empty_templates,
        _model_batch,
    )

    assert jax.device_count() == 4, jax.devices()
    params = _embedder_params()
    dense = collapse_identical_templates(_empty_templates())
    compact = compact_zero_template_pair_features(dense)
    z = jax.random.normal(
        jax.random.key(23), (1, N_TOKEN, N_TOKEN, 8), dtype=jnp.float32
    )
    pair_mask = jnp.ones((1, N_TOKEN, N_TOKEN), dtype=jnp.float32)
    traced = []

    def build(source):
        batch = _model_batch(source)

        def run(pair):
            traced.append(cp_layout())
            return template_embedder(
                batch, pair, params, pair_mask=pair_mask, no_heads=2
            )

        return run

    reference = jax.device_get(jax.jit(build(dense))(z))
    jax.clear_caches()
    with context_parallel(4, layout="1d"):
        actual = jax.device_get(jax.jit(build(compact))(z))

    assert traced[-2:] == [None, "1d"], traced
    np.testing.assert_allclose(reference, actual, atol=3e-5, rtol=3e-5)
    print("OPENFOLD3_COMPACT_TEMPLATE_CP_OK")
    """
)


def test_compact_template_matches_dense_on_forced_four_cpu_devices() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _COMPACT_TEMPLATE_CP_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OPENFOLD3_COMPACT_TEMPLATE_CP_OK" in completed.stdout
