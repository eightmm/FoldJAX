"""Collapsing a template axis whose rows are all the same template.

A query with no templates is featurized as the released fixed-width axis of four
identical empty ones, and the template stack embeds every one of them before
averaging. These cover the host-side reduction that keeps one of them, and check
that the value the model computes survives it.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.data import collapse_identical_templates
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
)
from foldjax.models.openfold3.models.triangle import TriangleMultiplicationParams
from foldjax.models.openfold3.models.triangle_attention import TriangleAttentionParams

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
