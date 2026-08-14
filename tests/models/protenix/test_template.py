from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.bridge.torch_mapping import (
    map_template_embedder_state_dict,
)
from foldjax.models.protenix.models.primitives.primitives import layer_norm, linear
from foldjax.models.protenix.models.trunk_blocks.template import (
    template_embedder,
    template_pair_features,
)


def test_template_pair_features_applies_masks_and_restypes() -> None:
    features = {
        "asym_id": jnp.asarray([1, 1, 2]),
        "template_distogram": jnp.ones((1, 3, 3, 39)),
        "template_pseudo_beta_mask": jnp.ones((1, 3, 3)),
        "template_aatype": jnp.asarray([[0, 1, 2]]),
        "template_unit_vector": jnp.ones((1, 3, 3, 3)) * 2.0,
        "template_backbone_frame_mask": jnp.ones((1, 3, 3)),
    }
    pair_mask = jnp.asarray(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=jnp.float32,
    )

    actual = np.asarray(template_pair_features(features, 0, pair_mask))

    assert actual.shape == (3, 3, 108)
    np.testing.assert_allclose(actual[0, 2, :39], 0.0)
    np.testing.assert_allclose(actual[1, 1, :39], 0.0)
    assert actual[0, 1, 39] == 1.0
    assert actual[0, 2, 39] == 0.0
    assert actual[0, 1, 41] == 1.0
    assert actual[0, 1, 72] == 1.0
    np.testing.assert_allclose(actual[0, 1, 104:107], 2.0)
    np.testing.assert_allclose(actual[0, 2, 104:107], 0.0)
    assert actual[0, 1, 107] == 1.0


def test_template_embedder_returns_zero_without_templates() -> None:
    params = map_template_embedder_state_dict(_template_state(np.random.default_rng(1)))
    z = jnp.ones((3, 3, 4), dtype=jnp.float32)

    actual = template_embedder({}, z, None, params)

    np.testing.assert_allclose(np.asarray(actual), np.zeros((3, 3, 4)))


def test_template_embedder_zero_pair_stack_matches_reference_projection() -> None:
    rng = np.random.default_rng(2)
    state = _template_state(rng)
    params = map_template_embedder_state_dict(state)
    z = rng.normal(size=(3, 3, 4)).astype(np.float32)
    features = {
        "asym_id": jnp.asarray([1, 1, 1]),
        "template_distogram": jnp.zeros((1, 3, 3, 39), dtype=jnp.float32),
        "template_pseudo_beta_mask": jnp.ones((1, 3, 3), dtype=jnp.float32),
        "template_aatype": jnp.asarray([[0, 1, 2]]),
        "template_unit_vector": jnp.zeros((1, 3, 3, 3), dtype=jnp.float32),
        "template_backbone_frame_mask": jnp.ones((1, 3, 3), dtype=jnp.float32),
    }

    actual = template_embedder(features, jnp.asarray(z), None, params)
    compiled = template_embedder(
        features,
        jnp.asarray(z),
        None,
        params,
        triangle_attention_backend="xla_jit",
    )

    at = template_pair_features(features, 0, None)
    z_norm = layer_norm(jnp.asarray(z), params.layernorm_z)
    v = linear(z_norm, params.linear_z) + linear(at, params.linear_a)
    expected = linear(
        jnp.maximum(layer_norm(v, params.layernorm_v), 0.0),
        params.linear_u,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(expected), atol=1e-5)


def test_map_template_embedder_state_dict_discovers_pairformer_blocks() -> None:
    params = map_template_embedder_state_dict(_template_state(np.random.default_rng(3)))

    assert tuple(params.linear_z.weight.shape) == (2, 4)
    assert tuple(params.linear_a.weight.shape) == (2, 108)
    assert len(params.pairformer_stack.blocks) == 1
    assert tuple(params.linear_u.weight.shape) == (4, 2)


def test_map_template_embedder_accepts_published_zero_block_architecture() -> None:
    state = _template_state(np.random.default_rng(4))
    state = {
        key: value
        for key, value in state.items()
        if not key.startswith("template_embedder.pairformer_stack.blocks.")
    }

    params = map_template_embedder_state_dict(state)
    output = template_embedder(
        {"template_aatype": jnp.zeros((1, 2), dtype=jnp.int32)},
        jnp.ones((2, 2, 4), dtype=jnp.float32),
        None,
        params,
    )

    assert params.pairformer_stack.blocks == ()
    np.testing.assert_array_equal(np.asarray(output), np.zeros((2, 2, 4)))


def _template_state(rng: np.random.Generator) -> dict[str, np.ndarray]:
    state = {
        "template_embedder.linear_no_bias_z.weight": rng.normal(
            size=(2, 4)
        ).astype(np.float32),
        "template_embedder.layernorm_z.weight": rng.normal(size=(4,)).astype(
            np.float32
        ),
        "template_embedder.layernorm_z.bias": rng.normal(size=(4,)).astype(np.float32),
        "template_embedder.linear_no_bias_a.weight": rng.normal(
            size=(2, 108)
        ).astype(np.float32),
        "template_embedder.layernorm_v.weight": rng.normal(size=(2,)).astype(
            np.float32
        ),
        "template_embedder.layernorm_v.bias": rng.normal(size=(2,)).astype(np.float32),
        "template_embedder.linear_no_bias_u.weight": rng.normal(
            size=(4, 2)
        ).astype(np.float32),
    }
    _add_zero_pairformer_block(state, "template_embedder.pairformer_stack.blocks.0")
    return state


def _add_zero_pairformer_block(state: dict[str, np.ndarray], prefix: str) -> None:
    c_z = 2
    heads = 1
    _add_zero_triangle_multiplication(state, f"{prefix}.tri_mul_out", c_z)
    _add_zero_triangle_multiplication(state, f"{prefix}.tri_mul_in", c_z)
    _add_zero_triangle_attention(state, f"{prefix}.tri_att_start", c_z, heads)
    _add_zero_triangle_attention(state, f"{prefix}.tri_att_end", c_z, heads)
    _add_zero_transition(state, f"{prefix}.pair_transition", c_z, factor=2)


def _add_zero_triangle_multiplication(
    state: dict[str, np.ndarray],
    prefix: str,
    c_z: int,
) -> None:
    state[f"{prefix}.layer_norm_in.weight"] = np.ones((c_z,), dtype=np.float32)
    state[f"{prefix}.layer_norm_in.bias"] = np.zeros((c_z,), dtype=np.float32)
    state[f"{prefix}.layer_norm_out.weight"] = np.ones((c_z,), dtype=np.float32)
    state[f"{prefix}.layer_norm_out.bias"] = np.zeros((c_z,), dtype=np.float32)
    for name in (
        "linear_a_p",
        "linear_a_g",
        "linear_b_p",
        "linear_b_g",
        "linear_z",
        "linear_g",
    ):
        state[f"{prefix}.{name}.weight"] = np.zeros((c_z, c_z), dtype=np.float32)


def _add_zero_triangle_attention(
    state: dict[str, np.ndarray],
    prefix: str,
    c_z: int,
    heads: int,
) -> None:
    state[f"{prefix}.layer_norm.weight"] = np.ones((c_z,), dtype=np.float32)
    state[f"{prefix}.layer_norm.bias"] = np.zeros((c_z,), dtype=np.float32)
    state[f"{prefix}.linear.weight"] = np.zeros((heads, c_z), dtype=np.float32)
    for name in ("linear_q", "linear_k", "linear_v", "linear_o", "linear_g"):
        state[f"{prefix}.mha.{name}.weight"] = np.zeros((c_z, c_z), dtype=np.float32)


def _add_zero_transition(
    state: dict[str, np.ndarray],
    prefix: str,
    c_in: int,
    *,
    factor: int,
) -> None:
    hidden = c_in * factor
    state[f"{prefix}.layernorm1.weight"] = np.ones((c_in,), dtype=np.float32)
    state[f"{prefix}.layernorm1.bias"] = np.zeros((c_in,), dtype=np.float32)
    state[f"{prefix}.linear_no_bias_a.weight"] = np.zeros(
        (hidden, c_in), dtype=np.float32
    )
    state[f"{prefix}.linear_no_bias_b.weight"] = np.zeros(
        (hidden, c_in), dtype=np.float32
    )
    state[f"{prefix}.linear_no_bias.weight"] = np.zeros(
        (c_in, hidden), dtype=np.float32
    )
