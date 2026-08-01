from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.opendde.models.structural_tokens as structural_tokens_impl
from foldjax.models.opendde.bridge.torch_mapping import (
    map_structural_token_expander_state_dict,
)
from foldjax.models.opendde.models.structural_tokens import (
    STRUCTURAL_TOKEN_ROLES,
    build_structural_pair_features,
    structural_token_expand,
)


def _feature_dict() -> dict[str, jnp.ndarray]:
    return {
        "subtoken_role_id": jnp.asarray(
            [
                STRUCTURAL_TOKEN_ROLES["protein_bb"],
                STRUCTURAL_TOKEN_ROLES["protein_sc"],
                STRUCTURAL_TOKEN_ROLES["protein_bb"],
                STRUCTURAL_TOKEN_ROLES["dna_bb"],
                STRUCTURAL_TOKEN_ROLES["dna_base"],
            ],
            dtype=jnp.int32,
        ),
        "parent_residue_idx": jnp.asarray([0, 0, 1, 2, 2], dtype=jnp.int32),
        "asym_id": jnp.asarray([0, 0, 1], dtype=jnp.int32),
        "residue_index": jnp.asarray([10, 11, 1], dtype=jnp.int32),
        "prev_parent_residue_idx": jnp.asarray([-1, -1, 0, -1, -1], dtype=jnp.int32),
        "next_parent_residue_idx": jnp.asarray([1, 1, -1, -1, -1], dtype=jnp.int32),
        "structural_polymer_type": jnp.asarray([1, 1, 1, 2, 2], dtype=jnp.int32),
    }


def _state(
    *,
    c_s: int = 4,
    c_z: int = 3,
    c_s_inputs: int = 5,
    n_roles: int = 7,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    prefix = "structural_token_expander"
    state = {
        f"{prefix}.single_split_mlp.0.weight": rng.normal(size=(c_s,)).astype(
            np.float32
        ),
        f"{prefix}.single_split_mlp.0.bias": rng.normal(size=(c_s,)).astype(np.float32),
        f"{prefix}.single_split_mlp.1.weight": rng.normal(size=(2 * c_s, c_s)).astype(
            np.float32
        ),
        f"{prefix}.single_split_mlp.3.weight": rng.normal(size=(c_s, 2 * c_s)).astype(
            np.float32
        ),
        f"{prefix}.single_input_role_embedding.weight": rng.normal(
            size=(n_roles, c_s_inputs)
        ).astype(np.float32),
        f"{prefix}.single_role_embedding.weight": rng.normal(
            size=(n_roles, c_s)
        ).astype(np.float32),
        f"{prefix}.same_parent_embedding.weight": rng.normal(size=(2, c_z)).astype(
            np.float32
        ),
        f"{prefix}.same_residue_twin_embedding.weight": rng.normal(
            size=(2, c_z)
        ).astype(np.float32),
        f"{prefix}.prev_bb_chain_embedding.weight": rng.normal(size=(2, c_z)).astype(
            np.float32
        ),
        f"{prefix}.next_bb_chain_embedding.weight": rng.normal(size=(2, c_z)).astype(
            np.float32
        ),
        f"{prefix}.role_pair_type_embedding.weight": rng.normal(size=(8, c_z)).astype(
            np.float32
        ),
        f"{prefix}.attn_bias_same_parent": np.asarray(0.11, dtype=np.float32),
        f"{prefix}.attn_bias_same_residue_twin": np.asarray(-0.07, dtype=np.float32),
        f"{prefix}.attn_bias_prev_bb_chain": np.asarray(0.03, dtype=np.float32),
        f"{prefix}.attn_bias_next_bb_chain": np.asarray(-0.05, dtype=np.float32),
        f"{prefix}.attn_bias_role_pair_type": rng.normal(size=(8,)).astype(np.float32),
    }
    for index in range(n_roles * n_roles):
        state[f"{prefix}.pair_block_proj.{index}.weight"] = rng.normal(
            size=(c_z, c_z)
        ).astype(np.float32)
    return state


def test_structural_pair_features_preserve_twin_and_backbone_direction() -> None:
    features = _feature_dict()

    pair = build_structural_pair_features(features)

    same_parent = np.asarray(pair["same_parent_residue"])
    same_twin = np.asarray(pair["same_residue_twin"])
    prev_chain = np.asarray(pair["prev_bb_chain"])
    next_chain = np.asarray(pair["next_bb_chain"])
    role_pair = np.asarray(pair["role_pair_type"])

    assert same_parent[0, 1]
    assert same_twin[0, 1] and same_twin[1, 0]
    assert same_twin[3, 4] and same_twin[4, 3]
    assert not same_twin[0, 0]
    assert next_chain[0, 2] and prev_chain[2, 0]
    assert not next_chain[2, 0] and not prev_chain[0, 2]
    assert role_pair[0, 2] == 0
    assert role_pair[0, 1] == 1
    assert role_pair[1, 0] == 2
    assert role_pair[3, 4] == 4
    assert role_pair[4, 3] == 5
    assert role_pair[4, 4] == 6


def test_structural_token_expand_matches_released_full_projection_formula() -> None:
    rng = np.random.default_rng(23)
    state = _state()
    params = map_structural_token_expander_state_dict(state)
    features = _feature_dict()
    s_inputs_res = rng.normal(size=(2, 3, 5)).astype(np.float32)
    s_res = rng.normal(size=(2, 3, 4)).astype(np.float32)
    z_res = rng.normal(size=(2, 3, 3, 3)).astype(np.float32)

    actual_inputs, actual_s, actual_z, pair = structural_token_expand(
        features,
        jnp.asarray(s_inputs_res),
        jnp.asarray(s_res),
        jnp.asarray(z_res),
        params,
    )

    parent = np.asarray(features["parent_residue_idx"])
    role = np.asarray(features["subtoken_role_id"])
    prefix = "structural_token_expander"
    expected_inputs = (
        s_inputs_res[:, parent]
        + state[f"{prefix}.single_input_role_embedding.weight"][role]
    )
    s_parent = s_res[:, parent]
    mean = s_parent.mean(axis=-1, keepdims=True)
    var = ((s_parent - mean) ** 2).mean(axis=-1, keepdims=True)
    hidden = (s_parent - mean) / np.sqrt(var + 1e-5)
    hidden = (
        hidden * state[f"{prefix}.single_split_mlp.0.weight"]
        + state[f"{prefix}.single_split_mlp.0.bias"]
    )
    hidden = hidden @ state[f"{prefix}.single_split_mlp.1.weight"].T
    hidden = hidden / (1.0 + np.exp(-hidden))
    hidden = hidden @ state[f"{prefix}.single_split_mlp.3.weight"].T
    expected_s = (
        s_parent + hidden + state[f"{prefix}.single_role_embedding.weight"][role]
    )

    z_parent = z_res[:, parent][:, :, parent]
    pair_weights = np.stack(
        [state[f"{prefix}.pair_block_proj.{index}.weight"] for index in range(49)]
    ).reshape(7, 7, 3, 3)
    selected = pair_weights[role[:, None], role[None, :]]
    expected_z = z_parent + np.einsum("...ijc,ijoc->...ijo", z_parent, selected)
    expected_z += state[f"{prefix}.same_parent_embedding.weight"][
        np.asarray(pair["same_parent_residue"], dtype=np.int32)
    ]
    expected_z += state[f"{prefix}.same_residue_twin_embedding.weight"][
        np.asarray(pair["same_residue_twin"], dtype=np.int32)
    ]
    expected_z += state[f"{prefix}.prev_bb_chain_embedding.weight"][
        np.asarray(pair["prev_bb_chain"], dtype=np.int32)
    ]
    expected_z += state[f"{prefix}.next_bb_chain_embedding.weight"][
        np.asarray(pair["next_bb_chain"], dtype=np.int32)
    ]
    expected_z += state[f"{prefix}.role_pair_type_embedding.weight"][
        np.asarray(pair["role_pair_type"])
    ]

    np.testing.assert_allclose(actual_inputs, expected_inputs, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_s, expected_s, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_z, expected_z, rtol=1e-5, atol=1e-5)
    assert actual_inputs.shape == (2, 5, 5)
    assert actual_s.shape == (2, 5, 4)
    assert actual_z.shape == (2, 5, 5, 3)
    assert pair["structural_pair_attn_bias"].shape == (5, 5)


def test_role_pair_projection_avoids_pairwise_weight_tensor() -> None:
    rng = np.random.default_rng(29)
    n_structural = 19
    c_z = 8
    z = rng.normal(size=(2, n_structural, n_structural, c_z)).astype(np.float32)
    role = (np.arange(n_structural) % 7).astype(np.int32)
    weights = rng.normal(size=(7, 7, c_z, c_z)).astype(np.float32)

    actual = structural_tokens_impl._pair_project_by_role(
        jnp.asarray(z),
        jnp.asarray(role),
        jnp.asarray(weights),
    )
    selected = weights[role[:, None], role[None, :]]
    expected = np.einsum("...ijc,ijoc->...ijo", z, selected)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    lowered = jax.jit(structural_tokens_impl._pair_project_by_role).lower(
        jnp.asarray(z),
        jnp.asarray(role),
        jnp.asarray(weights),
    )
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text().replace(" ", "")
    assert f"f32[{n_structural},{n_structural},{c_z},{c_z}]" not in hlo


def test_structural_expander_mapper_requires_all_49_pair_projections() -> None:
    state = _state()
    state.pop("structural_token_expander.pair_block_proj.48.weight")

    with pytest.raises(
        KeyError,
        match=r"structural_token_expander\.pair_block_proj\.48\.weight",
    ):
        map_structural_token_expander_state_dict(state)
