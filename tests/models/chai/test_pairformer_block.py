from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.models.pairformer import (
    attention_pair_bias,
    fused_triangle_attention,
    fused_triangle_multiplication,
    map_pairformer_block,
    map_pairformer_pair_block,
    pairformer_block,
    pairformer_pair_block,
    pairformer_transition,
)

pytestmark = pytest.mark.official_parity


@pytest.fixture(scope="module")
def block_state(chai_trunk_module):
    block0 = getattr(chai_trunk_module.pairformer_stack.blocks, "0")
    return {
        name: value.detach().cpu().numpy()
        for name, value in block0.state_dict().items()
    }


@pytest.fixture(scope="module")
def template_block_state(chai_trunk_module):
    block0 = getattr(chai_trunk_module.template_embedder.pairformer.blocks, "0")
    return {
        name: value.detach().cpu().numpy()
        for name, value in block0.state_dict().items()
    }


def test_pairformer_block_uses_parallel_pre_block_residuals(block_state) -> None:
    rng = np.random.default_rng(81)
    s = jnp.asarray(rng.normal(size=(1, 4, 384)).astype(np.float32))
    z = jnp.asarray(rng.normal(size=(1, 4, 4, 256)).astype(np.float32))
    token_mask = jnp.asarray([[1, 1, 1, 0]], dtype=bool)
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    params = map_pairformer_block(block_state)

    expected_z = z + fused_triangle_multiplication(
        z, pair_mask, params.triangle_multiplication
    )
    expected_z += fused_triangle_attention(
        z, pair_mask, params.triangle_attention
    )
    expected_z += pairformer_transition(z, params.transition_pair)
    expected_s = s + attention_pair_bias(
        s, z, pair_mask, token_mask, params.attention_pair_bias
    )
    expected_s += pairformer_transition(s, params.transition_single)

    actual_s, actual_z = pairformer_block(
        s, z, token_mask, pair_mask, params
    )

    np.testing.assert_array_equal(np.asarray(actual_s), np.asarray(expected_s))
    np.testing.assert_array_equal(np.asarray(actual_z), np.asarray(expected_z))

    sequential_z = z + fused_triangle_multiplication(
        z, pair_mask, params.triangle_multiplication
    )
    sequential_z += fused_triangle_attention(
        sequential_z, pair_mask, params.triangle_attention
    )
    assert not np.allclose(np.asarray(actual_z), np.asarray(sequential_z))


def test_pairformer_block_mapping_requires_all_official_weights(block_state) -> None:
    assert len(block_state) == 26
    incomplete = dict(block_state)
    incomplete.pop("triangle_attention.out_scalers")
    with pytest.raises(KeyError):
        map_pairformer_block(incomplete)


def test_template_pair_block_uses_same_parallel_pair_contract(
    template_block_state,
) -> None:
    rng = np.random.default_rng(82)
    z = jnp.asarray(rng.normal(size=(1, 4, 4, 64)).astype(np.float32))
    pair_mask = jnp.asarray(
        [[[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]],
        dtype=bool,
    )
    params = map_pairformer_pair_block(template_block_state)
    expected = z + fused_triangle_multiplication(
        z, pair_mask, params.triangle_multiplication
    )
    expected += fused_triangle_attention(
        z, pair_mask, params.triangle_attention
    )
    expected += pairformer_transition(z, params.transition_pair)

    actual = pairformer_pair_block(z, pair_mask, params)

    assert len(template_block_state) == 14
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
