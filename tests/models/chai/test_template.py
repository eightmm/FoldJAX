from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.models.pairformer import pairformer_pair_block
from foldjax.models.chai.models.primitives import layer_norm, linear_bf16
from foldjax.models.chai.models.template import (
    map_template_embedder,
    template_embedding,
)

pytestmark = pytest.mark.official_parity


@pytest.fixture(scope="module")
def template_state(chai_trunk_module):
    return {
        name: value.detach().cpu().numpy()
        for name, value in chai_trunk_module.template_embedder.state_dict().items()
    }


def _inputs(seed: int = 91):
    rng = np.random.default_rng(seed)
    z = jnp.asarray(rng.normal(size=(1, 5, 5, 256)).astype(np.float32))
    features = jnp.asarray(
        rng.normal(size=(1, 4, 5, 5, 64)).astype(np.float32)
    )
    token_mask = jnp.asarray([[1, 1, 1, 1, 0]], dtype=bool)
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    masks = jnp.broadcast_to(pair_mask[:, None], (1, 4, 5, 5))
    masks = masks.at[:, 3].set(False)
    return z, features, masks, pair_mask


def test_template_embedding_matches_graph_orchestration(template_state) -> None:
    z, features, masks, pair_mask = _inputs()
    params = map_template_embedder(template_state)
    pair_base = linear_bf16(
        layer_norm(
            z.astype(jnp.float32),
            params.proj_in_norm_weight,
            params.proj_in_norm_bias,
        ),
        params.proj_in_weight,
    )
    combined_mask = masks & pair_mask[:, None]
    outputs = []
    for template_index in range(4):
        value = pair_base + features[:, template_index]
        for block in params.blocks:
            value = pairformer_pair_block(
                value, combined_mask[:, template_index], block
            )
        outputs.append(value)
    stacked = jnp.stack(outputs, axis=1)
    stacked = layer_norm(
        stacked.astype(jnp.float32),
        params.output_norm_weight,
        params.output_norm_bias,
    )
    stacked *= combined_mask[..., None]
    count = jnp.maximum(jnp.any(masks, axis=(-2, -1)).sum(axis=1), 1)
    pooled = jnp.sum(stacked, axis=1, dtype=jnp.float32)
    pooled /= count[:, None, None, None]
    expected = z + linear_bf16(jnp.maximum(pooled, 0), params.proj_out_weight)

    actual = template_embedding(z, features, masks, pair_mask, params)

    assert len(template_state) == 34
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_fully_masked_template_is_output_invariant(template_state) -> None:
    z, features, masks, pair_mask = _inputs(92)
    params = map_template_embedder(template_state)
    changed = features.at[:, 3].set(1e4)

    baseline = template_embedding(z, features, masks, pair_mask, params)
    actual = template_embedding(z, changed, masks, pair_mask, params)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(baseline))


def test_template_embedding_requires_official_four_template_bucket(
    template_state,
) -> None:
    z, features, masks, pair_mask = _inputs(93)
    with pytest.raises(ValueError, match="exactly four templates"):
        template_embedding(
            z,
            features[:, :3],
            masks[:, :3],
            pair_mask,
            map_template_embedder(template_state),
        )
