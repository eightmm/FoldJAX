from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.bridge.torch_mapping import (
    map_conditioned_transition_state_dict,
    map_diffusion_transformer_block_state_dict,
)
from foldjax.models.protenix.models.diffusion.transformer import (
    conditioned_transition_block,
    diffusion_transformer_block,
)
from foldjax.models.protenix.models.primitives.attention import (
    AttentionParams,
    local_attention,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LinearParams,
    adaptive_layer_norm,
    linear,
    sigmoid,
    silu,
)


def test_conditioned_transition_block_matches_reference_formula() -> None:
    rng = np.random.default_rng(6)
    a = rng.normal(size=(1, 3, 4)).astype(np.float32)
    s = rng.normal(size=(1, 3, 5)).astype(np.float32)
    state = {
        "ctb.adaln.layernorm_s.weight": rng.normal(size=(5,)).astype(np.float32),
        "ctb.adaln.linear_s.weight": rng.normal(size=(4, 5)).astype(np.float32),
        "ctb.adaln.linear_s.bias": rng.normal(size=(4,)).astype(np.float32),
        "ctb.adaln.linear_nobias_s.weight": rng.normal(size=(4, 5)).astype(
            np.float32
        ),
        "ctb.linear_nobias_a1.weight": rng.normal(size=(8, 4)).astype(np.float32),
        "ctb.linear_nobias_a2.weight": rng.normal(size=(8, 4)).astype(np.float32),
        "ctb.linear_nobias_b.weight": rng.normal(size=(4, 8)).astype(np.float32),
        "ctb.linear_s.weight": rng.normal(size=(4, 5)).astype(np.float32),
        "ctb.linear_s.bias": rng.normal(size=(4,)).astype(np.float32),
    }
    params = map_conditioned_transition_state_dict(state, "ctb")

    actual = np.asarray(
        conditioned_transition_block(jnp.asarray(a), jnp.asarray(s), params)
    )

    a_norm = adaptive_layer_norm(jnp.asarray(a), jnp.asarray(s), params.adaln)
    hidden = silu(linear(a_norm, params.linear_a1)) * linear(a_norm, params.linear_a2)
    expected = sigmoid(linear(jnp.asarray(s), params.linear_s)) * linear(
        hidden,
        params.linear_b,
    )

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_map_conditioned_transition_state_dict_uses_protenix_keys() -> None:
    state = {
        "ctb.adaln.layernorm_s.weight": np.ones((5,), dtype=np.float32),
        "ctb.adaln.linear_s.weight": np.ones((4, 5), dtype=np.float32),
        "ctb.adaln.linear_s.bias": np.zeros((4,), dtype=np.float32),
        "ctb.adaln.linear_nobias_s.weight": np.ones((4, 5), dtype=np.float32),
        "ctb.linear_nobias_a1.weight": np.ones((8, 4), dtype=np.float32),
        "ctb.linear_nobias_a2.weight": np.ones((8, 4), dtype=np.float32),
        "ctb.linear_nobias_b.weight": np.ones((4, 8), dtype=np.float32),
        "ctb.linear_s.weight": np.ones((4, 5), dtype=np.float32),
        "ctb.linear_s.bias": np.zeros((4,), dtype=np.float32),
    }

    params = map_conditioned_transition_state_dict(state, "ctb")

    assert tuple(params.adaln.linear_s.weight.shape) == (4, 5)
    assert tuple(params.linear_a1.weight.shape) == (8, 4)
    assert tuple(params.linear_a2.weight.shape) == (8, 4)
    assert tuple(params.linear_b.weight.shape) == (4, 8)
    assert tuple(params.linear_s.weight.shape) == (4, 5)


def test_local_attention_matches_manual_window_attention() -> None:
    q_x = jnp.asarray([[1.0], [2.0], [3.0], [4.0]])
    params = AttentionParams(
        linear_q=LinearParams(weight=jnp.ones((1, 1)), bias=jnp.zeros((1,))),
        linear_k=LinearParams(weight=jnp.ones((1, 1)), bias=None),
        linear_v=LinearParams(weight=jnp.ones((1, 1)), bias=None),
        linear_o=LinearParams(weight=jnp.ones((1, 1)), bias=None),
        linear_g=None,
    )
    bias = jnp.zeros((1, 2, 2, 4), dtype=jnp.float32)

    actual = np.asarray(
        local_attention(
            q_x,
            q_x,
            params,
            num_heads=1,
            trunked_attn_bias=bias,
            n_queries=2,
            n_keys=4,
        )
    )

    key_windows = np.asarray([[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 0.0]])
    mask_windows = np.asarray(
        [
            [[False, True, True, True], [False, True, True, True]],
            [[True, True, True, False], [True, True, True, False]],
        ]
    )
    expected = []
    for idx, q_value in enumerate(np.asarray(q_x[:, 0])):
        trunk = idx // 2
        logits = q_value * key_windows[trunk]
        logits = np.where(mask_windows[trunk, idx % 2], logits, -1.0e10)
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        expected.append(np.sum(probs * key_windows[trunk]))

    np.testing.assert_allclose(actual[:, 0], np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_diffusion_transformer_block_local_path_preserves_shape() -> None:
    rng = np.random.default_rng(7)
    state = {
        "block.attention_pair_bias.layernorm_a.layernorm_s.weight": np.ones(
            (4,), dtype=np.float32
        ),
        "block.attention_pair_bias.layernorm_a.linear_s.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.attention_pair_bias.layernorm_a.linear_s.bias": np.zeros(
            (4,), dtype=np.float32
        ),
        "block.attention_pair_bias.layernorm_a.linear_nobias_s.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.attention_pair_bias.layernorm_kv.layernorm_s.weight": np.ones(
            (4,), dtype=np.float32
        ),
        "block.attention_pair_bias.layernorm_kv.linear_s.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.attention_pair_bias.layernorm_kv.linear_s.bias": np.zeros(
            (4,), dtype=np.float32
        ),
        "block.attention_pair_bias.layernorm_kv.linear_nobias_s.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.attention_pair_bias.attention.linear_q.weight": rng.normal(
            size=(4, 4)
        ).astype(np.float32),
        "block.attention_pair_bias.attention.linear_q.bias": np.zeros(
            (4,), dtype=np.float32
        ),
        "block.attention_pair_bias.attention.linear_k.weight": rng.normal(
            size=(4, 4)
        ).astype(np.float32),
        "block.attention_pair_bias.attention.linear_v.weight": rng.normal(
            size=(4, 4)
        ).astype(np.float32),
        "block.attention_pair_bias.attention.linear_o.weight": rng.normal(
            size=(4, 4)
        ).astype(np.float32),
        "block.attention_pair_bias.attention.linear_g.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.attention_pair_bias.layernorm_z.weight": np.ones(
            (2,), dtype=np.float32
        ),
        "block.attention_pair_bias.linear_nobias_z.weight": np.zeros(
            (1, 2), dtype=np.float32
        ),
        "block.attention_pair_bias.linear_a_last.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.attention_pair_bias.linear_a_last.bias": np.zeros(
            (4,), dtype=np.float32
        ),
        "block.conditioned_transition_block.adaln.layernorm_s.weight": np.ones(
            (4,), dtype=np.float32
        ),
        "block.conditioned_transition_block.adaln.linear_s.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.conditioned_transition_block.adaln.linear_s.bias": np.zeros(
            (4,), dtype=np.float32
        ),
        "block.conditioned_transition_block.adaln.linear_nobias_s.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.conditioned_transition_block.linear_nobias_a1.weight": rng.normal(
            size=(8, 4)
        ).astype(np.float32),
        "block.conditioned_transition_block.linear_nobias_a2.weight": rng.normal(
            size=(8, 4)
        ).astype(np.float32),
        "block.conditioned_transition_block.linear_nobias_b.weight": rng.normal(
            size=(4, 8)
        ).astype(np.float32),
        "block.conditioned_transition_block.linear_s.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "block.conditioned_transition_block.linear_s.bias": np.zeros(
            (4,), dtype=np.float32
        ),
    }
    params = map_diffusion_transformer_block_state_dict(
        state,
        "block",
        cross_attention_mode=True,
    )
    a = jnp.asarray(rng.normal(size=(4, 4)).astype(np.float32))
    s = jnp.asarray(rng.normal(size=(4, 4)).astype(np.float32))
    z = jnp.asarray(rng.normal(size=(2, 2, 4, 2)).astype(np.float32))

    out = diffusion_transformer_block(
        a,
        s,
        z,
        params,
        num_heads=1,
        n_queries=2,
        n_keys=4,
    )

    assert out.shape == a.shape
    assert np.isfinite(np.asarray(out)).all()
