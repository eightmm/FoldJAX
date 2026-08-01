from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.bridge.torch_mapping import (
    map_pairformer_block_state_dict,
    map_pairformer_stack_state_dict,
)
from foldjax.models.protenix.models.primitives.attention import attention_pair_bias
from foldjax.models.protenix.models.primitives.primitives import transition
from foldjax.models.protenix.models.triangle.triangle import (
    triangle_attention,
    triangle_multiplication,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    pairformer_block,
    pairformer_stack,
)


def test_pairformer_block_matches_reference_composition() -> None:
    rng = np.random.default_rng(11)
    s = rng.normal(size=(1, 3, 4)).astype(np.float32)
    z = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    pair_mask = np.ones((1, 3, 3), dtype=np.float32)
    state = _pairformer_block_state(rng, c_s=4, c_z=4, heads=2)
    params = map_pairformer_block_state_dict(state, "block", has_s=True)

    actual_s, actual_z = pairformer_block(
        jnp.asarray(s),
        jnp.asarray(z),
        jnp.asarray(pair_mask),
        params,
    )

    expected_z = jnp.asarray(z)
    expected_z = expected_z + triangle_multiplication(
        expected_z,
        jnp.asarray(pair_mask),
        params.tri_mul_out,
        "outgoing",
    )
    expected_z = expected_z + triangle_multiplication(
        expected_z,
        jnp.asarray(pair_mask),
        params.tri_mul_in,
        "incoming",
    )
    tri_heads = int(params.tri_att_start.linear.weight.shape[0])
    expected_z = expected_z + triangle_attention(
        expected_z,
        jnp.asarray(pair_mask),
        params.tri_att_start,
        num_heads=tri_heads,
    )
    expected_z_t = jnp.swapaxes(expected_z, -2, -3)
    expected_z_t = expected_z_t + triangle_attention(
        expected_z_t,
        jnp.swapaxes(jnp.asarray(pair_mask), -1, -2),
        params.tri_att_end,
        num_heads=tri_heads,
    )
    expected_z = jnp.swapaxes(expected_z_t, -2, -3)
    expected_z = expected_z + transition(expected_z, params.pair_transition)

    pair_heads = int(params.attention_pair_bias.linear_z.weight.shape[0])
    expected_s = jnp.asarray(s) + attention_pair_bias(
        jnp.asarray(s),
        None,
        expected_z,
        params.attention_pair_bias,
        num_heads=pair_heads,
    )
    expected_s = expected_s + transition(expected_s, params.single_transition)

    np.testing.assert_allclose(
        np.asarray(actual_z),
        np.asarray(expected_z),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_s),
        np.asarray(expected_s),
        rtol=1e-5,
        atol=1e-5,
    )


def test_map_pairformer_block_state_dict_uses_protenix_keys() -> None:
    rng = np.random.default_rng(12)
    state = _pairformer_block_state(rng, c_s=4, c_z=4, heads=2)

    params = map_pairformer_block_state_dict(state, "block", has_s=True)

    assert tuple(params.tri_mul_out.linear_a_p.weight.shape) == (4, 4)
    assert tuple(params.tri_att_start.linear.weight.shape) == (2, 4)
    assert tuple(params.pair_transition.linear_a.weight.shape) == (16, 4)
    assert tuple(params.attention_pair_bias.linear_z.weight.shape) == (2, 4)
    assert tuple(params.single_transition.linear_out.weight.shape) == (4, 16)


def test_pairformer_stack_scan_matches_loop() -> None:
    rng = np.random.default_rng(13)
    s = rng.normal(size=(1, 3, 4)).astype(np.float32)
    z = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    pair_mask = np.ones((1, 3, 3), dtype=np.float32)
    state = {}
    state.update(
        _pairformer_block_state(rng, c_s=4, c_z=4, heads=2, prefix="stack.blocks.0")
    )
    state.update(
        _pairformer_block_state(rng, c_s=4, c_z=4, heads=2, prefix="stack.blocks.1")
    )
    params = map_pairformer_stack_state_dict(state, "stack", has_s=True)

    loop_s, loop_z = pairformer_stack(
        jnp.asarray(s),
        jnp.asarray(z),
        jnp.asarray(pair_mask),
        params,
        use_scan=False,
    )
    scan_s, scan_z = pairformer_stack(
        jnp.asarray(s),
        jnp.asarray(z),
        jnp.asarray(pair_mask),
        params,
        use_scan=True,
    )
    compiled_s, compiled_z = pairformer_stack(
        jnp.asarray(s),
        jnp.asarray(z),
        jnp.asarray(pair_mask),
        params,
        use_scan=False,
        single_attention_backend="xla_jit",
        triangle_attention_backend="xla_jit",
    )

    np.testing.assert_allclose(
        np.asarray(scan_z),
        np.asarray(loop_z),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(scan_s),
        np.asarray(loop_s),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(compiled_z),
        np.asarray(loop_z),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(compiled_s),
        np.asarray(loop_s),
        rtol=1e-5,
        atol=1e-5,
    )


def test_map_pairformer_stack_state_dict_discovers_block_indices() -> None:
    rng = np.random.default_rng(14)
    state = {}
    state.update(
        _pairformer_block_state(rng, c_s=4, c_z=4, heads=2, prefix="stack.blocks.0")
    )
    state.update(
        _pairformer_block_state(rng, c_s=4, c_z=4, heads=2, prefix="stack.blocks.1")
    )

    params = map_pairformer_stack_state_dict(state, "stack", has_s=True)

    assert len(params.blocks) == 2
    assert tuple(params.blocks[0].tri_att_start.linear.weight.shape) == (2, 4)
    assert tuple(params.blocks[1].single_transition.linear_out.weight.shape) == (4, 16)


def _pairformer_block_state(
    rng: np.random.Generator,
    *,
    c_s: int,
    c_z: int,
    heads: int,
    prefix: str = "block",
) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    _add_triangle_multiplication_state(state, rng, f"{prefix}.tri_mul_out", c_z)
    _add_triangle_multiplication_state(state, rng, f"{prefix}.tri_mul_in", c_z)
    _add_triangle_attention_state(state, rng, f"{prefix}.tri_att_start", c_z, heads)
    _add_triangle_attention_state(state, rng, f"{prefix}.tri_att_end", c_z, heads)
    _add_transition_state(state, rng, f"{prefix}.pair_transition", c_z, factor=4)
    _add_attention_pair_bias_state(
        state,
        rng,
        f"{prefix}.attention_pair_bias",
        c_s,
        c_z,
        heads,
    )
    _add_transition_state(state, rng, f"{prefix}.single_transition", c_s, factor=4)
    return state


def _add_triangle_multiplication_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_z: int,
) -> None:
    state[f"{prefix}.layer_norm_in.weight"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm_in.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm_out.weight"] = rng.normal(size=(c_z,)).astype(
        np.float32
    )
    state[f"{prefix}.layer_norm_out.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    for name in ("linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g"):
        state[f"{prefix}.{name}.weight"] = rng.normal(size=(c_z, c_z)).astype(
            np.float32
        )
    state[f"{prefix}.linear_z.weight"] = rng.normal(size=(c_z, c_z)).astype(np.float32)
    state[f"{prefix}.linear_g.weight"] = rng.normal(size=(c_z, c_z)).astype(np.float32)


def _add_triangle_attention_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_z: int,
    heads: int,
) -> None:
    state[f"{prefix}.layer_norm.weight"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.linear.weight"] = rng.normal(size=(heads, c_z)).astype(np.float32)
    for name in ("linear_q", "linear_k", "linear_v", "linear_g"):
        state[f"{prefix}.mha.{name}.weight"] = rng.normal(size=(c_z, c_z)).astype(
            np.float32
        )
    state[f"{prefix}.mha.linear_o.weight"] = rng.normal(size=(c_z, c_z)).astype(
        np.float32
    )


def _add_transition_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_in: int,
    *,
    factor: int,
) -> None:
    hidden = c_in * factor
    state[f"{prefix}.layernorm1.weight"] = rng.normal(size=(c_in,)).astype(np.float32)
    state[f"{prefix}.layernorm1.bias"] = rng.normal(size=(c_in,)).astype(np.float32)
    state[f"{prefix}.linear_no_bias_a.weight"] = rng.normal(size=(hidden, c_in)).astype(
        np.float32
    )
    state[f"{prefix}.linear_no_bias_b.weight"] = rng.normal(size=(hidden, c_in)).astype(
        np.float32
    )
    state[f"{prefix}.linear_no_bias.weight"] = rng.normal(size=(c_in, hidden)).astype(
        np.float32
    )


def _add_attention_pair_bias_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_s: int,
    c_z: int,
    heads: int,
) -> None:
    state[f"{prefix}.layernorm_a.weight"] = rng.normal(size=(c_s,)).astype(np.float32)
    state[f"{prefix}.layernorm_a.bias"] = rng.normal(size=(c_s,)).astype(np.float32)
    state[f"{prefix}.layernorm_z.weight"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layernorm_z.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.linear_nobias_z.weight"] = rng.normal(size=(heads, c_z)).astype(
        np.float32
    )
    for name in ("linear_q", "linear_k", "linear_v", "linear_o", "linear_g"):
        state[f"{prefix}.attention.{name}.weight"] = rng.normal(size=(c_s, c_s)).astype(
            np.float32
        )
    state[f"{prefix}.attention.linear_q.bias"] = rng.normal(size=(c_s,)).astype(
        np.float32
    )
