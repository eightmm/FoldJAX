from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.bridge.torch_mapping import (
    map_released_structural_refiner_state_dict,
    map_structural_refiner_state_dict,
)
from foldjax.models.opendde.models.structural_refiner import structural_refiner_stack
from foldjax.models.protenix.models.primitives.attention import attention
from foldjax.models.protenix.models.primitives.primitives import (
    layer_norm,
    linear,
    transition,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import pairformer_block


def test_structural_refiner_adds_role_bias_to_single_attention() -> None:
    rng = np.random.default_rng(31)
    state = _pairformer_block_state(rng)
    params = map_structural_refiner_state_dict(state)
    s = jnp.asarray(rng.normal(size=(1, 3, 4)).astype(np.float32))
    z = jnp.asarray(rng.normal(size=(1, 3, 3, 4)).astype(np.float32))
    pair_mask = jnp.ones((1, 3, 3), dtype=jnp.float32)
    role_bias = jnp.asarray(rng.normal(size=(3, 3)).astype(np.float32))

    actual_s, actual_z = structural_refiner_stack(
        s,
        z,
        pair_mask,
        role_bias,
        params,
        use_scan=False,
    )

    block = params.blocks[0]
    pair_only = block._replace(
        attention_pair_bias=None,
        single_transition=None,
    )
    _, expected_z = pairformer_block(None, z, pair_mask, pair_only)
    apb = block.attention_pair_bias
    assert apb is not None
    q = layer_norm(s, apb.layernorm_a)
    pair_bias = linear(layer_norm(expected_z, apb.layernorm_z), apb.linear_z)
    pair_bias = jnp.moveaxis(pair_bias, -1, -3)
    pair_bias = pair_bias + role_bias[None, None, :, :]
    heads = int(apb.linear_z.weight.shape[0])
    expected_s = s + attention(
        q,
        q,
        apb.attention,
        heads,
        pair_bias,
    )
    assert block.single_transition is not None
    expected_s = expected_s + transition(expected_s, block.single_transition)

    np.testing.assert_allclose(actual_z, expected_z, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_s, expected_s, rtol=1e-5, atol=1e-5)


def test_structural_refiner_scan_matches_loop_with_batched_bias() -> None:
    rng = np.random.default_rng(32)
    state = {}
    state.update(
        _pairformer_block_state(
            rng,
            prefix="structural_token_refiner.blocks.0",
        )
    )
    state.update(
        _pairformer_block_state(
            rng,
            prefix="structural_token_refiner.blocks.1",
        )
    )
    params = map_structural_refiner_state_dict(state)
    s = jnp.asarray(rng.normal(size=(2, 3, 4)).astype(np.float32))
    z = jnp.asarray(rng.normal(size=(2, 3, 3, 4)).astype(np.float32))
    role_bias = jnp.asarray(rng.normal(size=(2, 3, 3)).astype(np.float32))

    loop_s, loop_z = structural_refiner_stack(
        s,
        z,
        None,
        role_bias,
        params,
        use_scan=False,
    )
    scan_s, scan_z = structural_refiner_stack(
        s,
        z,
        None,
        role_bias,
        params,
        use_scan=True,
    )

    np.testing.assert_allclose(scan_z, loop_z, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scan_s, loop_s, rtol=1e-5, atol=1e-5)


def test_released_structural_refiner_requires_four_blocks() -> None:
    state = _pairformer_block_state(np.random.default_rng(33))

    with pytest.raises(KeyError, match=r"structural_token_refiner\.blocks\.1"):
        map_released_structural_refiner_state_dict(state)


def _pairformer_block_state(
    rng: np.random.Generator,
    *,
    prefix: str = "structural_token_refiner.blocks.0",
    c_s: int = 4,
    c_z: int = 4,
    heads: int = 2,
) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    _add_triangle_multiplication(state, rng, f"{prefix}.tri_mul_out", c_z)
    _add_triangle_multiplication(state, rng, f"{prefix}.tri_mul_in", c_z)
    _add_triangle_attention(
        state,
        rng,
        f"{prefix}.tri_att_start",
        c_z,
        heads,
    )
    _add_triangle_attention(
        state,
        rng,
        f"{prefix}.tri_att_end",
        c_z,
        heads,
    )
    _add_transition(state, rng, f"{prefix}.pair_transition", c_z, factor=2)
    _add_attention_pair_bias(
        state,
        rng,
        f"{prefix}.attention_pair_bias",
        c_s,
        c_z,
        heads,
    )
    _add_transition(state, rng, f"{prefix}.single_transition", c_s, factor=4)
    return state


def _add_triangle_multiplication(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    channels: int,
) -> None:
    for name in ("layer_norm_in", "layer_norm_out"):
        state[f"{prefix}.{name}.weight"] = _random(rng, (channels,))
        state[f"{prefix}.{name}.bias"] = _random(rng, (channels,))
    for name in ("linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g"):
        state[f"{prefix}.{name}.weight"] = _random(rng, (channels, channels))
    state[f"{prefix}.linear_z.weight"] = _random(rng, (channels, channels))
    state[f"{prefix}.linear_g.weight"] = _random(rng, (channels, channels))


def _add_triangle_attention(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    channels: int,
    heads: int,
) -> None:
    state[f"{prefix}.layer_norm.weight"] = _random(rng, (channels,))
    state[f"{prefix}.layer_norm.bias"] = _random(rng, (channels,))
    state[f"{prefix}.linear.weight"] = _random(rng, (heads, channels))
    for name in ("linear_q", "linear_k", "linear_v", "linear_g", "linear_o"):
        state[f"{prefix}.mha.{name}.weight"] = _random(
            rng,
            (channels, channels),
        )


def _add_transition(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    channels: int,
    *,
    factor: int,
) -> None:
    hidden = channels * factor
    state[f"{prefix}.layernorm1.weight"] = _random(rng, (channels,))
    state[f"{prefix}.layernorm1.bias"] = _random(rng, (channels,))
    state[f"{prefix}.linear_no_bias_a.weight"] = _random(
        rng,
        (hidden, channels),
    )
    state[f"{prefix}.linear_no_bias_b.weight"] = _random(
        rng,
        (hidden, channels),
    )
    state[f"{prefix}.linear_no_bias.weight"] = _random(rng, (channels, hidden))


def _add_attention_pair_bias(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_s: int,
    c_z: int,
    heads: int,
) -> None:
    state[f"{prefix}.layernorm_a.weight"] = _random(rng, (c_s,))
    state[f"{prefix}.layernorm_a.bias"] = _random(rng, (c_s,))
    state[f"{prefix}.layernorm_z.weight"] = _random(rng, (c_z,))
    state[f"{prefix}.layernorm_z.bias"] = _random(rng, (c_z,))
    state[f"{prefix}.linear_nobias_z.weight"] = _random(rng, (heads, c_z))
    for name in ("linear_q", "linear_k", "linear_v", "linear_o", "linear_g"):
        state[f"{prefix}.attention.{name}.weight"] = _random(rng, (c_s, c_s))
    state[f"{prefix}.attention.linear_q.bias"] = _random(rng, (c_s,))


def _random(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    return rng.normal(scale=0.2, size=shape).astype(np.float32)
