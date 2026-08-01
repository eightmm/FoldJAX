from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.models.diffusion_transformer import (
    diffusion_transformer_block,
    diffusion_transformer_stack,
    map_diffusion_transformer,
)

pytestmark = pytest.mark.official_parity

@pytest.fixture(scope="module")
def diffusion_module(official_asset_path):
    torch = pytest.importorskip("torch")
    path = official_asset_path("diffusion_module.pt")
    return torch.jit.load(str(path), map_location="cpu").eval()


@pytest.fixture(scope="module")
def transformer_state(diffusion_module):
    return {
        key: value.detach().cpu().numpy()
        for key, value in diffusion_module.state_dict().items()
        if key.startswith("diffusion_transformer.")
    }


@pytest.fixture(scope="module")
def transformer_params(transformer_state):
    return map_diffusion_transformer(transformer_state)


def _inputs(*, samples: int, tokens: int, seed: int):
    rng = np.random.default_rng(seed)
    single = rng.normal(size=(1, samples, tokens, 768)).astype(np.float32)
    cond = rng.normal(size=(1, samples, tokens, 384)).astype(np.float32)
    pair = rng.normal(size=(1, tokens, tokens, 256)).astype(np.float32)
    token_mask = np.ones((1, tokens), dtype=np.bool_)
    token_mask[:, -1] = False
    return single, cond, pair, token_mask


def test_mapping_covers_16_blocks_and_all_224_tensors(
    transformer_state, transformer_params
) -> None:
    assert len(transformer_state) == 224
    assert len(transformer_params.blocks) == 16
    assert all(len(jax.tree.leaves(block)) == 14 for block in transformer_params.blocks)
    assert len(jax.tree.leaves(transformer_params)) == 224


def _torch_block(block, single, cond, pair, token_mask):
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    batch, samples, tokens, _ = single.shape

    pair_normalized = functional.layer_norm(
        pair,
        (pair.shape[-1],),
        block.pair_layer_norm.weight,
        block.pair_layer_norm.bias,
    )
    pair_bias = functional.linear(pair_normalized, block.pair_linear.weight)
    pair_bias = pair_bias.permute(0, 3, 1, 2)[:, :, None]
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    pair_bias = pair_bias.masked_fill(~pair_mask[:, None, None], -10000.0)

    normalized = functional.layer_norm(single, (single.shape[-1],), eps=0.1)
    scale, shift = functional.linear(cond, block.norm_in.lin_s_merged.weight).chunk(
        2, -1
    )
    normalized = normalized * (scale + 1.0) + shift
    qkv = block.to_qkv(normalized)
    qkv = qkv.reshape(batch, samples, tokens, 16, 144).permute(0, 3, 1, 2, 4)
    q, k, v = qkv.chunk(3, -1)
    q = q + block.q_bias.reshape(1, 16, 1, 1, 48)
    attended = functional.scaled_dot_product_attention(q, k, v, pair_bias)
    attended = attended.permute(0, 2, 3, 1, 4).reshape(batch, samples, tokens, 768)
    attention_update = block.gate_proj(cond) * block.to_out(attended)

    transition = block.transition
    normalized = transition.ada_ln(single, cond)
    value, gate = functional.linear(
        normalized, transition.linear_a_nobias_double.weight
    ).chunk(2, -1)
    hidden = functional.silu(value) * gate
    transition_update = functional.linear(hidden, transition.linear_b_nobias.weight)
    transition_update = transition_update * torch.sigmoid(
        functional.linear(
            cond,
            transition.linear_s_biasinit_m2.weight,
            transition.linear_s_biasinit_m2.bias,
        )
    )
    return single + attention_update + transition_update


def test_one_block_matches_official_leaf_formula(
    diffusion_module, transformer_params
) -> None:
    torch = pytest.importorskip("torch")
    inputs = _inputs(samples=2, tokens=3, seed=47)
    torch_inputs = tuple(torch.from_numpy(value) for value in inputs)
    official_block = getattr(diffusion_module.diffusion_transformer.blocks, "0")
    with torch.no_grad():
        expected = _torch_block(official_block, *torch_inputs)

    actual = diffusion_transformer_block(
        *(jnp.asarray(value) for value in inputs), transformer_params.blocks[0]
    )
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=2e-3, atol=2e-3
    )


def test_static_query_chunks_match_unchunked(transformer_params) -> None:
    inputs = tuple(
        jnp.asarray(value) for value in _inputs(samples=2, tokens=3, seed=53)
    )
    expected = diffusion_transformer_block(*inputs, transformer_params.blocks[0])
    actual = diffusion_transformer_block(
        *inputs, transformer_params.blocks[0], query_chunk_size=2
    )
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-3, atol=1e-2
    )


def test_all_16_blocks_match_official_leaf_formula(
    diffusion_module, transformer_params
) -> None:
    torch = pytest.importorskip("torch")
    inputs = _inputs(samples=1, tokens=2, seed=59)
    torch_single, torch_cond, torch_pair, torch_mask = (
        torch.from_numpy(value) for value in inputs
    )
    with torch.no_grad():
        expected = torch_single
        for index in range(16):
            block = getattr(diffusion_module.diffusion_transformer.blocks, str(index))
            expected = _torch_block(block, expected, torch_cond, torch_pair, torch_mask)

    actual = diffusion_transformer_stack(
        *(jnp.asarray(value) for value in inputs), transformer_params
    )
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=3e-3, atol=3e-3
    )


def test_query_chunk_size_must_be_positive(transformer_params) -> None:
    inputs = tuple(
        jnp.asarray(value) for value in _inputs(samples=1, tokens=2, seed=61)
    )
    with pytest.raises(ValueError, match="query_chunk_size"):
        diffusion_transformer_block(
            *inputs, transformer_params.blocks[0], query_chunk_size=0
        )
